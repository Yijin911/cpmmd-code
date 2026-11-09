"""
Deployment-cost experiment: support claims 1+2 (storage + per-test cost)
of Corollary 4 (CP-MMD vs MMDAgg/MMD-FUSE).

NOT a total-wall-clock comparison. We measure ONLY:
  (1) Deployment storage = peak memory the trained method needs to remain
      ready for testing, after kernel selection finishes.
  (2) Per-test wall-clock = time to evaluate one new (X', Y') pair AT TEST
      TIME (training cost excluded).

Theory predicts: MMDAgg/MMD-FUSE scale as O(B * N^2) in both metrics; CP-MMD
stays O(N^2) regardless of B (only the selected kernel survives).

Setting: P = N(0, I_d), Q = N(delta * 1, I_d), simple Gaussian shift on which
all methods reach high power. Pure Gaussian kernel + bandwidth grid for
MMDAgg/MMD-FUSE (B in {5, 10, 20, 50, 100}). CP-MMD selects a single bandwidth
sigma via continuous gradient descent.

What we time/measure:
- Deployment memory: tracemalloc peak after the kernel-select stage, while
  the method holds whatever it needs to test on a new (X', Y').
- Per-test wall-clock: median over 30 trials of evaluating one new pair.
- Sample size n = 200, dimension d = 10, delta = 0.5.
"""

import os
import sys
import csv
import time
import gc
import tracemalloc
import warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, _to_tensor, _DEVICE, mmd_loss_torch,
)

try:
    from mmdagg import mmdagg as mmdagg_fn
    from mmdfuse import mmdfuse as mmdfuse_fn
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    HAS_AGG = True
except ImportError:
    HAS_AGG = False


# JAX-JIT permutation test for CP-MMD: implementation parity with MMD-FUSE.
# Optimization: precompute the full kernel matrix K_full = k(Z, Z) once, then
# permutations index into it instead of recomputing K from scratch. The math
# is identical (kernel values k(z_i, z_j) don't depend on the X/Y labelling)
# so power and Type-I are unchanged; only the wall-clock drops by ~N_PERM x.
if HAS_AGG:
    def _gaussian_K(Z, sigma):
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b avoids materialising the
        # N x N x D difference tensor (2.9 GB for degree-3 features of d = 28).
        sq_norm = jnp.sum(Z * Z, axis=1)
        sq = jnp.maximum(sq_norm[:, None] + sq_norm[None, :] - 2.0 * (Z @ Z.T), 0.0)
        return jnp.exp(-sq / (sigma ** 2))

    def _mmd_u_from_K(K_full, idx_x, idx_y):
        """Unbiased MMD^2 U-statistic computed by indexing into a precomputed
        full kernel matrix K_full."""
        K_xx = K_full[idx_x][:, idx_x]
        K_yy = K_full[idx_y][:, idx_y]
        K_xy = K_full[idx_x][:, idx_y]
        n_x = idx_x.shape[0]
        n_y = idx_y.shape[0]
        s_xx = (jnp.sum(K_xx) - jnp.trace(K_xx)) / (n_x * (n_x - 1))
        s_yy = (jnp.sum(K_yy) - jnp.trace(K_yy)) / (n_y * (n_y - 1))
        s_xy = jnp.sum(K_xy) / (n_x * n_y)
        return s_xx + s_yy - 2.0 * s_xy

    _N_PERM_JIT = 200  # Python int, captured statically when JIT traces.

    def _mmd_u_batched(K_full, A, m, n):
        """Unbiased MMD^2 for every labelling at once. Column p of the 0/1
        matrix A (N x P) marks the X-sample of labelling p; all P statistics
        come from the single product K_full @ A."""
        KA = K_full @ A
        KB = jnp.sum(K_full, axis=1)[:, None] - KA      # K_full @ (1 - A)
        B = 1.0 - A
        diag = jnp.diag(K_full)
        s_xx = jnp.sum(A * KA, axis=0) - A.T @ diag
        s_yy = jnp.sum(B * KB, axis=0) - B.T @ diag
        s_xy = jnp.sum(B * KA, axis=0)
        return s_xx / (m * (m - 1)) + s_yy / (n * (n - 1)) - 2.0 * s_xy / (m * n)

    @jax.jit
    def _jax_perm_test(X, Y, sigma, key):
        """JIT-compiled permutation test: the kernel matrix is computed once and
        all N_PERM permutation statistics are evaluated with one matrix product.
        The permutations are the same as in a per-permutation loop over
        jax.random.split(key, N_PERM), so the decision is unchanged."""
        Z = jnp.concatenate([X, Y], axis=0)
        m = X.shape[0]
        N = Z.shape[0]
        K_full = _gaussian_K(Z, sigma)
        obs = _mmd_u_from_K(K_full, jnp.arange(m), jnp.arange(m, N))
        keys = jax.random.split(key, _N_PERM_JIT)
        perms = jax.vmap(lambda k: jax.random.permutation(k, N))(keys)
        A = jnp.zeros((_N_PERM_JIT, N), K_full.dtype).at[
            jnp.arange(_N_PERM_JIT)[:, None], perms[:, :m]].set(1.0).T
        null = _mmd_u_batched(K_full, A, m, N - m)
        threshold = jnp.percentile(null, 95.0)
        return jnp.where(obs > threshold, 1, 0)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'deployment_cost')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N = 200            # samples per group (matches Higgs n=200 in Fig 3 panel a)
DIM = 28           # dimension matches Higgs (d=28)
DELTA = 0.5
N_PERM = 200
N_TRIALS = 30      # trials for per-test wall-clock
B_GRID = [5, 10, 20, 50, 100]
SIGMA_RANGE = (0.1, 50.0)  # widened for d=28
N_OPT_STEPS = 100
LR = 0.05


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def gen(n, d, delta, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32) + delta
    return X, Y


# ----------------------------------------------------------------------
# CP-MMD: continuous bandwidth selection via gradient descent on log sigma
# ----------------------------------------------------------------------
class CPMMDDeployedKernel:
    """A trained CP-MMD bandwidth selector. After construction, only sigma
    needs to be retained for testing."""

    def __init__(self, sigma):
        self.sigma = float(sigma)

    @classmethod
    def fit(cls, X, Y, c1=0.01, n_steps=N_OPT_STEPS, lr=LR, seed=0):
        torch.manual_seed(seed)
        log_sigma = nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam([log_sigma], lr=lr)
        Xt = _to_tensor(X)
        Yt = _to_tensor(Y)
        Z = np.concatenate([X, Y], axis=0)
        data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
        for _ in range(n_steps):
            opt.zero_grad()
            sigma = log_sigma.exp()
            sigma_c = torch.clamp(sigma, SIGMA_RANGE[0], SIGMA_RANGE[1])
            mmd2 = mmd_loss_torch(Xt, Yt, sigma_c)
            penalty = c1 * data_norm / sigma_c
            loss = -(mmd2 - penalty)
            loss.backward()
            opt.step()
        with torch.no_grad():
            sigma_final = float(torch.clamp(log_sigma.exp(),
                                            SIGMA_RANGE[0],
                                            SIGMA_RANGE[1]).item())
        return cls(sigma_final)

    def per_test(self, X, Y, n_perm=N_PERM, seed=0):
        # JIT-compiled JAX permutation test for implementation parity with
        # MMD-FUSE. The first call triggers a one-time JAX trace+compile;
        # subsequent calls run the compiled kernel.
        key = jr.PRNGKey(seed)
        return int(np.asarray(_jax_perm_test(
            jnp.asarray(X), jnp.asarray(Y), float(self.sigma), key)))


def _poly_features(X, degree):
    """Compute degree-d polynomial features (without intercept). Stable order
    via sorted multi-indices. For d=10, deg=3 yields 10+55+220=285 features."""
    n, d = X.shape
    feats = [X]
    if degree >= 2:
        # Degree-2: pairwise products x_i * x_j for i<=j
        idx_i, idx_j = np.triu_indices(d)
        feats.append(X[:, idx_i] * X[:, idx_j])
    if degree >= 3:
        # Degree-3: triples i<=j<=k
        triples = np.array([(i, j, k)
                            for i in range(d)
                            for j in range(i, d)
                            for k in range(j, d)])
        feats.append(X[:, triples[:, 0]] * X[:, triples[:, 1]] * X[:, triples[:, 2]])
    return np.concatenate(feats, axis=1).astype(np.float32)


class CPMMDPolyDeployed:
    """CP-MMD with polynomial-feature kernel: Gaussian on degree-d polynomial
    features. After construction, only sigma + degree need retention."""

    def __init__(self, sigma, degree):
        self.sigma = float(sigma)
        self.degree = int(degree)

    @classmethod
    def fit(cls, X, Y, degree=3, c1=0.01, n_steps=N_OPT_STEPS, lr=LR, seed=0):
        Xf = _poly_features(X, degree)
        Yf = _poly_features(Y, degree)
        # Reuse the linear bandwidth selector on transformed features.
        torch.manual_seed(seed)
        log_sigma = nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam([log_sigma], lr=lr)
        Xt = _to_tensor(Xf)
        Yt = _to_tensor(Yf)
        Z = np.concatenate([Xf, Yf], axis=0)
        data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
        for _ in range(n_steps):
            opt.zero_grad()
            sigma = log_sigma.exp()
            sigma_c = torch.clamp(sigma, SIGMA_RANGE[0], SIGMA_RANGE[1] * 10)
            mmd2 = mmd_loss_torch(Xt, Yt, sigma_c)
            penalty = c1 * data_norm / sigma_c
            loss = -(mmd2 - penalty)
            loss.backward()
            opt.step()
        with torch.no_grad():
            sigma_final = float(torch.clamp(log_sigma.exp(),
                                            SIGMA_RANGE[0],
                                            SIGMA_RANGE[1] * 10).item())
        return cls(sigma_final, degree)

    def per_test(self, X, Y, n_perm=N_PERM, seed=0):
        # Feature extraction is part of deployment cost.
        Xf = _poly_features(X, self.degree)
        Yf = _poly_features(Y, self.degree)
        key = jr.PRNGKey(seed)
        return int(np.asarray(_jax_perm_test(
            jnp.asarray(Xf), jnp.asarray(Yf), float(self.sigma), key)))


# ----------------------------------------------------------------------
# MMDAgg: must keep B bandwidths and run B permutation tests at deployment
# ----------------------------------------------------------------------
class MMDAggDeployed:
    def __init__(self, sigmas):
        self.sigmas = np.asarray(sigmas, dtype=np.float32)

    @classmethod
    def fit(cls, X, Y, B):
        # Bandwidth grid log-spaced over the same range CP-MMD searches.
        sigmas = np.geomspace(SIGMA_RANGE[0], SIGMA_RANGE[1], B)
        return cls(sigmas)

    # Permutations for MMDAgg's quantiles (B1) and level correction (B2):
    # N_PERM each, matching the N_PERM permutations of CP-MMD and MMD-FUSE.
    B1 = N_PERM
    B2 = N_PERM

    def per_test(self, X, Y, seed=0):
        # kernel='laplace_gaussian' uses BOTH Laplacian and Gaussian kernels
        # (Schrab et al.'s standard recommendation), giving 2*B kernels.
        return int(np.asarray(mmdagg_fn(
            X.astype(np.float32), Y.astype(np.float32),
            alpha=ALPHA, kernel='laplace_gaussian',
            number_bandwidths=len(self.sigmas),
            permutations_same_sample_size=True,
            seed=seed, B1=self.B1, B2=self.B2)))


class MMDAggDefaultDeployed(MMDAggDeployed):
    """MMDAgg at its package defaults (B1 = B2 = 2000, B3 = 50)."""
    B1 = 2000
    B2 = 2000


class MMDFuseDeployed:
    def __init__(self, B):
        self.B = B

    @classmethod
    def fit(cls, X, Y, B):
        return cls(B)

    def per_test(self, X, Y, seed=0):
        key = jr.PRNGKey(seed)
        return int(np.asarray(mmdfuse_fn(
            jnp.array(X), jnp.array(Y), key,
            alpha=ALPHA, number_bandwidths=self.B,
            number_permutations=N_PERM)))


# ----------------------------------------------------------------------
# Measurement: deployment memory + per-test wall-clock
# ----------------------------------------------------------------------
def measure_deployment(method_constructor, train_args, test_pairs):
    """Train the method, then measure peak memory + per-test wall clock."""
    # 1. Fit.
    method = method_constructor(**train_args)
    gc.collect()

    # 2. Memory of the deployed object (only what's kept after fit).
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    # Touch the method to retain references that matter at deployment.
    _ = method
    snapshot_after = tracemalloc.take_snapshot()
    deploy_mem_bytes = sum(
        s.size for s in snapshot_after.statistics('filename')
    )
    tracemalloc.stop()

    # 3. Per-test wall-clock: median over N_TRIALS evaluations.
    # First, do a single warmup call so any JIT compilation is excluded
    # from the timed measurements. The warmup uses the first test pair.
    Xw, Yw = test_pairs[0]
    _ = method.per_test(Xw, Yw, seed=-1)
    times = []
    for i, (Xt, Yt) in enumerate(test_pairs):
        gc.collect()
        t0 = time.perf_counter()
        _ = method.per_test(Xt, Yt, seed=i)
        times.append(time.perf_counter() - t0)
    return {
        'deploy_mem_bytes': deploy_mem_bytes,
        'per_test_median_s': float(np.median(times)),
        'per_test_p10_s': float(np.percentile(times, 10)),
        'per_test_p90_s': float(np.percentile(times, 90)),
    }


def main():
    log("=" * 70)
    log(f"Deployment-cost experiment: n={N}, d={DIM}, delta={DELTA}")
    log(f"B grid: {B_GRID}, trials per measurement: {N_TRIALS}")
    log("=" * 70)

    # Generate one fixed (X, Y) for fitting; N_TRIALS new pairs for testing.
    X, Y = gen(N, DIM, DELTA, seed=0)
    test_pairs = [gen(N, DIM, DELTA, seed=1000 + i) for i in range(N_TRIALS)]

    rows = []

    # --- CP-MMD: independent of B (single sigma) ---
    # The fit below only fixes the bandwidth of the timed tests; the per-test
    # cost does not depend on its value. The selection cost reported in
    # Sec. 4.4 (grid calibration + selection) is timed by run_selection_cost.py.
    log("\n--- CP-MMD-linear (single learned sigma, Gaussian kernel) ---")
    res = measure_deployment(
        CPMMDDeployedKernel.fit,
        train_args={'X': X, 'Y': Y, 'c1': 0.01, 'seed': 0},
        test_pairs=test_pairs,
    )
    log(f"  deploy_mem={res['deploy_mem_bytes']/1024:.0f} KB  "
        f"per_test={res['per_test_median_s']*1000:.1f} ms")
    rows.append({'method': 'CP-MMD-linear', 'B': 1, **res})

    # --- CP-MMD with degree-3 polynomial features + Gaussian ---
    log("\n--- CP-MMD-poly3 (degree-3 polynomial features + Gaussian) ---")
    res = measure_deployment(
        CPMMDPolyDeployed.fit,
        train_args={'X': X, 'Y': Y, 'degree': 3, 'c1': 0.01, 'seed': 0},
        test_pairs=test_pairs,
    )
    log(f"  deploy_mem={res['deploy_mem_bytes']/1024:.0f} KB  "
        f"per_test={res['per_test_median_s']*1000:.1f} ms")
    rows.append({'method': 'CP-MMD-poly3', 'B': 1, **res})

    # --- MMDAgg and MMD-FUSE for each B ---
    if HAS_AGG:
        for B in B_GRID:
            log(f"\n--- MMDAgg, B={B} ---")
            res = measure_deployment(
                MMDAggDeployed.fit,
                train_args={'X': X, 'Y': Y, 'B': B},
                test_pairs=test_pairs,
            )
            log(f"  deploy_mem={res['deploy_mem_bytes']/1024:.0f} KB  "
                f"per_test={res['per_test_median_s']*1000:.1f} ms")
            rows.append({'method': 'MMDAgg', 'B': B, **res})

            log(f"--- MMDAgg (package defaults B1=B2=2000, B3=50), B={B} ---")
            res = measure_deployment(
                MMDAggDefaultDeployed.fit,
                train_args={'X': X, 'Y': Y, 'B': B},
                test_pairs=test_pairs,
            )
            log(f"  per_test={res['per_test_median_s']*1000:.1f} ms")
            rows.append({'method': 'MMDAgg (default)', 'B': B, **res})

            log(f"--- MMD-FUSE, B={B} ---")
            res = measure_deployment(
                MMDFuseDeployed.fit,
                train_args={'X': X, 'Y': Y, 'B': B},
                test_pairs=test_pairs,
            )
            log(f"  deploy_mem={res['deploy_mem_bytes']/1024:.0f} KB  "
                f"per_test={res['per_test_median_s']*1000:.1f} ms")
            rows.append({'method': 'MMD-FUSE', 'B': B, **res})
    else:
        log("WARN: mmdagg / mmdfuse not installed; skipping baselines.")

    out = os.path.join(RESULTS_DIR, 'deployment_cost.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"\nSaved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
