"""
Killer experiment: high-dimensional sparse covariance shift.

Setting designed so that ONLY the complexity-penalized criterion succeeds,
while all alternatives fail.

P = N(0, I_d) with d=100
Q = N(0, Sigma) where Sigma = I + eta * v * v^T
  v is sparse: v_i = 1/sqrt(10) for i in {1,...,10}, 0 otherwise
  eta controls signal strength

Why this is hard:
  - Median heuristic fails: difference is in ONE covariance direction out of 100
  - Plain fails: d/n=1, deep network overfits to noise
  - Liu fails: variance collapse -- optimizer drives both MMD and sigma_hat to zero
  - CP-MMD succeeds: complexity penalty prevents overfitting while deep kernel learns
    the sparse direction

Methods:
  1. CPMMD_1 + PermSplit_2:  complexity-penalized deep kernel
  2. Liu_1   + PermSplit_2:  variance-normalized deep kernel
  3. Plain_1 + PermSplit_2:  unregularized deep kernel
  4. Median  + Perm:         median heuristic + full permutation
  5. MMDAgg  + Perm:         bandwidth aggregation (Bonferroni, 10 bandwidths)
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, train_discriminator,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)
from calibration_engine import compute_S

warnings.filterwarnings('ignore')

# ======================================================================
# Configuration
# ======================================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'killer')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
N_PERM = 200
DIM = 100
N_SAMPLES = 100  # n = m = 100, so d/n = 1

ETAS = [0, 0.3, 0.5, 0.8, 1.0]

LOG_FILE = os.path.join(RESULTS_DIR, 'killer_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ======================================================================
# Data generation
# ======================================================================
def generate_sparse_cov_shift(n, d, eta, seed=None):
    """P = N(0, I_d), Q = N(0, I + eta*v*v^T) with sparse v."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    # Sparse direction
    v = np.zeros(d)
    v[:10] = 1.0 / np.sqrt(10)
    Sigma = np.eye(d) + eta * np.outer(v, v)
    L = np.linalg.cholesky(Sigma)
    Y = rng.randn(n, d) @ L.T
    return X, Y


# ======================================================================
# Complexity measures
# ======================================================================
def compute_G_spectral(model, X, Y, sigma):
    spectral_product = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return spectral_product * np.linalg.norm(Z) / (np.sqrt(N) * sigma)


def compute_G_bandwidth(X, Y, sigma):
    """G for bandwidth-only optimization (no network)."""
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return np.linalg.norm(Z) / (np.sqrt(N) * sigma)


def median_heuristic(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z)
    dists = _sq_euclidean_distances(Z_t, Z_t)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    return max(torch.sqrt(dists[mask]).median().item(), 1e-6)


# ======================================================================
# Step 1: Kernel/model selection methods
# ======================================================================
def step1_cpmmd(X, Y, dim, C1):
    """CP-MMD criterion: J = MMD^2 - C1*G_h (deep kernel with complexity penalty)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        # Differentiable G_spectral
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        Z = np.concatenate([X, Y], axis=0)
        data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
        G_h = spec_prod * data_geom / sigma
        J = mmd2 - C1 * G_h
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D, 'mode': 'deep'}


def step1_liu(X, Y, dim):
    """Liu et al.'s criterion: J = MMD^2 / sigma_hat (deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        # Variance estimate
        m = f_x.shape[0]
        K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
        K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
        h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
        var_h = torch.var(h_x).clamp(min=1e-10)
        sigma_hat = torch.sqrt(4 * var_h / m)
        J = mmd2 / sigma_hat
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D, 'mode': 'deep'}


def step1_plain(X, Y, dim):
    """Plain MMD maximization (no regularization, deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        (-mmd2).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D, 'mode': 'deep'}


def step1_median(X, Y, dim):
    """Median heuristic (no optimization)."""
    sigma = median_heuristic(X, Y)
    mmd_val = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
    return {'sigma': sigma, 'mmd': mmd_val, 'model': None, 'mode': 'bandwidth'}


def step1_mmdagg(X, Y, dim):
    """MMDAgg bandwidth aggregation (Bonferroni, 10 bandwidths)."""
    sigmas = np.logspace(-1, 1, 10)
    m = len(X)
    pooled = np.concatenate([X, Y], axis=0)
    N = m + len(Y)
    # For each sigma, compute MMD and permutation p-value
    p_values = []
    for sigma in sigmas:
        mmd_obs = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(N)
            null_mmds.append(compute_mmd_u_statistic(
                pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))
        p_val = np.mean(np.array(null_mmds) >= mmd_obs)
        p_values.append(p_val)
    # Bonferroni correction
    min_p = min(p_values) * len(sigmas)
    reject = min_p < ALPHA
    return {'decision': int(reject), 'p_value': min_p, 'mode': 'aggregation'}


# ======================================================================
# Step 2: Hypothesis testing methods
# ======================================================================
def step2_perm_split(X, Y, step1_result, dim):
    """Permutation on held-out 50% (Liu et al.'s approach)."""
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_te, Y_te = X[m_tr:], Y[n_tr:]

    if step1_result['model'] is not None:
        model = step1_result['model']
        sigma = step1_result['sigma']
        X_te_t = _to_tensor(X_te).to(_DEVICE)
        Y_te_t = _to_tensor(Y_te).to(_DEVICE)
        with torch.no_grad():
            f_all = model(torch.cat([X_te_t, Y_te_t]))
        m_te, n_te = len(X_te), len(Y_te)
        test_stat = mmd_loss_torch(f_all[:m_te], f_all[m_te:], sigma).item()
        # Permutation null
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(m_te + n_te)
            null_mmds.append(mmd_loss_torch(f_all[perm[:m_te]], f_all[perm[m_te:]], sigma).item())
    else:
        sigma = step1_result['sigma']
        m_te, n_te = len(X_te), len(Y_te)
        test_stat = compute_mmd_u_statistic(X_te, Y_te, 'gaussian', sigma=sigma)
        pooled_te = np.concatenate([X_te, Y_te], axis=0)
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(m_te + n_te)
            null_mmds.append(compute_mmd_u_statistic(
                pooled_te[perm[:m_te]], pooled_te[perm[m_te:]], 'gaussian', sigma=sigma))

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


def step2_perm_full(X, Y, step1_result):
    """Full permutation test (no split) -- for Median heuristic."""
    m, n = len(X), len(Y)
    sigma = step1_result['sigma']
    test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
    pooled = np.concatenate([X, Y], axis=0)
    N = m + n
    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(N)
        null_mmds.append(compute_mmd_u_statistic(
            pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))
    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


# ======================================================================
# Calibration
# ======================================================================
def calibrate_C1(dim, n_samples, n_reps=10):
    """Calibrate C1 (95th percentile of MMD/G_ref) from null permutations.

    Journal procedure (shared G_ref): G is computed ONCE on the original
    un-permuted null data by training a fresh deep kernel on it, and is
    reused as the denominator across all permutations. Only the numerator
    (max-MMD on the permuted data) varies. This makes the calibration
    ratio exchangeable under the null permutation distribution
    (cf. Prop 4.23 in the JMLR draft).
    """
    log("  Calibrating C1 from null permutations (shared G_ref)...")
    # Generate null data: P = Q = N(0, I_d)
    X, Y = generate_sparse_cov_shift(n_samples, dim, eta=0.0, seed=42)
    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)

    # Shared G_ref: train once on un-permuted (X, Y)
    res_ref = step1_plain(X, Y, dim)
    G_ref = compute_G_spectral(res_ref['model'], X, Y, res_ref['sigma'])
    if abs(G_ref) <= 1e-12:
        G_ref = 1e-12

    ratios = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = step1_plain(X_p, Y_p, dim)
        ratios.append(res['mmd'] / G_ref)

    C1 = float(np.percentile(ratios, 95)) if ratios else 0.01
    log(f"  C1 = {C1:.6f}")
    return C1


# ======================================================================
# Main experiment
# ======================================================================
def run_killer_experiment():
    log(f"{'='*70}")
    log(f"KILLER EXPERIMENT: High-dimensional sparse covariance shift")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"d={DIM}, n=m={N_SAMPLES}, d/n={DIM/N_SAMPLES:.1f}")
    log(f"etas={ETAS}")
    log(f"N_REPS={N_REPS}, N_PERM={N_PERM}, N_STEPS={N_STEPS}, LR={LR}")
    log(f"Architecture: {DIM} -> {HIDDEN_WIDTH} -> {HIDDEN_WIDTH} -> {FEATURE_DIM}")
    log(f"{'='*70}")

    # Calibrate C1
    C1 = calibrate_C1(DIM, N_SAMPLES, n_reps=10)

    # Define methods
    methods = {
        'CPMMD_1+PermSplit_2': None,
        'Liu_1+PermSplit_2': None,
        'Plain_1+PermSplit_2': None,
        'Median+Perm': None,
        'MMDAgg+Perm': None,
    }

    all_results = []

    for eta in ETAS:
        log(f"\n{'='*70}")
        log(f"  eta = {eta}")
        log(f"{'='*70}")

        for method_name in methods:
            rejects = 0
            valid = 0
            t0 = time.time()

            for rep in range(N_REPS):
                seed = rep * 1000 + int(eta * 1000)
                np.random.seed(seed)
                torch.manual_seed(seed)

                X, Y = generate_sparse_cov_shift(N_SAMPLES, DIM, eta, seed=seed)

                try:
                    if method_name == 'CPMMD_1+PermSplit_2':
                        # Train on first half, test on second half
                        m_tr = len(X) // 2
                        n_tr = len(Y) // 2
                        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                        s1_result = step1_cpmmd(X_tr, Y_tr, DIM, C1)
                        s2_result = step2_perm_split(X, Y, s1_result, DIM)
                        rejects += int(s2_result['reject'])
                        valid += 1

                    elif method_name == 'Liu_1+PermSplit_2':
                        m_tr = len(X) // 2
                        n_tr = len(Y) // 2
                        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                        s1_result = step1_liu(X_tr, Y_tr, DIM)
                        s2_result = step2_perm_split(X, Y, s1_result, DIM)
                        rejects += int(s2_result['reject'])
                        valid += 1

                    elif method_name == 'Plain_1+PermSplit_2':
                        m_tr = len(X) // 2
                        n_tr = len(Y) // 2
                        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                        s1_result = step1_plain(X_tr, Y_tr, DIM)
                        s2_result = step2_perm_split(X, Y, s1_result, DIM)
                        rejects += int(s2_result['reject'])
                        valid += 1

                    elif method_name == 'Median+Perm':
                        s1_result = step1_median(X, Y, DIM)
                        s2_result = step2_perm_full(X, Y, s1_result)
                        rejects += int(s2_result['reject'])
                        valid += 1

                    elif method_name == 'MMDAgg+Perm':
                        res = step1_mmdagg(X, Y, DIM)
                        rejects += int(res['decision'])
                        valid += 1

                except Exception as e:
                    if rep == 0:
                        log(f"    WARNING: {method_name} rep 0 error: {e}")

            dt = time.time() - t0
            rate = rejects / max(valid, 1)
            label = "TYPE_I" if eta == 0 else "POWER"
            log(f"    {method_name:25s}: {label}={rate:.3f} ({valid}/{N_REPS} valid, {dt:.1f}s)")

            all_results.append({
                'eta': eta,
                'method': method_name,
                'rejection_rate': rate,
                'n_valid': valid,
                'n_reps': N_REPS,
                'time_s': round(dt, 1),
            })

        # Save intermediate results after each eta
        save_results(all_results)

    return all_results


def save_results(results):
    filepath = os.path.join(RESULTS_DIR, 'killer_results.csv')
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    log(f"  Saved {filepath}")


def print_summary(results):
    """Print a nice summary table."""
    log(f"\n{'='*70}")
    log(f"SUMMARY: Power curve (rejection rate)")
    log(f"{'='*70}")

    methods = list(dict.fromkeys(r['method'] for r in results))
    etas = list(dict.fromkeys(r['eta'] for r in results))

    # Header
    header = f"{'eta':>6s}"
    for m in methods:
        header += f"  {m:>25s}"
    log(header)
    log("-" * len(header))

    # Rows
    for eta in etas:
        row = f"{eta:6.2f}"
        for m in methods:
            rate = [r['rejection_rate'] for r in results if r['eta'] == eta and r['method'] == m]
            if rate:
                row += f"  {rate[0]:25.3f}"
            else:
                row += f"  {'N/A':>25s}"
        log(row)

    log(f"\n  eta=0: Type I error (should be <= {ALPHA})")
    log(f"  eta>0: Power (higher is better)")


def main():
    results = run_killer_experiment()
    print_summary(results)
    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
