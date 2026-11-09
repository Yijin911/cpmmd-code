"""
Linear pillar: CP-MMD bandwidth selection vs Median heuristic.

CP-MMD here uses the proper criterion
    J_CP(sigma) = gamma_hat^2(sigma) - C1_hat * G_tilde(sigma),
with G_tilde(sigma) = ||D||_F / (sigma * N) (linear regime, Appendix
proxy-bounds), and C1_hat the 95th percentile of permutation ratios
gamma_hat^2 / G_tilde under null permutations of the training half
(Prop. conformal-c1). PermSplit: select sigma on the train half, test
on the held-out half.

Setting: multi-scale Gaussian mixture. P has two well-separated clusters
of small-scale points; Q is the same but with a small mean shift. The
median pairwise distance is dominated by the inter-cluster separation,
which is much larger than the within-cluster scale on which the
discriminative signal lives. CP-MMD bandwidth selection finds the
within-cluster scale and detects the shift; median uses the wrong scale
and misses it.
"""

import os, sys, time, csv, warnings
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import compute_mmd_u_statistic, optimize_bandwidth, _DEVICE

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'linear_pillar')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 100
N_PERM = 200
N_SAMPLES = 200
BW_LOW, BW_HIGH = 0.05, 5.0  # broad search interval


# ----------------------------------------------------------------------
# Data generation: multi-scale mixture
# ----------------------------------------------------------------------

def generate_multiscale(n, d=2, delta=0.0, seed=None):
    """
    P: 50% N(0, 0.01 I_d) + 50% N((3,0,...,0), 0.01 I_d)
    Q: 50% N(delta*e_1, 0.01 I_d) + 50% N((3+delta,0,...,0), 0.01 I_d)

    Inter-cluster gap is 3 (sets the median scale).
    Within-cluster scale is 0.1 (sets the discriminative scale).
    delta = mean shift along e_1; small delta requires the small bandwidth.
    """
    rng = np.random.default_rng(seed)
    cluster_std = 0.1
    cluster_sep = 3.0

    def sample(mean_offset):
        labels = rng.integers(0, 2, size=n)
        centers = np.zeros((n, d))
        centers[labels == 1, 0] = cluster_sep
        return centers + cluster_std * rng.standard_normal((n, d)) + mean_offset

    P = sample(np.zeros(d))
    shift = np.zeros(d); shift[0] = delta
    Q = sample(shift)
    return P.astype(np.float32), Q.astype(np.float32)


def median_pairwise(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    n = len(Z)
    diffs = Z[:, None, :] - Z[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    return float(np.median(dists[iu]))


# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------

def mmd_at_sigma(X, Y, sigma):
    return float(compute_mmd_u_statistic(X, Y, kernel_type='gaussian', sigma=sigma))


def _G_tilde_linear(sigma, X, Y):
    """G_tilde(sigma) = ||D||_F / (sigma * N), linear regime (Appendix proxy-bounds)."""
    pooled = np.concatenate([X, Y], axis=0)
    DF = float(np.linalg.norm(pooled))
    N = pooled.shape[0]
    return DF / (sigma * N)


def calibrate_C1_linear(X, Y, sigma_grid, n_perm=10, seed=None):
    """C1 = 95th percentile of (best-J_CP-relevant ratio) over null permutations."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([X, Y], axis=0)
    m, n = len(X), len(Y)
    ratios = []
    for _ in range(n_perm):
        perm = rng.permutation(m + n)
        Xp, Yp = pooled[perm[:m]], pooled[perm[m:]]
        best = -float('inf')
        for s in sigma_grid:
            mmd2 = max(mmd_at_sigma(Xp, Yp, s), 0.0)
            G = _G_tilde_linear(s, Xp, Yp)
            r = mmd2 / G if G > 0 else 0.0
            if r > best:
                best = r
        ratios.append(max(best, 0.0))
    return float(np.quantile(ratios, 0.95))


def cpmmd_select(X, Y, seed=None):
    """CP-MMD: argmax over sigma of J_CP(sigma) = mmd^2(sigma) - C1_hat * G_tilde(sigma).

    Calibration of C1_hat uses null permutations of (X, Y) per Prop. conformal-c1.
    Selection over a fixed geometric grid spanning [BW_LOW, BW_HIGH]; falls back
    to the grid argmax which is consistent with the deployed criterion.
    """
    sigma_grid = list(np.geomspace(BW_LOW, BW_HIGH, 24))
    C1 = calibrate_C1_linear(X, Y, sigma_grid, n_perm=10, seed=seed)
    best_sigma = sigma_grid[0]
    best_J = -float('inf')
    for s in sigma_grid:
        mmd2 = mmd_at_sigma(X, Y, s)
        G = _G_tilde_linear(s, X, Y)
        J = mmd2 - C1 * G
        if J > best_J:
            best_J = J
            best_sigma = s
    return best_sigma, best_J


def median_select(X, Y):
    sigma = max(median_pairwise(X, Y), 1e-6)
    return sigma, mmd_at_sigma(X, Y, sigma)


def permutation_test(X, Y, sigma, n_perm=N_PERM, alpha=ALPHA, seed=None):
    rng = np.random.default_rng(seed)
    obs = mmd_at_sigma(X, Y, sigma)
    pooled = np.concatenate([X, Y], axis=0)
    m, n = len(X), len(Y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(m + n)
        Xp, Yp = pooled[perm[:m]], pooled[perm[m:]]
        null[k] = mmd_at_sigma(Xp, Yp, sigma)
    threshold = float(np.percentile(null, (1 - alpha) * 100))
    return int(obs > threshold), obs, threshold


def cpmmd_test_split(X, Y, n_perm=N_PERM, alpha=ALPHA, seed=None):
    """
    CP-MMD with sample splitting:
        select sigma on first half, test on second half.
    This is the honest comparison vs. Median (which uses all data).
    """
    n = len(X) // 2
    Xtr, Ytr = X[:n], Y[:n]
    Xte, Yte = X[n:], Y[n:]
    sigma, _ = cpmmd_select(Xtr, Ytr, seed=None if seed is None else seed + 1)
    return permutation_test(Xte, Yte, sigma, n_perm=n_perm, alpha=alpha,
                            seed=seed) + (sigma,)


def median_test(X, Y, n_perm=N_PERM, alpha=ALPHA, seed=None):
    sigma, _ = median_select(X, Y)
    return permutation_test(X, Y, sigma, n_perm=n_perm, alpha=alpha,
                            seed=seed) + (sigma,)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LINEAR PILLAR: CP-MMD bandwidth selection vs Median heuristic")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {_DEVICE}")
    print(f"N_REPS={N_REPS}, N_PERM={N_PERM}, N_SAMPLES={N_SAMPLES}")
    print(f"Bandwidth search: [{BW_LOW}, {BW_HIGH}]")
    print("=" * 70)

    deltas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]  # 0 = H0, rest = H1

    all_rows = []
    for delta in deltas:
        is_null = (delta == 0.0)
        label = "TYPE_I" if is_null else "POWER"
        print(f"\n--- delta={delta:.2f} ({label}) ---")

        cpmmd_count = 0
        median_count = 0
        cpmmd_sigmas = []
        median_sigmas = []
        n_total = N_SAMPLES  # every method receives N_SAMPLES per group; CP-MMD splits it in half

        t0 = time.time()
        for rep in range(N_REPS):
            seed = rep * 7919 + int(delta * 10000)
            X, Y = generate_multiscale(n_total, d=2, delta=delta, seed=seed)
            try:
                rej, _, _, sig = cpmmd_test_split(X, Y, seed=seed)
                cpmmd_count += rej
                cpmmd_sigmas.append(sig)
            except Exception as e:
                pass

            try:
                Xh, Yh = X, Y
                rej, _, _, sig = median_test(Xh, Yh, seed=seed)
                median_count += rej
                median_sigmas.append(sig)
            except Exception as e:
                pass

        dt = time.time() - t0
        cpmmd_rate = cpmmd_count / N_REPS
        median_rate = median_count / N_REPS
        cp_med_sigma = float(np.median(cpmmd_sigmas)) if cpmmd_sigmas else float('nan')
        med_med_sigma = float(np.median(median_sigmas)) if median_sigmas else float('nan')
        print(f"  CP-MMD bw-select : {label}={cpmmd_rate:.3f}  "
              f"(median sigma={cp_med_sigma:.3f})")
        print(f"  Median heuristic : {label}={median_rate:.3f}  "
              f"(median sigma={med_med_sigma:.3f})")
        print(f"  ({dt:.1f}s)")

        all_rows.append({
            'delta': delta, 'hypothesis': 'H0' if is_null else 'H1',
            'method': 'CPMMD_bw_select', 'rate': cpmmd_rate,
            'median_sigma': cp_med_sigma, 'n_reps': N_REPS,
        })
        all_rows.append({
            'delta': delta, 'hypothesis': 'H0' if is_null else 'H1',
            'method': 'Median_heuristic', 'rate': median_rate,
            'median_sigma': med_med_sigma, 'n_reps': N_REPS,
        })

    out_path = os.path.join(RESULTS_DIR, 'linear_pillar_results.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nSaved {out_path}")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
