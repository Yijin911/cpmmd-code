"""
Blob dataset theta sweep: find the Goldilocks zone where deep J_CP beats
fixed-kernel methods.

Sweep theta in [0.05, 0.1, 0.15, 0.2, 0.3] with n=200, 50 reps.
All 6 methods: CPMMD, Liu, Plain, Median, MMDAgg, MMDFUSE.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)

try:
    from mmdagg import mmdagg as mmdagg_fn
    from mmdfuse import mmdfuse as mmdfuse_fn
    import jax.random
    HAS_AGG = True
except ImportError:
    HAS_AGG = False

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'blob_sweep')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 200
N_PERM = 200
N_SAMPLES = 200
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
N_C1_REPS = 10
SEED_BASE = 42
DIM = 2

THETAS = [0.05, 0.1, 0.15, 0.2, 0.3]

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    log_fh.write(line + '\n')
    log_fh.flush()


def generate_blob(n, theta=0.0, seed=None):
    rng = np.random.default_rng(seed)
    centers = np.array([(i, j) for i in range(3) for j in range(3)], dtype=np.float64)
    center_of_mass = np.array([1.0, 1.0])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    centers_q = (centers - center_of_mass) @ R.T + center_of_mass
    std = np.sqrt(0.03)
    assignments_p = rng.integers(0, 9, size=n)
    X = centers[assignments_p] + rng.normal(0, std, size=(n, 2))
    assignments_q = rng.integers(0, 9, size=n)
    Y = centers_q[assignments_q] + rng.normal(0, std, size=(n, 2))
    return X.astype(np.float32), Y.astype(np.float32)


def median_heuristic(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z)
    dists = _sq_euclidean_distances(Z_t, Z_t)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    return max(torch.sqrt(dists[mask]).median().item(), 1e-6)


def compute_G_spectral(model, X, Y, sigma):
    spectral_product = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return spectral_product * np.linalg.norm(Z) / (np.sqrt(N) * sigma)


def permutation_threshold_deep(X, Y, model, sigma, n_perm=N_PERM):
    m, n = len(X), len(Y)
    pooled = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(pooled).to(_DEVICE)
    with torch.no_grad():
        f_all = model(Z_t)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        f_x = f_all[perm[:m]]
        f_y = f_all[perm[m:m+n]]
        null_mmds[i] = mmd_loss_torch(f_x, f_y, sigma).item()
    return np.percentile(null_mmds, (1 - ALPHA) * 100)


def permutation_threshold_fixed(X, Y, sigma, n_perm=N_PERM):
    m, n = len(X), len(Y)
    pooled = np.concatenate([X, Y], axis=0)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        Xp = pooled[perm[:m]]
        Yp = pooled[perm[m:m+n]]
        null_mmds[i] = compute_mmd_u_statistic(Xp, Yp, 'gaussian', sigma=sigma)
    return np.percentile(null_mmds, (1 - ALPHA) * 100)


def train_plain(X, Y):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
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
    return D, sigma_val, mmd_val


def train_liu(X, Y):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
        K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
        h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
        var_h = torch.var(h_x).clamp(min=1e-10)
        sigma_hat = torch.sqrt(4 * var_h / f_x.shape[0])
        J = mmd2 / (sigma_hat + 1e-8)
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return D, sigma_val, mmd_val


def train_cpmmd(X, Y, C1):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = np.concatenate([X, Y], axis=0)
    data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
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
    return D, sigma_val, mmd_val


def calibrate_C1():
    log(f"  Calibrating C1 with {N_C1_REPS} null permutations...")
    X, Y = generate_blob(N_SAMPLES, theta=0.0, seed=999)
    pooled = np.concatenate([X, Y], axis=0)
    ratios = []
    for rep in range(N_C1_REPS):
        perm = np.random.permutation(2 * N_SAMPLES)
        X_p = pooled[perm[:N_SAMPLES]]
        Y_p = pooled[perm[N_SAMPLES:2*N_SAMPLES]]
        D, sigma_val, mmd_val = train_plain(X_p, Y_p)
        G_h = compute_G_spectral(D, X_p, Y_p, sigma_val)
        if abs(G_h) > 1e-10:
            ratios.append(mmd_val / G_h)
    C1 = float(np.percentile(ratios, 95)) if ratios else 0.01
    log(f"  C1 = {C1:.6f}")
    return C1


def run_one_rep(theta, C1, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = {}

    n_total = N_SAMPLES * 2
    X_all, Y_all = generate_blob(n_total, theta=theta, seed=seed)
    X_tr, X_te = X_all[:N_SAMPLES], X_all[N_SAMPLES:]
    Y_tr, Y_te = Y_all[:N_SAMPLES], Y_all[N_SAMPLES:]

    # Deep methods: train on first half, test on second half
    for name, train_fn in [('CPMMD+PermSplit', lambda Xr, Yr: train_cpmmd(Xr, Yr, C1)),
                            ('Liu+PermSplit', lambda Xr, Yr: train_liu(Xr, Yr)),
                            ('Plain+PermSplit', lambda Xr, Yr: train_plain(Xr, Yr))]:
        try:
            D, sigma, mmd_val = train_fn(X_tr, Y_tr)
            X_te_t = _to_tensor(X_te).to(_DEVICE)
            Y_te_t = _to_tensor(Y_te).to(_DEVICE)
            with torch.no_grad():
                f_x, f_y = D(X_te_t), D(Y_te_t)
                test_mmd = mmd_loss_torch(f_x, f_y, sigma).item()
            threshold = permutation_threshold_deep(X_te, Y_te, D, sigma)
            results[name] = int(test_mmd > threshold)
        except Exception as e:
            results[name] = 0

    # Fixed kernel methods: use n samples (no split)
    X_full = X_all[:N_SAMPLES]
    Y_full = Y_all[:N_SAMPLES]

    try:
        sigma_med = median_heuristic(X_full, Y_full)
        mmd_obs = compute_mmd_u_statistic(X_full, Y_full, 'gaussian', sigma=sigma_med)
        threshold = permutation_threshold_fixed(X_full, Y_full, sigma_med)
        results['Median+Perm'] = int(mmd_obs > threshold)
    except:
        results['Median+Perm'] = 0

    try:
        result = mmdagg_fn(X_full, Y_full, alpha=ALPHA,
                           kernel='laplace_gaussian', number_bandwidths=10,
                           permutations_same_sample_size=True, seed=seed)
        results['MMDAgg+Perm'] = int(result)
    except:
        results['MMDAgg+Perm'] = 0

    try:
        key = jax.random.PRNGKey(seed)
        result = mmdfuse_fn(X_full, Y_full, key, alpha=ALPHA,
                            number_bandwidths=10, number_permutations=N_PERM)
        results['MMDFUSE+Perm'] = int(result)
    except:
        results['MMDFUSE+Perm'] = 0

    return results


def run_experiment():
    log("=" * 70)
    log("Blob Theta Sweep Experiment")
    log(f"thetas={THETAS}, n={N_SAMPLES}, reps={N_REPS}")
    log("=" * 70)

    C1 = calibrate_C1()

    methods = ['CPMMD+PermSplit', 'Liu+PermSplit', 'Plain+PermSplit',
               'Median+Perm', 'MMDAgg+Perm', 'MMDFUSE+Perm']

    all_results = []

    for theta in THETAS:
        log(f"\n{'='*60}")
        log(f"theta = {theta:.4f}")
        log(f"{'='*60}")

        counts = {m: 0 for m in methods}

        for rep in range(N_REPS):
            seed = SEED_BASE + rep + int(theta * 10000)
            t_rep = time.time()
            res = run_one_rep(theta, C1, seed)
            for m in methods:
                counts[m] += res.get(m, 0)

            if (rep + 1) % 10 == 0 or rep == 0:
                elapsed = time.time() - t_rep
                log(f"  rep {rep+1}/{N_REPS} ({elapsed:.1f}s)")
                for m in methods:
                    rate = counts[m] / (rep + 1)
                    log(f"    {m:25s}: {rate:.3f} ({counts[m]}/{rep+1})")

        log(f"\n--- Final Power (theta={theta:.4f}) ---")
        for m in methods:
            val = counts[m] / N_REPS
            log(f"  {m:25s}: Power = {val:.3f} ({counts[m]}/{N_REPS})")
            all_results.append({
                'theta': theta, 'method': m, 'power': val,
                'rejects': counts[m], 'reps': N_REPS, 'C1': C1,
            })

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'blob_sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    log(f"\nResults saved to {csv_path}")

    # Summary table
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    header = f"{'theta':>8s}"
    for m in methods:
        header += f" {m:>15s}"
    log(header)
    log("-" * (8 + 16 * len(methods)))
    for theta in THETAS:
        row = f"{theta:>8.3f}"
        for m in methods:
            r = [x for x in all_results if x['theta'] == theta and x['method'] == m]
            row += f" {r[0]['power']:>15.3f}" if r else f" {'N/A':>15s}"
        log(row)
    log("=" * 70)

    return all_results


if __name__ == '__main__':
    run_experiment()
    log_fh.close()
