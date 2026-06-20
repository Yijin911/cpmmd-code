"""
Phase 2c: Computational cost experiment.

Measures wall-clock time for each method on the synthetic benchmark (d=50, m=n=100).

For each method, decomposes timing into:
  - Calibration time (for CP-MMD only)
  - Selection time (training/optimization)
  - Test time (permutation threshold)
  - Total

Methods: CPMMD_1+PermSplit_2, Liu_1+PermSplit_2, Plain_1+PermSplit_2, Median+Perm, MMDAgg+Perm

Also measures scaling with n: run at n in {100, 200, 500} and report total time.

Uses 10 repetitions. Reports mean and std. Saves to results/timing/timing_results.csv.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, train_discriminator,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)
from calibration_engine import compute_S
from data_generator import generate_data

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'timing')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 10
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
N_PERM = 200

LOG_FILE = os.path.join(RESULTS_DIR, 'timing_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)


def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ======================================================================
# Complexity measures (copied inline from run_cross_comparison.py)
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
# Step 1: Kernel/model selection methods (copied inline)
# ======================================================================
def step1_cpmmd(X, Y, dim, C1, mode='bandwidth'):
    """CP-MMD criterion: J = MMD^2 - C1*G_h."""
    if mode == 'bandwidth':
        search_sigmas = np.logspace(-1, 1, 20)
        best_J = -float('inf')
        best_sigma = 1.0
        best_mmd = 0.0
        for sigma in search_sigmas:
            mmd_val = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
            G = compute_G_bandwidth(X, Y, sigma)
            J = mmd_val - C1 * G
            if J > best_J:
                best_J = J
                best_sigma = sigma
                best_mmd = mmd_val
        return {'sigma': best_sigma, 'mmd': best_mmd, 'model': None, 'mode': 'bandwidth'}
    else:
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


def step1_liu(X, Y, dim, mode='bandwidth'):
    """Liu et al.'s criterion: J = MMD^2 / sigma_hat."""
    if mode == 'bandwidth':
        search_sigmas = np.logspace(-1, 1, 20)
        best_J = -float('inf')
        best_sigma = 1.0
        best_mmd = 0.0
        for sigma in search_sigmas:
            mmd_val = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
            m = len(X)
            Z = np.concatenate([X, Y], axis=0)
            Z_t = _to_tensor(Z)
            K = torch.exp(-_sq_euclidean_distances(Z_t, Z_t) / (sigma**2))
            h_vals = []
            for i in range(min(m, 50)):
                h_vals.append(K[i, :m].mean().item() - K[i, m:].mean().item())
            var_h = np.var(h_vals) + 1e-10
            J = mmd_val / np.sqrt(var_h / m)
            if J > best_J:
                best_J = J
                best_sigma = sigma
                best_mmd = mmd_val
        return {'sigma': best_sigma, 'mmd': best_mmd, 'model': None, 'mode': 'bandwidth'}
    else:
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


def step1_plain(X, Y, dim, mode='bandwidth'):
    """Plain MMD maximization (no regularization)."""
    if mode == 'bandwidth':
        opt = optimize_bandwidth(X, Y, 'gaussian', search_space=(0.1, 10.0))
        return {'sigma': opt['optimal_sigma'], 'mmd': opt['max_mmd'],
                'model': None, 'mode': 'bandwidth'}
    else:
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
    """MMDAgg bandwidth aggregation."""
    try:
        from mmdagg import mmdagg as mmdagg_fn
        result = mmdagg_fn(X, Y, alpha=ALPHA, return_dictionary=True)
        return {'decision': result, 'mode': 'aggregation'}
    except ImportError:
        sigmas = np.logspace(-1, 1, 10)
        m, n = len(X), len(Y)
        pooled = np.concatenate([X, Y], axis=0)
        N = m + n
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
        min_p = min(p_values) * len(sigmas)
        reject = min_p < ALPHA
        return {'decision': int(reject), 'p_value': min_p, 'mode': 'aggregation'}


# ======================================================================
# Step 2: Hypothesis testing (PermSplit_2, copied inline)
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


# ======================================================================
# Calibration (copied inline from run_cross_comparison.py)
# ======================================================================
def calibrate_C1_C2(dim, n_samples=200, dist_type='gaussian', mode='bandwidth', n_reps=10):
    """Calibrate C1 (95th percentile of MMD/G_ref) and C2.

    Journal procedure (shared G_ref): G is computed ONCE on the original
    un-permuted (X, Y) and reused as the denominator across all
    permutations. Only the numerator (max-MMD on the permuted data)
    varies, ensuring the calibration ratio is exchangeable under the
    null permutation distribution (cf. Prop 4.23 in the JMLR draft).
    """
    X, Y = generate_data(n_samples, dist_type=dist_type, shift_type='none', delta=0.0, dim=dim)
    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)

    # Shared G_ref: train kernel / pick bandwidth once on un-permuted data
    if mode == 'deep':
        res_ref = step1_plain(X, Y, dim, mode='deep')
        G_ref = compute_G_spectral(res_ref['model'], X, Y, res_ref['sigma'])
    else:
        res_ref = step1_plain(X, Y, dim, mode='bandwidth')
        G_ref = compute_G_bandwidth(X, Y, res_ref['sigma'])
    if abs(G_ref) <= 1e-12:
        G_ref = 1e-12

    ratios = []
    mmd_vals = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        if mode == 'deep':
            res = step1_plain(X_p, Y_p, dim, mode='deep')
        else:
            res = step1_plain(X_p, Y_p, dim, mode='bandwidth')
        mmd_vals.append(res['mmd'])
        ratios.append(res['mmd'] / G_ref)

    C1 = float(np.percentile(ratios, 95)) if ratios else 0.01

    if mode == 'deep':
        res = step1_plain(X, Y, dim, mode='deep')
        model, sigma = res['model'], res['sigma']
        X_t = _to_tensor(X).to(_DEVICE)
        Y_t = _to_tensor(Y).to(_DEVICE)
        with torch.no_grad():
            f_all = model(torch.cat([X_t, Y_t]))
        null_mmds = []
        for _ in range(200):
            perm = np.random.permutation(N)
            null_mmds.append(mmd_loss_torch(f_all[perm[:m]], f_all[perm[m:]], sigma).item())
    else:
        opt = optimize_bandwidth(X, Y, 'gaussian', search_space=(0.1, 10.0))
        sigma = opt['optimal_sigma']
        null_mmds = []
        for _ in range(200):
            perm = np.random.permutation(N)
            null_mmds.append(compute_mmd_u_statistic(
                pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))

    S_val = compute_S(ALPHA, m, n)
    q95 = np.percentile(null_mmds, 95)
    C2_fixed = q95 / S_val if S_val > 1e-12 else 0.0
    std_opt = np.std(mmd_vals) if len(mmd_vals) > 1 else 0.0
    std_null = np.std(null_mmds)
    var_ratio = max(std_opt / std_null, 1.0) if std_null > 1e-12 else 1.0
    C2 = C2_fixed * var_ratio

    return C1, C2


# ======================================================================
# Timing helpers
# ======================================================================
def time_method_decomposed(method_name, X, Y, dim, C1=None, mode='bandwidth'):
    """Time a single method, decomposed into calibration/selection/test phases.

    Returns dict with keys: calibration_time, selection_time, test_time, total_time.
    """
    calibration_time = 0.0
    selection_time = 0.0
    test_time = 0.0

    if method_name == 'CPMMD_1+PermSplit_2':
        # Calibration (only for CP-MMD)
        t0 = time.perf_counter()
        C1_local, C2_local = calibrate_C1_C2(dim, n_samples=len(X),
                                              dist_type='gaussian', mode=mode)
        calibration_time = time.perf_counter() - t0

        # Selection: train on first half
        m_tr, n_tr = len(X) // 2, len(Y) // 2
        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
        t0 = time.perf_counter()
        s1_result = step1_cpmmd(X_tr, Y_tr, dim, C1_local, mode=mode)
        selection_time = time.perf_counter() - t0

        # Test: permutation on second half
        t0 = time.perf_counter()
        step2_perm_split(X, Y, s1_result, dim)
        test_time = time.perf_counter() - t0

    elif method_name == 'Liu_1+PermSplit_2':
        # No calibration for Liu
        m_tr, n_tr = len(X) // 2, len(Y) // 2
        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
        t0 = time.perf_counter()
        s1_result = step1_liu(X_tr, Y_tr, dim, mode=mode)
        selection_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        step2_perm_split(X, Y, s1_result, dim)
        test_time = time.perf_counter() - t0

    elif method_name == 'Plain_1+PermSplit_2':
        # No calibration for Plain
        m_tr, n_tr = len(X) // 2, len(Y) // 2
        X_tr, Y_tr = X[:m_tr], Y[:n_tr]
        t0 = time.perf_counter()
        s1_result = step1_plain(X_tr, Y_tr, dim, mode=mode)
        selection_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        step2_perm_split(X, Y, s1_result, dim)
        test_time = time.perf_counter() - t0

    elif method_name == 'Median+Perm':
        # No calibration, no real selection
        t0 = time.perf_counter()
        s1_result = step1_median(X, Y, dim)
        selection_time = time.perf_counter() - t0

        # Permutation test on full data (no split needed for median)
        m, n = len(X), len(Y)
        pooled = np.concatenate([X, Y], axis=0)
        N = m + n
        sigma = s1_result['sigma']
        test_stat = s1_result['mmd']
        t0 = time.perf_counter()
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(N)
            null_mmds.append(compute_mmd_u_statistic(
                pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))
        threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
        test_time = time.perf_counter() - t0

    elif method_name == 'MMDAgg+Perm':
        # MMDAgg does everything internally
        t0 = time.perf_counter()
        step1_mmdagg(X, Y, dim)
        selection_time = 0.0  # aggregation includes both
        test_time = time.perf_counter() - t0
        # For MMDAgg, selection and test are interleaved; report as test_time

    total_time = calibration_time + selection_time + test_time
    return {
        'calibration_time': calibration_time,
        'selection_time': selection_time,
        'test_time': test_time,
        'total_time': total_time,
    }


# ======================================================================
# Main experiment
# ======================================================================
def run_decomposed_timing(dim=50, n_samples=100, mode='bandwidth'):
    """Run decomposed timing for all methods at fixed d=50, m=n=100."""
    methods = [
        'CPMMD_1+PermSplit_2',
        'Liu_1+PermSplit_2',
        'Plain_1+PermSplit_2',
        'Median+Perm',
        'MMDAgg+Perm',
    ]

    log(f"\n{'='*70}")
    log(f"Decomposed Timing: dim={dim}, n={n_samples}, mode={mode}")
    log(f"{'='*70}")

    # Pre-calibrate C1 once for CP-MMD (used in decomposition)
    log("  Pre-calibrating C1, C2 for CP-MMD...")
    C1, C2 = calibrate_C1_C2(dim, n_samples, 'gaussian', mode=mode)
    log(f"  C1={C1:.6f}, C2={C2:.4f}")

    results = []
    for method_name in methods:
        log(f"\n  Method: {method_name}")
        times_calib = []
        times_select = []
        times_test = []
        times_total = []

        for rep in range(N_REPS):
            np.random.seed(rep * 7777)
            torch.manual_seed(rep * 7777)
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean', delta=0.5, dim=dim)
            try:
                t = time_method_decomposed(method_name, X, Y, dim, C1=C1, mode=mode)
                times_calib.append(t['calibration_time'])
                times_select.append(t['selection_time'])
                times_test.append(t['test_time'])
                times_total.append(t['total_time'])
            except Exception as e:
                log(f"    Rep {rep} failed: {e}")

        if times_total:
            row = {
                'method': method_name,
                'dim': dim,
                'n_samples': n_samples,
                'mode': mode,
                'n_reps': len(times_total),
                'calib_mean': np.mean(times_calib),
                'calib_std': np.std(times_calib),
                'select_mean': np.mean(times_select),
                'select_std': np.std(times_select),
                'test_mean': np.mean(times_test),
                'test_std': np.std(times_test),
                'total_mean': np.mean(times_total),
                'total_std': np.std(times_total),
            }
            results.append(row)
            log(f"    Calib:  {row['calib_mean']:.3f} +/- {row['calib_std']:.3f} s")
            log(f"    Select: {row['select_mean']:.3f} +/- {row['select_std']:.3f} s")
            log(f"    Test:   {row['test_mean']:.3f} +/- {row['test_std']:.3f} s")
            log(f"    Total:  {row['total_mean']:.3f} +/- {row['total_std']:.3f} s")

    return results


def run_scaling_timing(dim=50, mode='bandwidth'):
    """Run total timing for all methods at n in {100, 200, 500}."""
    methods = [
        'CPMMD_1+PermSplit_2',
        'Liu_1+PermSplit_2',
        'Plain_1+PermSplit_2',
        'Median+Perm',
        'MMDAgg+Perm',
    ]
    sample_sizes = [100, 200, 500]

    log(f"\n{'='*70}")
    log(f"Scaling Timing: dim={dim}, n in {sample_sizes}, mode={mode}")
    log(f"{'='*70}")

    results = []
    for n_samples in sample_sizes:
        log(f"\n  n={n_samples}")

        # Calibrate for this sample size
        log(f"    Calibrating C1, C2...")
        C1, C2 = calibrate_C1_C2(dim, n_samples, 'gaussian', mode=mode)
        log(f"    C1={C1:.6f}, C2={C2:.4f}")

        for method_name in methods:
            log(f"    Method: {method_name}")
            times_total = []

            for rep in range(N_REPS):
                np.random.seed(rep * 8888 + n_samples)
                torch.manual_seed(rep * 8888 + n_samples)
                X, Y = generate_data(n_samples, dist_type='gaussian',
                                     shift_type='mean', delta=0.5, dim=dim)
                try:
                    t = time_method_decomposed(method_name, X, Y, dim, C1=C1, mode=mode)
                    times_total.append(t['total_time'])
                except Exception as e:
                    log(f"      Rep {rep} failed: {e}")

            if times_total:
                row = {
                    'method': method_name,
                    'dim': dim,
                    'n_samples': n_samples,
                    'mode': mode,
                    'experiment': 'scaling',
                    'n_reps': len(times_total),
                    'total_mean': np.mean(times_total),
                    'total_std': np.std(times_total),
                }
                results.append(row)
                log(f"      Total: {row['total_mean']:.3f} +/- {row['total_std']:.3f} s")

    return results


def save_results(results, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not results:
        log(f"  No results to save for {filename}")
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    log(f"  Saved {filepath}")


def main():
    log(f"{'='*70}")
    log(f"Phase 2c: Computational Cost Experiment")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"N_REPS={N_REPS}, N_PERM={N_PERM}")
    log(f"{'='*70}")

    all_results = []

    # Part 1: Decomposed timing at d=50, n=100 (bandwidth mode)
    decomp = run_decomposed_timing(dim=50, n_samples=100, mode='bandwidth')
    all_results.extend(decomp)

    # Part 2: Scaling with n at d=50 (bandwidth mode)
    scaling = run_scaling_timing(dim=50, mode='bandwidth')
    all_results.extend(scaling)

    # Save combined results
    save_results(all_results, 'timing_results.csv')

    # Also save decomposed and scaling separately for convenience
    save_results(decomp, 'timing_decomposed.csv')
    save_results(scaling, 'timing_scaling.csv')

    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()


if __name__ == '__main__':
    main()
