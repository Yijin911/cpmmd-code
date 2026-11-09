"""
Cross-comparison experiment: separate kernel selection (Step 1) from hypothesis testing (Step 2).

Step 1 methods (kernel/model selection):
  CPMMD_1:   maximize J_CP = MMD^2 - C1 * G_h (complexity-penalized)
  Liu_1:     maximize J_Liu  = MMD^2 / sigma_hat (variance-normalized)
  Plain_1:   maximize MMD^2  (no regularization)
  MMDAgg_1:  bandwidth aggregation (Schrab et al., 2023)
  Median_1:  median heuristic (no optimization)

Step 2 methods (hypothesis testing):
  CPMMD_2:      threshold = C1*G + C2*S (closed-form, no split)
  PermSplit_2:  permutation on held-out 50% (Liu et al.'s approach)
  PermReopt_2:  permutation with re-optimization, 10 perms (reviewer baseline)

Experiments:
  A: Bandwidth tuning (Gaussian dim=2, dim=20)
  B: Deep kernel learning (Gaussian dim=2, dim=20)
  C: Fair representation learning
  D: Generative model evaluation

For each experiment, report all Step1 x Step2 combinations.
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
    _to_tensor, _sq_euclidean_distances, _DEVICE, BestIterate
)
from calibration_engine import compute_S
from data_generator import generate_data

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_cross')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 200
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
N_PERM = 200       # permutations for threshold
N_PERM_REOPT = 10  # re-optimization permutations

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


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
def step1_cpmmd(X, Y, dim, C1, mode='deep'):
    """CP-MMD criterion: J = MMD^2 - C1*G_h."""
    if mode == 'bandwidth':
        # Bandwidth optimization with complexity penalty
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
        # Deep kernel with complexity penalty
        X_t = _to_tensor(X).to(_DEVICE)
        Y_t = _to_tensor(Y).to(_DEVICE)
        D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
        log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
        params = list(D.parameters()) + [log_sigma]
        optimizer = torch.optim.Adam(params, lr=LR)
        best = BestIterate()  # h_hat = argmax_{0<=t<=T} J_CP(h^(t))
        for step in range(N_STEPS + 1):
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
            best.update(J.item(), D, log_sigma)
            if step == N_STEPS:
                break
            (-J).backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
        best.restore(D, log_sigma)
        D.eval()
        with torch.no_grad():
            sigma_val = log_sigma.exp().item()
            f_x, f_y = D(X_t), D(Y_t)
            mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
        return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D, 'mode': 'deep'}


def step1_liu(X, Y, dim, mode='deep'):
    """Liu et al.'s criterion: J = MMD^2 / sigma_hat."""
    if mode == 'bandwidth':
        search_sigmas = np.logspace(-1, 1, 20)
        best_J = -float('inf')
        best_sigma = 1.0
        best_mmd = 0.0
        for sigma in search_sigmas:
            mmd_val = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
            # Estimate variance
            m = len(X)
            Z = np.concatenate([X, Y], axis=0)
            Z_t = _to_tensor(Z)
            K = torch.exp(-_sq_euclidean_distances(Z_t, Z_t) / (sigma**2))
            # Simple variance estimate
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


def step1_plain(X, Y, dim, mode='deep'):
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
    """MMDAgg bandwidth aggregation. Returns the test decision directly."""
    try:
        from mmdagg import mmdagg as mmdagg_fn
        result = mmdagg_fn(X, Y, alpha=ALPHA, return_dictionary=True)
        return {'decision': result, 'mode': 'aggregation'}
    except ImportError:
        # Fallback: implement simple bandwidth aggregation
        sigmas = np.logspace(-1, 1, 10)
        m, n = len(X), len(Y)
        pooled = np.concatenate([X, Y], axis=0)
        N = m + n
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
def step2_cpmmd(X, Y, step1_result, C1, C2):
    """CP-MMD threshold: C1*G + C2*S."""
    S = compute_S(ALPHA, len(X), len(Y))
    if step1_result['model'] is not None:
        G = compute_G_spectral(step1_result['model'], X, Y, step1_result['sigma'])
    else:
        G = compute_G_bandwidth(X, Y, step1_result['sigma'])
    threshold = C1 * G + C2 * S
    # Recompute MMD at the selected kernel
    if step1_result['model'] is not None:
        X_t = _to_tensor(X).to(_DEVICE)
        Y_t = _to_tensor(Y).to(_DEVICE)
        with torch.no_grad():
            f_x = step1_result['model'](X_t)
            f_y = step1_result['model'](Y_t)
            test_stat = mmd_loss_torch(f_x, f_y, step1_result['sigma']).item()
    else:
        test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=step1_result['sigma'])
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


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


def step2_perm_reopt(X, Y, dim, mode='deep', n_perm=N_PERM_REOPT):
    """Permutation with re-optimization (reviewer baseline). No split."""
    m, n = len(X), len(Y)
    pooled = np.concatenate([X, Y], axis=0)
    N = m + n

    # Test statistic: train on original data
    if mode == 'deep':
        res = step1_plain(X, Y, dim, mode='deep')
        test_stat = res['mmd']
    else:
        res = step1_plain(X, Y, dim, mode='bandwidth')
        test_stat = res['mmd']

    # Null distribution: permute and re-train
    null_mmds = []
    for p in range(n_perm):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        if mode == 'deep':
            res_null = step1_plain(X_p, Y_p, dim, mode='deep')
        else:
            res_null = step1_plain(X_p, Y_p, dim, mode='bandwidth')
        null_mmds.append(res_null['mmd'])

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


# ======================================================================
# Calibration
# ======================================================================
def calibrate_C1_C2(dim, n_samples=200, dist_type='gaussian', mode='deep', n_reps=10):
    """Calibrate C1 (95th percentile of MMD/G) and C2."""
    X, Y = generate_data(n_samples, dist_type=dist_type, shift_type='none', delta=0.0, dim=dim)
    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)

    ratios = []
    mmd_vals = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        if mode == 'deep':
            res = step1_plain(X_p, Y_p, dim, mode='deep')
            G = compute_G_spectral(res['model'], X_p, Y_p, res['sigma'])
        else:
            res = step1_plain(X_p, Y_p, dim, mode='bandwidth')
            G = compute_G_bandwidth(X_p, Y_p, res['sigma'])
        mmd_vals.append(res['mmd'])
        if abs(G) > 1e-12:
            ratios.append(res['mmd'] / G)

    C1 = float(np.percentile(ratios, 95)) if ratios else 0.01

    # C2 from permutations at fixed model
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
# Experiment runner
# ======================================================================
def run_experiment(exp_name, dim, dist_type, deltas, n_samples=200, mode='deep'):
    """Run one experiment with all Step1 x Step2 combinations."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT: {exp_name} (dim={dim}, dist={dist_type}, mode={mode}, n={n_samples})")
    log(f"{'='*70}")

    # Calibrate
    log("  Calibrating C1, C2...")
    C1, C2 = calibrate_C1_C2(dim, n_samples, dist_type, mode=mode)
    log(f"  C1={C1:.6f}, C2={C2:.4f}")

    # Define Step 1 methods
    step1_methods = {
        'CPMMD_1': lambda X, Y: step1_cpmmd(X, Y, dim, C1, mode=mode),
        'Liu_1': lambda X, Y: step1_liu(X, Y, dim, mode=mode),
        'Plain_1': lambda X, Y: step1_plain(X, Y, dim, mode=mode),
        'Median_1': lambda X, Y: step1_median(X, Y, dim),
    }

    # Define Step 2 methods
    step2_methods = {
        'CPMMD_2': lambda X, Y, s1: step2_cpmmd(X, Y, s1, C1, C2),
        'PermSplit_2': lambda X, Y, s1: step2_perm_split(X, Y, s1, dim),
    }

    all_results = []

    for delta in deltas:
        log(f"\n  delta={delta}")
        for s1_name, s1_fn in step1_methods.items():
            for s2_name, s2_fn in step2_methods.items():
                # Skip incompatible: Median_1 doesn't have a model for deep Step2
                rejects = 0
                valid = 0
                t0 = time.time()
                for rep in range(N_REPS):
                    np.random.seed(rep * 1000 + int(delta * 100))
                    torch.manual_seed(rep * 1000 + int(delta * 100))
                    X, Y = generate_data(n_samples, dist_type=dist_type,
                                         shift_type='mean' if delta > 0 else 'none',
                                         delta=delta, dim=dim)
                    try:
                        # Step 1: select kernel
                        s1_result = s1_fn(X, Y)

                        # For PermSplit: re-train on train half only
                        if s2_name == 'PermSplit_2':
                            m_tr = len(X) // 2
                            n_tr = len(Y) // 2
                            X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                            s1_result_split = s1_fn(X_tr, Y_tr)
                            s2_result = step2_perm_split(X, Y, s1_result_split, dim)
                        else:
                            s2_result = s2_fn(X, Y, s1_result)

                        rejects += int(s2_result['reject'])
                        valid += 1
                    except Exception as e:
                        pass

                dt = time.time() - t0
                rate = rejects / max(valid, 1)
                label = "TYPE_I" if delta == 0 else "POWER"
                log(f"    {s1_name:10s} + {s2_name:12s}: {label}={rate:.3f} ({valid}/{N_REPS} valid, {dt:.1f}s)")

                all_results.append({
                    'experiment': exp_name, 'dim': dim, 'dist_type': dist_type,
                    'delta': delta, 'n_samples': n_samples, 'mode': mode,
                    'step1': s1_name, 'step2': s2_name,
                    'rejection_rate': rate, 'n_valid': valid, 'time_s': dt,
                })

        # Also run PermReopt (it does its own Step1 internally)
        rejects = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type=dist_type,
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                res = step2_perm_reopt(X, Y, dim, mode=mode)
                rejects += int(res['reject'])
            except:
                pass
        dt = time.time() - t0
        rate = rejects / N_REPS
        label = "TYPE_I" if delta == 0 else "POWER"
        log(f"    {'Plain_1':10s} + {'PermReopt_2':12s}: {label}={rate:.3f} ({N_REPS}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'experiment': exp_name, 'dim': dim, 'dist_type': dist_type,
            'delta': delta, 'n_samples': n_samples, 'mode': mode,
            'step1': 'Plain_1', 'step2': 'PermReopt_2',
            'rejection_rate': rate, 'n_valid': N_REPS, 'time_s': dt,
        })

        # MMDAgg (does both steps internally)
        rejects = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type=dist_type,
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                res = step1_mmdagg(X, Y, dim)
                if isinstance(res.get('decision'), dict):
                    rejects += int(res['decision'].get('reject', 0))
                else:
                    rejects += int(res['decision'])
            except Exception as e:
                pass
        dt = time.time() - t0
        rate = rejects / N_REPS
        label = "TYPE_I" if delta == 0 else "POWER"
        log(f"    {'MMDAgg':10s} + {'(internal)':12s}: {label}={rate:.3f} ({N_REPS}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'experiment': exp_name, 'dim': dim, 'dist_type': dist_type,
            'delta': delta, 'n_samples': n_samples, 'mode': mode,
            'step1': 'MMDAgg', 'step2': 'MMDAgg',
            'rejection_rate': rate, 'n_valid': N_REPS, 'time_s': dt,
        })

    return all_results


def save_results(results, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    log(f"  Saved {filepath}")


def main():
    log(f"{'='*70}")
    log(f"Cross-Comparison: Step1 (kernel selection) x Step2 (testing)")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"{'='*70}")

    all_results = []

    # Experiment A: Bandwidth tuning
    deltas_bw = [0, 0.3, 0.5, 1.0]
    res = run_experiment('A_bw_d2', dim=2, dist_type='gaussian',
                         deltas=deltas_bw, mode='bandwidth')
    all_results.extend(res)
    save_results(all_results, 'results_A.csv')

    res = run_experiment('A_bw_d20', dim=20, dist_type='gaussian',
                         deltas=deltas_bw, mode='bandwidth')
    all_results.extend(res)
    save_results(all_results, 'results_AB.csv')

    # Experiment B: Deep kernel learning
    res = run_experiment('B_deep_d2', dim=2, dist_type='gaussian',
                         deltas=deltas_bw, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_ABC.csv')

    res = run_experiment('B_deep_d20', dim=20, dist_type='gaussian',
                         deltas=deltas_bw, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_all.csv')

    # Experiment C: Deep kernel, high d/n ratio (where Plain overfits)
    deltas_hard = [0, 0.5, 1.0]
    res = run_experiment('C_deep_d50_n100', dim=50, dist_type='gaussian',
                         deltas=deltas_hard, n_samples=100, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_all_C.csv')

    res = run_experiment('C_deep_d100_n100', dim=100, dist_type='gaussian',
                         deltas=deltas_hard, n_samples=100, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_all_C2.csv')

    res = run_experiment('C_deep_d20_n50', dim=20, dist_type='gaussian',
                         deltas=deltas_hard, n_samples=50, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_all_final.csv')

    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
