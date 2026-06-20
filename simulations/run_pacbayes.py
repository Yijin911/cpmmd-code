"""
PAC-Bayes criterion for MMD kernel selection.

Implements the PAC-Bayes bound (Theorem 5) for a fully rigorous test
that requires NO calibration and NO data splitting.

PAC-Bayes bound:
  E_{theta~rho}[gamma^2(h_theta)] >= E_{theta~rho}[gamma_hat^2(h_theta)]
      - nu * sqrt(2*(KL(rho||pi) + ln(2*sqrt(N)/delta)) / N)

where:
  rho = N(theta_hat, sigma^2 I)    posterior centered at trained weights
  pi  = N(theta_0, sigma_0^2 I)    prior centered at initialization
  KL(rho||pi) = ||theta_hat - theta_0||^2 / (2*sigma_0^2)
              + (p/2)*(sigma^2/sigma_0^2 - 1 - ln(sigma^2/sigma_0^2))

Test: reject H0 if J_PB > 0, where
  J_PB = E_rho[MMD^2] - nu * sqrt(2*(KL + ln(2*sqrt(N)/delta)) / N)

Methods compared:
  PACBayes_1 + Direct_2:  J_PB selection + direct threshold (no split, no perm)
  CPMMD_1    + PermSplit_2
  Liu_1      + PermSplit_2
  Plain_1    + PermSplit_2
  Median_1   + Perm_2
"""

import os, sys, time, csv, warnings, copy
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'pacbayes')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
N_PERM = 200
N_MC_SAMPLES = 20    # Monte Carlo samples for E_rho estimation
DELTA_PB = 0.05      # PAC-Bayes confidence parameter
NU = 1.0             # sup k(u,u) = 1 for Gaussian kernel

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = None


def log(msg):
    global log_fh
    if log_fh is None:
        log_fh = open(LOG_FILE, 'w', buffering=1)
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ======================================================================
# Utility functions
# ======================================================================
def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_params(model):
    """Flatten all parameters into a single 1D tensor."""
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def compute_kl_divergence(theta_hat, theta_0, sigma_sq, sigma_0_sq, p):
    """
    KL(rho || pi) for diagonal Gaussians with shared variance per coord.

    rho = N(theta_hat, sigma^2 I)
    pi  = N(theta_0, sigma_0^2 I)

    KL = ||theta_hat - theta_0||^2 / (2 * sigma_0^2)
       + (p/2) * (sigma^2/sigma_0^2 - 1 - ln(sigma^2/sigma_0^2))
    """
    diff_sq = (theta_hat - theta_0).pow(2).sum().item()
    ratio = sigma_sq / sigma_0_sq
    kl = diff_sq / (2.0 * sigma_0_sq) + (p / 2.0) * (ratio - 1.0 - np.log(ratio))
    return kl


def compute_pacbayes_penalty(kl, N, delta=DELTA_PB, nu=NU):
    """
    Compute the PAC-Bayes penalty term:
      nu * sqrt(2 * (KL + ln(2*sqrt(N)/delta)) / N)
    """
    log_term = np.log(2.0 * np.sqrt(N) / delta)
    return nu * np.sqrt(2.0 * (kl + log_term) / N)


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


def compute_G_bandwidth(X, Y, sigma):
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return np.linalg.norm(Z) / (np.sqrt(N) * sigma)


# ======================================================================
# Step 1: Kernel/model selection methods
# ======================================================================
def step1_pacbayes(X, Y, dim):
    """
    PAC-Bayes criterion: train deep kernel by maximizing J_PB.

    Training objective (each step):
      maximize MMD^2(D(X), D(Y)) - lambda * sqrt(KL / N)
    where lambda = nu * sqrt(2/N) and KL = ||theta - theta_0||^2 / (2*sigma_0^2) + const

    After training:
      - sigma_0^2 = 1/p  (unit prior scaled by dimension)
      - sigma^2 = sigma_0^2 / N  (Corollary 1)
      - Compute KL(rho || pi)
      - Estimate E_rho[MMD^2] via Monte Carlo with K=N_MC_SAMPLES perturbations
      - J_PB = E_rho[MMD^2] - penalty
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    N = len(X) + len(Y)

    # Build model
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))

    # Save initial weights (prior center theta_0)
    theta_0 = flatten_params(D).clone()
    p = theta_0.numel()

    # Prior and posterior variances
    sigma_0_sq = 1.0 / p        # unit prior scaled by dimension
    sigma_sq = sigma_0_sq / N   # Corollary 1

    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    # Precompute training regularization coefficient
    # lambda = nu * sqrt(2/N)
    lam = NU * np.sqrt(2.0 / N)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma_k = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma_k)

        # KL divergence (only the weight-dependent part for gradient)
        theta_cur = torch.cat([p_val.view(-1) for p_val in D.parameters()])
        theta_0_dev = theta_0.to(_DEVICE)
        diff_sq = (theta_cur - theta_0_dev).pow(2).sum()
        # KL = diff_sq / (2*sigma_0_sq) + const (const doesn't affect gradient)
        kl_approx = diff_sq / (2.0 * sigma_0_sq)
        penalty = lam * torch.sqrt(kl_approx / N + 1e-10)

        J = mmd2 - penalty
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()

    # --- Post-training: compute full J_PB ---
    theta_hat = flatten_params(D).clone().to(_DEVICE)
    theta_0_dev = theta_0.to(_DEVICE)

    # Full KL divergence
    kl = compute_kl_divergence(theta_hat, theta_0_dev, sigma_sq, sigma_0_sq, p)

    # Monte Carlo estimate of E_rho[MMD^2]
    mc_mmd_values = []
    for k in range(N_MC_SAMPLES):
        # Perturb weights: theta_hat + epsilon, epsilon ~ N(0, sigma^2 I)
        D_perturbed = copy.deepcopy(D).to(_DEVICE)
        with torch.no_grad():
            for param in D_perturbed.parameters():
                noise = torch.randn_like(param) * np.sqrt(sigma_sq)
                param.add_(noise)
            f_x_p = D_perturbed(X_t)
            f_y_p = D_perturbed(Y_t)
            mmd2_k = mmd_loss_torch(f_x_p, f_y_p, sigma_val).item()
            mc_mmd_values.append(mmd2_k)

    E_rho_mmd2 = np.mean(mc_mmd_values)

    # PAC-Bayes penalty
    penalty_val = compute_pacbayes_penalty(kl, N)

    # J_PB = E_rho[MMD^2] - penalty
    J_PB = E_rho_mmd2 - penalty_val

    # Also compute the unperturbed MMD for reporting
    with torch.no_grad():
        f_x, f_y = D(X_t), D(Y_t)
        mmd_unperturbed = mmd_loss_torch(f_x, f_y, sigma_val).item()

    return {
        'sigma': sigma_val,
        'mmd': mmd_unperturbed,
        'model': D,
        'mode': 'deep',
        'J_PB': J_PB,
        'E_rho_mmd2': E_rho_mmd2,
        'kl': kl,
        'penalty': penalty_val,
        'theta_0': theta_0,
        'sigma_sq': sigma_sq,
        'sigma_0_sq': sigma_0_sq,
    }


def step1_cpmmd(X, Y, dim, C1):
    """CP-MMD criterion: J = MMD^2 - C1*G_h (deep kernel)."""
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


# ======================================================================
# Step 2: Hypothesis testing methods
# ======================================================================
def step2_pacbayes_direct(X, Y, step1_result):
    """
    PAC-Bayes direct test: reject H0 if J_PB > 0.

    No permutation needed, no data splitting needed.
    Uses the FULL sample for both selection and testing.
    """
    J_PB = step1_result['J_PB']
    reject = J_PB > 0
    return {
        'reject': reject,
        'test_stat': step1_result['E_rho_mmd2'],
        'threshold': step1_result['penalty'],
        'J_PB': J_PB,
    }


def step2_cpmmd(X, Y, step1_result, C1, C2):
    """CP-MMD threshold: C1*G + C2*S."""
    S = compute_S(ALPHA, len(X), len(Y))
    if step1_result['model'] is not None:
        G = compute_G_spectral(step1_result['model'], X, Y, step1_result['sigma'])
    else:
        G = compute_G_bandwidth(X, Y, step1_result['sigma'])
    threshold = C1 * G + C2 * S
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
    """Permutation test on full data (for Median heuristic baseline)."""
    m, n = len(X), len(Y)
    sigma = step1_result['sigma']

    if step1_result['model'] is not None:
        model = step1_result['model']
        X_t = _to_tensor(X).to(_DEVICE)
        Y_t = _to_tensor(Y).to(_DEVICE)
        with torch.no_grad():
            f_all = model(torch.cat([X_t, Y_t]))
        test_stat = mmd_loss_torch(f_all[:m], f_all[m:], sigma).item()
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(m + n)
            null_mmds.append(mmd_loss_torch(f_all[perm[:m]], f_all[perm[m:]], sigma).item())
    else:
        test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
        pooled = np.concatenate([X, Y], axis=0)
        null_mmds = []
        for _ in range(N_PERM):
            perm = np.random.permutation(m + n)
            null_mmds.append(compute_mmd_u_statistic(
                pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    reject = test_stat > threshold
    return {'reject': reject, 'test_stat': test_stat, 'threshold': threshold}


# ======================================================================
# Calibration (for CP-MMD method)
# ======================================================================
def calibrate_C1_C2(dim, n_samples=200, n_reps=10):
    """Calibrate C1 (95th percentile of MMD/G_ref) and C2.

    Journal procedure (shared G_ref): G is computed ONCE on the original
    un-permuted (X, Y) by training a fresh deep kernel on it, and reused
    across all permutations. Only the numerator (max-MMD trained on the
    permuted data) varies with the permutation. This makes the
    calibration ratio exchangeable under the permutation null
    (cf. Prop 4.23).
    """
    X, Y = generate_data(n_samples, dist_type='gaussian', shift_type='none', delta=0.0, dim=dim)
    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)

    # Shared G_ref: train kernel once on un-permuted data
    res_ref = step1_plain(X, Y, dim)
    G_ref = compute_G_spectral(res_ref['model'], X, Y, res_ref['sigma'])
    if abs(G_ref) <= 1e-12:
        G_ref = 1e-12

    ratios = []
    mmd_vals = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = step1_plain(X_p, Y_p, dim)
        mmd_vals.append(res['mmd'])
        ratios.append(res['mmd'] / G_ref)

    C1 = float(np.percentile(ratios, 95)) if ratios else 0.01

    # C2 from permutations at fixed model
    res = step1_plain(X, Y, dim)
    model, sigma = res['model'], res['sigma']
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    with torch.no_grad():
        f_all = model(torch.cat([X_t, Y_t]))
    null_mmds = []
    for _ in range(200):
        perm = np.random.permutation(N)
        null_mmds.append(mmd_loss_torch(f_all[perm[:m]], f_all[perm[m:]], sigma).item())

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
def run_setting(setting_name, dim, n_samples, deltas):
    """Run one experimental setting with all method combinations."""
    log(f"\n{'='*70}")
    log(f"SETTING: {setting_name} (dim={dim}, n={n_samples})")
    log(f"{'='*70}")

    # Calibrate C1, C2 for the CP-MMD method
    log("  Calibrating C1, C2 for CP-MMD method...")
    C1, C2 = calibrate_C1_C2(dim, n_samples, n_reps=10)
    log(f"  C1={C1:.6f}, C2={C2:.4f}")

    all_results = []

    for delta in deltas:
        is_null = (delta == 0)
        label = "TYPE_I" if is_null else "POWER"
        log(f"\n  delta={delta} ({label})")

        # ---- Method 1: PACBayes_1 + Direct_2 ----
        rejects = 0
        valid = 0
        jpb_vals = []
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                s1 = step1_pacbayes(X, Y, dim)
                s2 = step2_pacbayes_direct(X, Y, s1)
                rejects += int(s2['reject'])
                jpb_vals.append(s1['J_PB'])
                valid += 1
            except Exception as e:
                log(f"      [PACBayes rep {rep}] Error: {e}")
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        mean_jpb = np.mean(jpb_vals) if jpb_vals else float('nan')
        log(f"    PACBayes_1  + Direct_2   : {label}={rate:.3f}  (mean J_PB={mean_jpb:.6f}, {valid}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'setting': setting_name, 'dim': dim, 'n_samples': n_samples,
            'delta': delta, 'step1': 'PACBayes_1', 'step2': 'Direct_2',
            'rejection_rate': rate, 'n_valid': valid, 'time_s': round(dt, 1),
            'mean_J_PB': round(mean_jpb, 8) if not np.isnan(mean_jpb) else '',
        })

        # ---- Method 2: CPMMD_1 + PermSplit_2 ----
        rejects = 0
        valid = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                m_tr = len(X) // 2
                n_tr = len(Y) // 2
                X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                s1 = step1_cpmmd(X_tr, Y_tr, dim, C1)
                s2 = step2_perm_split(X, Y, s1, dim)
                rejects += int(s2['reject'])
                valid += 1
            except Exception as e:
                pass
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        log(f"    CPMMD_1     + PermSplit_2: {label}={rate:.3f}  ({valid}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'setting': setting_name, 'dim': dim, 'n_samples': n_samples,
            'delta': delta, 'step1': 'CPMMD_1', 'step2': 'PermSplit_2',
            'rejection_rate': rate, 'n_valid': valid, 'time_s': round(dt, 1),
            'mean_J_PB': '',
        })

        # ---- Method 3: Liu_1 + PermSplit_2 ----
        rejects = 0
        valid = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                m_tr = len(X) // 2
                n_tr = len(Y) // 2
                X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                s1 = step1_liu(X_tr, Y_tr, dim)
                s2 = step2_perm_split(X, Y, s1, dim)
                rejects += int(s2['reject'])
                valid += 1
            except Exception as e:
                pass
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        log(f"    Liu_1       + PermSplit_2: {label}={rate:.3f}  ({valid}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'setting': setting_name, 'dim': dim, 'n_samples': n_samples,
            'delta': delta, 'step1': 'Liu_1', 'step2': 'PermSplit_2',
            'rejection_rate': rate, 'n_valid': valid, 'time_s': round(dt, 1),
            'mean_J_PB': '',
        })

        # ---- Method 4: Plain_1 + PermSplit_2 ----
        rejects = 0
        valid = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                m_tr = len(X) // 2
                n_tr = len(Y) // 2
                X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                s1 = step1_plain(X_tr, Y_tr, dim)
                s2 = step2_perm_split(X, Y, s1, dim)
                rejects += int(s2['reject'])
                valid += 1
            except Exception as e:
                pass
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        log(f"    Plain_1     + PermSplit_2: {label}={rate:.3f}  ({valid}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'setting': setting_name, 'dim': dim, 'n_samples': n_samples,
            'delta': delta, 'step1': 'Plain_1', 'step2': 'PermSplit_2',
            'rejection_rate': rate, 'n_valid': valid, 'time_s': round(dt, 1),
            'mean_J_PB': '',
        })

        # ---- Method 5: Median_1 + Perm_2 (full data, no split) ----
        rejects = 0
        valid = 0
        t0 = time.time()
        for rep in range(N_REPS):
            np.random.seed(rep * 1000 + int(delta * 100))
            torch.manual_seed(rep * 1000 + int(delta * 100))
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                s1 = step1_median(X, Y, dim)
                s2 = step2_perm_full(X, Y, s1)
                rejects += int(s2['reject'])
                valid += 1
            except Exception as e:
                pass
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        log(f"    Median_1    + Perm_2     : {label}={rate:.3f}  ({valid}/{N_REPS} valid, {dt:.1f}s)")
        all_results.append({
            'setting': setting_name, 'dim': dim, 'n_samples': n_samples,
            'delta': delta, 'step1': 'Median_1', 'step2': 'Perm_2',
            'rejection_rate': rate, 'n_valid': valid, 'time_s': round(dt, 1),
            'mean_J_PB': '',
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
    log(f"PAC-Bayes Criterion for MMD Kernel Selection")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")
    log(f"N_REPS={N_REPS}, N_MC_SAMPLES={N_MC_SAMPLES}, N_STEPS={N_STEPS}")
    log(f"N_PERM={N_PERM}, DELTA_PB={DELTA_PB}, NU={NU}")
    log(f"Architecture: d -> {HIDDEN_WIDTH} -> {HIDDEN_WIDTH} -> {FEATURE_DIM}")
    log(f"{'='*70}")

    deltas = [0, 0.5, 1.0]
    all_results = []

    # Setting 1: d=20, n=200 (moderate)
    res = run_setting('moderate_d20_n200', dim=20, n_samples=200, deltas=deltas)
    all_results.extend(res)
    save_results(all_results, 'pacbayes_results.csv')

    # Setting 2: d=50, n=100 (high d/n)
    res = run_setting('high_dn_d50_n100', dim=50, n_samples=100, deltas=deltas)
    all_results.extend(res)
    save_results(all_results, 'pacbayes_results.csv')

    # Setting 3: d=20, n=50 (small sample)
    res = run_setting('small_d20_n50', dim=20, n_samples=50, deltas=deltas)
    all_results.extend(res)
    save_results(all_results, 'pacbayes_results.csv')

    # Print summary table
    log(f"\n{'='*70}")
    log(f"SUMMARY")
    log(f"{'='*70}")
    log(f"{'Setting':<25s} {'Method':<30s} {'delta':>6s} {'Rate':>8s}")
    log(f"{'-'*70}")
    for r in all_results:
        method = f"{r['step1']} + {r['step2']}"
        label = "TYPE_I" if r['delta'] == 0 else "POWER"
        log(f"{r['setting']:<25s} {method:<30s} {r['delta']:>6.1f} {r['rejection_rate']:>8.3f} ({label})")

    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {os.path.join(RESULTS_DIR, 'pacbayes_results.csv')}")


if __name__ == '__main__':
    main()
