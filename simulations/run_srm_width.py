"""
SRM (Structural Risk Minimization) empirical instantiation:
vary HIDDEN_WIDTH across {50, 100, 200, 500, 1000, 2000} on a fixed deep
two-sample task; compare Plain (no penalty) vs CP-MMD (data-adaptive penalty).

Hypothesis (Cor. 4 'SRM across model classes'):
- Plain power should peak at moderate width then COLLAPSE as width grows
  (over-parameterised network overfits the empirical MMD on the training half,
   so the held-out U-statistic at the selected kernel has degraded power).
- CP-MMD power should remain FLAT across widths because the spectral-norm
  penalty Ĉ₁·G̃_T grows with width, automatically picking the right complexity.

Setting: d=50, n=100, delta=0.5, gaussian mean shift. 50 reps per width.
Single fixed C1 calibrated at width=200 (the architecture used elsewhere).
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
    _to_tensor, _DEVICE
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'srm_width')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_SAMPLES = 100
DIM = 50
DELTA = 0.5
FEATURE_DIM = 10
N_STEPS = 100
LR = 0.005
N_CAL = 5

WIDTHS = [50, 100, 200, 500, 1000]


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def gen(n, d, delta, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32) + delta
    return X, Y


def train_disc(X, Y, d, width, criterion, C1=0.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
    D = DeepDiscriminator(d, FEATURE_DIM, width).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        opt.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        if criterion == 'cpmmd':
            spec = torch.tensor(1.0, device=_DEVICE)
            for name, p in D.named_parameters():
                if 'weight' in name:
                    spec = spec * torch.linalg.matrix_norm(p, ord=2)
            G = spec * data_norm / sigma
            loss = -(mmd2 - C1 * G)
        elif criterion == 'liu':
            n_ = f_x.size(0)
            K_xx = torch.exp(-torch.cdist(f_x, f_x) ** 2 / (sigma ** 2 + 1e-8))
            K_yy = torch.exp(-torch.cdist(f_y, f_y) ** 2 / (sigma ** 2 + 1e-8))
            K_xy = torch.exp(-torch.cdist(f_x, f_y) ** 2 / (sigma ** 2 + 1e-8))
            H = K_xx + K_yy - K_xy - K_xy.t()
            tau2 = H.mean(dim=1).var() + 1e-8
            loss = -(mmd2 / torch.sqrt(tau2))
        else:
            loss = -mmd2
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        prod = 1.0
        for name, p in D.named_parameters():
            if 'weight' in name:
                prod *= torch.linalg.matrix_norm(p, ord=2).item()
    return D, sigma_val, prod


def perm_test(D, X, Y, sigma, seed):
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    Z_t = _to_tensor(Z).to(_DEVICE)
    with torch.no_grad():
        f_all = D(Z_t).cpu().numpy()
    obs = compute_mmd_u_statistic(f_all[:m], f_all[m:], 'gaussian', sigma)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = rng.permutation(2 * m)
        null[i] = compute_mmd_u_statistic(f_all[perm[:m]], f_all[perm[m:]],
                                           'gaussian', sigma)
    thr = float(np.percentile(null, (1 - ALPHA) * 100))
    return int(obs > thr)


def calibrate_c1(X, Y, d, width, seed):
    rng = np.random.default_rng(seed + 999)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    ratios = []
    for i in range(N_CAL):
        perm = rng.permutation(2 * m)
        Xp, Yp = Z[perm[:m]], Z[perm[m:]]
        try:
            D, sigma, prod = train_disc(Xp, Yp, d, width, 'plain', seed=seed + 100 + i)
            X_t = _to_tensor(Xp).to(_DEVICE); Y_t = _to_tensor(Yp).to(_DEVICE)
            with torch.no_grad():
                f_x, f_y = D(X_t), D(Y_t)
                mmd2 = mmd_loss_torch(f_x, f_y, torch.tensor(sigma, device=_DEVICE)).item()
            data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
            G = prod * data_norm / max(sigma, 1e-8)
            if G > 1e-10:
                ratios.append(abs(mmd2) / G)
        except Exception:
            pass
    return float(np.percentile(ratios, 95)) if ratios else 0.01


def run_width(width, n_reps=N_REPS, C1=None):
    seed_base = width * 13
    if C1 is None:
        # Calibrate C1 at this width using seed=0 split
        X0, Y0 = gen(N_SAMPLES, DIM, DELTA, seed=seed_base)
        C1 = calibrate_c1(X0[:N_SAMPLES//2], Y0[:N_SAMPLES//2], DIM, width, seed=seed_base)
    log(f"  width={width}: C1 = {C1:.5f}")

    pow_plain = pow_cp = pow_liu = 0
    spec_plain_vals, spec_cp_vals, spec_liu_vals = [], [], []
    t0 = time.time()
    for rep in range(n_reps):
        seed = seed_base + rep * 7
        X, Y = gen(N_SAMPLES * 2, DIM, DELTA, seed=seed)
        Xtr, Ytr = X[:N_SAMPLES], Y[:N_SAMPLES]
        Xte, Yte = X[N_SAMPLES:], Y[N_SAMPLES:]
        # Plain
        D, sigma, prod = train_disc(Xtr, Ytr, DIM, width, 'plain', seed=seed)
        pow_plain += perm_test(D, Xte, Yte, sigma, seed=seed)
        spec_plain_vals.append(prod)
        # Liu
        D, sigma, prod = train_disc(Xtr, Ytr, DIM, width, 'liu', seed=seed)
        pow_liu += perm_test(D, Xte, Yte, sigma, seed=seed)
        spec_liu_vals.append(prod)
        # CP-MMD
        D, sigma, prod = train_disc(Xtr, Ytr, DIM, width, 'cpmmd', C1=C1, seed=seed)
        pow_cp += perm_test(D, Xte, Yte, sigma, seed=seed)
        spec_cp_vals.append(prod)
        if (rep + 1) % 10 == 0:
            log(f"    rep {rep+1}/{n_reps} ({time.time()-t0:.0f}s)")
    return {
        'width': width,
        'C1': C1,
        'power_plain': pow_plain / n_reps,
        'power_liu':   pow_liu   / n_reps,
        'power_cpmmd': pow_cp / n_reps,
        'spec_plain_mean': float(np.mean(spec_plain_vals)),
        'spec_liu_mean':   float(np.mean(spec_liu_vals)),
        'spec_cpmmd_mean': float(np.mean(spec_cp_vals)),
        'n_reps': n_reps,
    }


def main():
    log("=" * 70)
    log(f"SRM width sweep: d={DIM}, n={N_SAMPLES}, delta={DELTA}, {N_REPS} reps")
    log(f"Widths: {WIDTHS}")
    log(f"Device: {_DEVICE}")
    log("=" * 70)

    # Calibrate C1 ONCE at the reference width=200 used elsewhere in the paper.
    # Apply this fixed C1 across all widths -- this is the SRM message:
    # one moderate C1 keeps the penalty active at every complexity level.
    log("\nCalibrating reference C1 at width=200...")
    X0, Y0 = gen(N_SAMPLES, DIM, DELTA, seed=0)
    C1_REF = calibrate_c1(X0[:N_SAMPLES//2], Y0[:N_SAMPLES//2], DIM, 200, seed=0)
    if C1_REF < 1e-5:
        C1_REF = 0.01  # floor at typical paper-wide value
    log(f"Reference C1 = {C1_REF:.5f} (applied to all widths)")

    rows = []
    for w in WIDTHS:
        log(f"\n--- width={w} ---")
        t0 = time.time()
        res = run_width(w, C1=C1_REF)
        log(f"  Plain  power = {res['power_plain']:.3f}  spec={res['spec_plain_mean']:.2e}")
        log(f"  Liu    power = {res['power_liu']:.3f}  spec={res['spec_liu_mean']:.2e}")
        log(f"  CP-MMD power = {res['power_cpmmd']:.3f}  spec={res['spec_cpmmd_mean']:.2e}")
        log(f"  width time: {time.time()-t0:.1f}s")
        rows.append(res)

    out = os.path.join(RESULTS_DIR, 'srm_width_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"\nSaved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
