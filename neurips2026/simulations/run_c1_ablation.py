"""
C1 ablation: sweep C_1 hat to validate that calibration picks the operationally optimal value.

Setting (fixed): deep kernel, d=20, n=200, delta=0.5 (mean shift).
Sweep C1 in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10} plus the calibrated value.
Report:
 - Power (50 reps)
 - Observed Pi^(t) at end of training (= product of spectral norms)
 - Effective kernel-gradient bound magnitude
Expected: U-curve with maximum at the calibrated C1. C1=0 is plain max
(may overfit under some settings); very large C1 over-penalises and kills power.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE, BestIterate
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'c1_ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_SAMPLES = 200
DIM = 20
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005
DELTA = 0.5


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def generate_data(n, d, delta=0.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32) + delta
    return X, Y


def spec_product(model):
    prod = 1.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name:
                prod *= torch.linalg.matrix_norm(p, ord=2).item()
    return prod


def train_cpmmd(X, Y, C1, seed=None):
    """Train discriminator maximising J_CP = hat_gamma^2 - C1 * G_spectral."""
    if seed is not None:
        torch.manual_seed(seed)
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    opt = torch.optim.Adam(params, lr=LR)

    best = BestIterate()  # h_hat = argmax_{0<=t<=T} J_CP(h^(t))
    for step in range(N_STEPS + 1):
        opt.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        spec = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec = spec * torch.linalg.matrix_norm(p, ord=2)
        G = spec * data_norm / sigma
        loss = -(mmd2 - C1 * G)
        best.update(-loss.item(), D, log_sigma)
        if step == N_STEPS:
            break
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    best.restore(D, log_sigma)

    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
    return D, sigma_val, spec_product(D)


def permutation_test(model, X, Y, sigma, n_perm=N_PERM, alpha=ALPHA, seed=None):
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    m = len(X)
    Z_t = _to_tensor(Z).to(_DEVICE)
    with torch.no_grad():
        f_all = model(Z_t).cpu().numpy()
    obs = compute_mmd_u_statistic(f_all[:m], f_all[m:], 'gaussian', sigma)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(2 * m)
        null[i] = compute_mmd_u_statistic(f_all[perm[:m]], f_all[perm[m:]],
                                           'gaussian', sigma)
    thr = float(np.percentile(null, (1 - alpha) * 100))
    return int(obs > thr)


def run_one_c1(C1_val, n_reps=N_REPS):
    """Return (power, mean_pi)."""
    rejects = 0
    valid = 0
    pi_values = []
    for rep in range(n_reps):
        seed = rep * 17 + int(C1_val * 10000) + 1
        n_total = N_SAMPLES * 2
        X_all, Y_all = generate_data(n_total, DIM, delta=DELTA, seed=seed)
        X_tr, X_te = X_all[:N_SAMPLES], X_all[N_SAMPLES:]
        Y_tr, Y_te = Y_all[:N_SAMPLES], Y_all[N_SAMPLES:]
        try:
            D, sigma, pi = train_cpmmd(X_tr, Y_tr, C1_val, seed=seed)
            rej = permutation_test(D, X_te, Y_te, sigma, seed=seed)
            rejects += rej
            pi_values.append(pi)
            valid += 1
        except Exception as e:
            log(f"    rep {rep}: exception {e}")
    rate = rejects / valid if valid > 0 else 0
    mean_pi = float(np.mean(pi_values)) if pi_values else float('nan')
    return rate, mean_pi, valid


def main():
    log("=" * 70)
    log(f"C1 ablation: d={DIM}, n={N_SAMPLES}, delta={DELTA}, reps={N_REPS}")
    log(f"Device: {_DEVICE}")
    log("=" * 70)

    # Sweep C1 values
    C1_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    rows = []
    for c1 in C1_values:
        t0 = time.time()
        power, mean_pi, valid = run_one_c1(c1)
        dt = time.time() - t0
        log(f"  C1={c1:10.4g}  power={power:.3f}  mean_Pi={mean_pi:.2f}  ({valid}/{N_REPS} valid, {dt:.1f}s)")
        rows.append({
            'C1': c1,
            'power': power,
            'mean_pi': mean_pi,
            'n_valid': valid,
            'n_reps': N_REPS,
        })

    out = os.path.join(RESULTS_DIR, 'c1_ablation_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"\nSaved {out}")
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
