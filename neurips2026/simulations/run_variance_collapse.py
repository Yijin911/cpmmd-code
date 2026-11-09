"""
Variance-collapse table experiment for App. C (Table 3).

Trains a deep discriminator under H_0 (P = Q = N(0, I_10)) to maximize Liu's
ratio J_Liu, sweeping MLP widths {10, 50, 200, 400} with 10 seeds per width.
At the final step records J_Liu, gamma_hat^2, tau_hat_h, G_tilde, J_CP and a
binary "collapsed" flag (gamma_hat^2 < 0.01 and tau_hat_h < 0.001 and
J_Liu > 10). Per-width means and the collapse rate populate Table 3.

Output: results/variance_collapse/variance_collapse_results.csv
        (one row per (width, seed) plus a per-width summary printed to stdout)
"""

import os, sys, csv, warnings
import numpy as np
import torch
import torch.nn as nn

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE,
)

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'variance_collapse')
os.makedirs(RESULTS_DIR, exist_ok=True)

DIM = 10
N_SAMPLES = 200
FEATURE_DIM = 10
N_STEPS = 200
LR = 0.005
C1 = 0.001
WIDTHS = [10, 50, 200, 400]
N_SEEDS = 10
SEED_BASE = 42

COLLAPSE_MMD_THRESH = 0.01
COLLAPSE_TAU_THRESH = 0.001
COLLAPSE_JLIU_THRESH = 10.0


def generate_null_data(n, d, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = rng.randn(n, d)
    return X, Y


def train_liu(X, Y, hidden_width):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, hidden_width).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    Z = np.concatenate([X, Y], axis=0)
    data_norm = float(np.linalg.norm(Z) / np.sqrt(len(Z)))

    final = {}
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
        tau_h = torch.sqrt(4 * var_h / m)
        j_liu = mmd2 / tau_h

        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_tilde = spec_prod * data_norm / sigma
        j_cp = mmd2 - C1 * G_tilde

        if step == N_STEPS - 1:
            final = {
                'mmd2': float(mmd2.item()),
                'tau_h': float(tau_h.item()),
                'j_liu': float(j_liu.item()),
                'G_tilde': float(G_tilde.item()),
                'j_cp': float(j_cp.item()),
                'sigma': float(sigma.item()),
                'spec_prod': float(spec_prod.item()),
            }

        loss = -j_liu
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

    return final


def is_collapsed(rec):
    return (rec['mmd2'] < COLLAPSE_MMD_THRESH
            and rec['tau_h'] < COLLAPSE_TAU_THRESH
            and rec['j_liu'] > COLLAPSE_JLIU_THRESH)


def main():
    print(f"Device: {_DEVICE}")
    print(f"Widths: {WIDTHS}, seeds per width: {N_SEEDS}")
    print(f"Collapse criterion: gamma^2 < {COLLAPSE_MMD_THRESH} "
          f"AND tau_h < {COLLAPSE_TAU_THRESH} "
          f"AND J_Liu > {COLLAPSE_JLIU_THRESH}\n")

    csv_path = os.path.join(RESULTS_DIR, 'variance_collapse_results.csv')
    fieldnames = ['width', 'seed', 'mmd2', 'tau_h', 'j_liu',
                  'G_tilde', 'j_cp', 'sigma', 'spec_prod', 'collapsed']
    rows = []

    for w in WIDTHS:
        per_width = []
        for run in range(N_SEEDS):
            seed = SEED_BASE + run * 100
            np.random.seed(seed)
            torch.manual_seed(seed)
            X, Y = generate_null_data(N_SAMPLES, DIM, seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rec = train_liu(X, Y, w)
            rec['width'] = w
            rec['seed'] = seed
            rec['collapsed'] = int(is_collapsed(rec))
            rows.append(rec)
            per_width.append(rec)
            print(f"  width={w:>4d} seed={seed:>4d}  "
                  f"J_Liu={rec['j_liu']:>10.2f}  "
                  f"gamma^2={rec['mmd2']:>8.4f}  "
                  f"tau_h={rec['tau_h']:>9.2e}  "
                  f"G_tilde={rec['G_tilde']:>7.2f}  "
                  f"J_CP={rec['j_cp']:>7.4f}  "
                  f"collapsed={rec['collapsed']}")

        n = len(per_width)
        rate = sum(r['collapsed'] for r in per_width) / n
        m_jliu = np.mean([r['j_liu'] for r in per_width])
        m_mmd = np.mean([r['mmd2'] for r in per_width])
        m_tau = np.mean([r['tau_h'] for r in per_width])
        m_G = np.mean([r['G_tilde'] for r in per_width])
        m_jcp = np.mean([r['j_cp'] for r in per_width])
        print(f"  width={w}: collapse_rate={rate*100:.0f}%  "
              f"mean J_Liu={m_jliu:.0f}  mean gamma^2={m_mmd:.3f}  "
              f"mean tau_h={m_tau:.1e}  mean G_tilde={m_G:.0f}  "
              f"mean J_CP={m_jcp:.3f}\n")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
