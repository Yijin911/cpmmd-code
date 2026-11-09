"""
Generate variance-collapse figure: J_Liu vs J_CP trajectories under H0.

Trains discriminators under H0 (P = Q = N(0, I_10)) and logs per-step
values of J_Liu and J_CP, showing that J_Liu diverges while J_CP stays
negative. Produces figures/collapse_trajectory.pdf.
"""

import os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embedded TrueType, no Type 3 fonts
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmd_core import (
    DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)

warnings.filterwarnings('ignore')

# Configuration
DIM = 10
N_SAMPLES = 200
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 200
LR = 0.005
N_RUNS = 5  # number of independent runs to overlay
SEED_BASE = 42

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_null_data(n, d, seed):
    """P = Q = N(0, I_d)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = rng.randn(n, d)
    return X, Y


def train_and_log(X, Y, criterion='liu', C1=0.001):
    """Train discriminator and return per-step J_Liu, J_CP, MMD, and Pi values."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    j_liu_trace = []
    j_cp_trace = []
    mmd_trace = []
    pi_trace = []  # product of layer spectral norms

    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))

    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)

        # Compute J_Liu components
        m = f_x.shape[0]
        K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
        K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
        h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
        var_h = torch.var(h_x).clamp(min=1e-10)
        sigma_hat = torch.sqrt(4 * var_h / m)
        j_liu = (mmd2 / sigma_hat).item()

        # Compute G_spectral and Pi for J_CP
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_h = spec_prod * data_norm / sigma
        j_cp = (mmd2 - C1 * G_h).item()
        pi_trace.append(spec_prod.item())

        j_liu_trace.append(j_liu)
        j_cp_trace.append(j_cp)
        mmd_trace.append(mmd2.item())

        # Train using the specified criterion
        if criterion == 'liu':
            loss = -mmd2 / sigma_hat
        elif criterion == 'plain':
            loss = -mmd2
        else:
            loss = -(mmd2 - C1 * G_h)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

    return j_liu_trace, j_cp_trace, mmd_trace, pi_trace


def main():
    print("Generating variance-collapse trajectory figure (4 panels)...")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Colours
    liu_color = '#d62728'   # red
    cp_color = '#1f77b4'    # blue
    plain_color = '#ff7f0e' # orange

    # Run multiple independent trials under H0 with Liu's criterion
    for run in range(N_RUNS):
        seed = SEED_BASE + run * 100
        X, Y = generate_null_data(N_SAMPLES, DIM, seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        j_liu, j_cp, mmd, pi_liu = train_and_log(X, Y, criterion='liu', C1=0.001)
        steps = range(N_STEPS)
        alpha = 0.4 if run > 0 else 0.8
        lw = 1.0 if run > 0 else 1.5
        label_liu = r'$J_{\mathrm{Liu}}$' if run == 0 else None
        label_cp = r'$J_{\mathrm{CP}}$' if run == 0 else None
        label_mmd = r'$\widehat{\gamma}^2_{k,u}$' if run == 0 else None

        axes[0].plot(steps, j_liu, color=liu_color, alpha=alpha, lw=lw, label=label_liu)
        axes[1].plot(steps, j_cp, color=cp_color, alpha=alpha, lw=lw, label=label_cp)
        axes[2].plot(steps, mmd, color='#2ca02c', alpha=alpha, lw=lw, label=label_mmd)

    # Panel (d): Pi(t) for CP-MMD vs Plain under H0
    for run in range(N_RUNS):
        seed = SEED_BASE + run * 100
        X, Y = generate_null_data(N_SAMPLES, DIM, seed)

        # CP-MMD trajectory
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, _, _, pi_cp = train_and_log(X, Y, criterion='cp', C1=0.001)

        # Plain trajectory (same init)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, _, _, pi_plain = train_and_log(X, Y, criterion='plain', C1=0.001)

        alpha = 0.4 if run > 0 else 0.8
        lw = 1.0 if run > 0 else 1.5
        label_cp_pi = r'CP-MMD $\Pi^{(t)}$' if run == 0 else None
        label_plain_pi = r'Plain $\Pi^{(t)}$' if run == 0 else None
        axes[3].plot(steps, pi_cp, color=cp_color, alpha=alpha, lw=lw, label=label_cp_pi)
        axes[3].plot(steps, pi_plain, color=plain_color, alpha=alpha, lw=lw,
                     linestyle='--', label=label_plain_pi)

    # Formatting
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel(r'$J_{\mathrm{Liu}}$')
    axes[0].set_title(r'(a) $J_{\mathrm{Liu}}$ diverges under $H_0$')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].set_yscale('symlog', linthresh=10)

    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel(r'$J_{\mathrm{CP}}$')
    axes[1].set_title(r'(b) $J_{\mathrm{CP}}$ stays negative under $H_0$')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='$0$')
    axes[1].legend(loc='lower left', fontsize=8)

    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel(r'$\widehat{\gamma}^2_{k,u}$')
    axes[2].set_title(r'(c) Empirical MMD stays small')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=8)

    axes[3].set_xlabel('Training step')
    axes[3].set_ylabel(r'$\Pi^{(t)} = \prod_j \|W_j\|_2$')
    axes[3].set_title(r'(d) Spectral-norm product $\Pi^{(t)}$')
    axes[3].legend(loc='upper left', fontsize=8)
    axes[3].grid(alpha=0.2)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'collapse_trajectory.pdf')
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.savefig(outpath.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"Saved {outpath}")
    print(f"Saved {outpath.replace('.pdf', '.png')}")


if __name__ == '__main__':
    main()
