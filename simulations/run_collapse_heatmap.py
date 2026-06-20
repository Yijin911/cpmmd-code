"""
Phase diagram of variance-collapse rate for J_Liu vs J_CP, over a grid
of (data dimension d, network width).

For each (d, width) cell we run N_REPS independent training runs under
H_0: P = Q = N(0, I_d), and compute the fraction satisfying the
collapse criteria (gamma_hat^2 < 0.01, tau_hat < 0.001, J_Liu > 10).

This panel demonstrates that variance collapse is not specific to the
single setting (d=10, width=200) shown in Figure 1, but occupies a
2D region of the architectural parameter space. CP-MMD's collapse rate
is essentially zero across the same region.

Saves intermediate data to results/collapse_heatmap.npz.
Produces figures/collapse_heatmap.pdf.
"""

import os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmd_core import (
    DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)

warnings.filterwarnings('ignore')

# Configuration (grid kept small for ~10-15 min runtime)
D_LIST = [2, 5, 10, 20]
WIDTH_LIST = [20, 50, 100, 200]
N_SAMPLES = 200
FEATURE_DIM = 10
N_STEPS = 200
LR = 0.005
N_REPS = 10
SEED_BASE = 1000
C1_CP = 0.001

# Collapse-detection thresholds
THRESH_MMD = 0.01
THRESH_TAU = 0.001
THRESH_JLIU = 10.0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    # Half-frame: keep bottom + left spines, drop top + right.
    # (Heatmap cells already define a visual rectangle, so this just
    # removes redundant frame chrome.)
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
})


def generate_null_data(n, d, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d), rng.randn(n, d)


def train_one(X, Y, d, width, criterion):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(d, FEATURE_DIM, width).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))

    last = {}
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
        j_liu = (mmd2 / sigma_hat).item()

        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_h = spec_prod * data_norm / sigma

        if criterion == 'liu':
            loss = -mmd2 / sigma_hat
        else:
            loss = -(mmd2 - C1_CP * G_h)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

        last = {
            'mmd': mmd2.item(),
            'tau': sigma_hat.item(),
            'j_liu': j_liu,
        }

    return last


def is_collapsed(stats):
    return (stats['mmd'] < THRESH_MMD
            and stats['tau'] < THRESH_TAU
            and stats['j_liu'] > THRESH_JLIU)


def run():
    print(f"Sweeping (d, width) over {len(D_LIST)}x{len(WIDTH_LIST)} cells, "
          f"{N_REPS} reps/cell, 2 criteria => "
          f"{len(D_LIST) * len(WIDTH_LIST) * N_REPS * 2} runs")

    rate_liu = np.zeros((len(D_LIST), len(WIDTH_LIST)))
    rate_cp = np.zeros_like(rate_liu)

    for di, d in enumerate(D_LIST):
        for wi, width in enumerate(WIDTH_LIST):
            n_collapse_liu = 0
            n_collapse_cp = 0
            for rep in range(N_REPS):
                seed = SEED_BASE + 1000 * di + 100 * wi + rep
                X, Y = generate_null_data(N_SAMPLES, d, seed)

                np.random.seed(seed); torch.manual_seed(seed)
                stats_l = train_one(X, Y, d, width, 'liu')
                if is_collapsed(stats_l):
                    n_collapse_liu += 1

                np.random.seed(seed); torch.manual_seed(seed)
                stats_c = train_one(X, Y, d, width, 'cp')
                if is_collapsed(stats_c):
                    n_collapse_cp += 1

            rate_liu[di, wi] = n_collapse_liu / N_REPS
            rate_cp[di, wi] = n_collapse_cp / N_REPS
            print(f"  d={d:>3}, width={width:>3}: "
                  f"J_Liu collapse {rate_liu[di, wi]:.2f}, "
                  f"J_CP collapse {rate_cp[di, wi]:.2f}")

    np.savez(os.path.join(RESULTS_DIR, 'collapse_heatmap.npz'),
             d_list=np.array(D_LIST), width_list=np.array(WIDTH_LIST),
             rate_liu=rate_liu, rate_cp=rate_cp,
             n_reps=N_REPS, n_samples=N_SAMPLES, n_steps=N_STEPS)
    print("Saved results/collapse_heatmap.npz")
    return rate_liu, rate_cp


def plot(rate_liu, rate_cp):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax_l, ax_c = axes

    for ax, rate, title in [
        (ax_l, rate_liu, r'(a) $J_{\mathrm{Liu}}$ collapse rate'),
        (ax_c, rate_cp, r'(b) $J_{\mathrm{CP}}$ collapse rate'),
    ]:
        im = ax.imshow(rate, vmin=0, vmax=1, cmap='Reds', aspect='auto',
                       origin='lower')
        ax.set_xticks(range(len(WIDTH_LIST)))
        ax.set_xticklabels(WIDTH_LIST)
        ax.set_yticks(range(len(D_LIST)))
        ax.set_yticklabels(D_LIST)
        ax.set_xlabel('Network width')
        ax.set_ylabel(r'Data dimension $d$')
        ax.set_title(title)
        for i in range(len(D_LIST)):
            for j in range(len(WIDTH_LIST)):
                text_color = 'white' if rate[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{rate[i, j]:.2f}', ha='center', va='center',
                        color=text_color, fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('collapse rate')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, 'collapse_heatmap.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"Saved {out}")


def load_cache():
    """Return (rate_liu, rate_cp) from results/collapse_heatmap.npz if the
    saved config matches the current one; else (None, None)."""
    cache = os.path.join(RESULTS_DIR, 'collapse_heatmap.npz')
    if not os.path.exists(cache):
        return None, None
    data = np.load(cache)
    if (list(data['d_list']) != D_LIST
            or list(data['width_list']) != WIDTH_LIST
            or int(data['n_reps']) != N_REPS):
        return None, None  # config drift, force re-simulation
    return data['rate_liu'], data['rate_cp']


def main():
    cached_liu, cached_cp = load_cache()
    if cached_liu is not None:
        print(f"Loaded cached collapse rates from results/collapse_heatmap.npz "
              f"(n_reps={N_REPS}); skipping simulation.")
        rate_liu, rate_cp = cached_liu, cached_cp
    else:
        rate_liu, rate_cp = run()
    plot(rate_liu, rate_cp)


if __name__ == '__main__':
    main()
