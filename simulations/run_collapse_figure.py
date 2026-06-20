"""
Generate variance-collapse figure: trajectories under H_0 for J_Liu vs J_CP,
plus the layer spectral-norm product showing implicit regularisation.

Trains discriminators under H_0 (P = Q = N(0, I_10)) and logs per-step
values of J_Liu, J_CP, the empirical MMD, and Pi^(t) = prod_j ||W_j^(t)||_2.
Two training runs per overlay: J_Liu (collapses) and J_CP (does not).
Produces figures/collapse_trajectory.pdf as a 2x2 grid (4 panels).
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

# Configuration
DIM = 10
N_SAMPLES = 200
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 200
LR = 0.005
N_RUNS = 5
SEED_BASE = 42
C1_CP = 0.001

# Collapse-detection thresholds (cf. Section 2.5 of the paper)
THRESH_MMD = 0.01
THRESH_JLIU = 10.0

# Plot styling
LIU_COLOR = '#d62728'   # red
CP_COLOR = '#1f77b4'    # blue
MMD_COLOR = '#2ca02c'   # green
PI_PLAIN_COLOR = '#d62728'  # red, dashed (plain max trained under J_Liu)
PI_CP_COLOR = '#1f77b4'     # blue, solid (CP-MMD)
THRESHOLD_COLOR = '#7f7f7f'  # grey

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    # Half-frame: keep bottom + left spines, drop top + right.
    'axes.spines.top': False,
    'axes.spines.right': False,
    # Subtle dotted grid behind data.
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
})


def generate_null_data(n, d, seed):
    """P = Q = N(0, I_d)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = rng.randn(n, d)
    return X, Y


def train_and_log(X, Y, criterion='liu', C1=C1_CP):
    """Train discriminator and log per-step traces.

    Returns
    -------
    j_liu_trace : list[float]
    j_cp_trace : list[float]
    mmd_trace : list[float]
    pi_trace : list[float]   product of layer spectral norms at every step
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    j_liu_trace, j_cp_trace, mmd_trace, pi_trace = [], [], [], []

    Z = np.concatenate([X, Y], axis=0)
    data_norm = np.linalg.norm(Z) / np.sqrt(len(Z))

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

        # Spectral-norm product Pi^(t) = prod_j ||W_j^(t)||_2
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_h = spec_prod * data_norm / sigma
        j_cp = (mmd2 - C1 * G_h).item()

        j_liu_trace.append(j_liu)
        j_cp_trace.append(j_cp)
        mmd_trace.append(mmd2.item())
        pi_trace.append(spec_prod.item())

        if criterion == 'liu':
            loss = -mmd2 / sigma_hat
        else:
            loss = -(mmd2 - C1 * G_h)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

    return j_liu_trace, j_cp_trace, mmd_trace, pi_trace


def main():
    print("Generating variance-collapse trajectory figure (2x2 grid)...")

    # Layout (causal narrative, top-left -> top-right -> bottom-left -> bottom-right):
    #   (a) MMD (signal stays small under H_0)
    #   (b) Pi^(t) (mechanism: layer norms diverge under J_Liu, plateau under J_CP)
    #   (c) J_Liu (consequence: criterion diverges because tau collapses with Pi)
    #   (d) J_CP (consequence: criterion stays negative, structurally precluding (c))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    ax_mmd, ax_pi = axes[0, 0], axes[0, 1]
    ax_jliu, ax_jcp = axes[1, 0], axes[1, 1]

    steps = list(range(N_STEPS))

    # Run N_RUNS independent trials. For each trial, train BOTH J_Liu and J_CP
    # so that the spectral-norm-product panel can compare CP-MMD vs plain max.
    pi_liu_runs, pi_cp_runs = [], []

    for run in range(N_RUNS):
        seed = SEED_BASE + run * 100
        X, Y = generate_null_data(N_SAMPLES, DIM, seed)

        # Liu-trained run
        np.random.seed(seed); torch.manual_seed(seed)
        j_liu_l, j_cp_l, mmd_l, pi_l = train_and_log(X, Y, criterion='liu')
        pi_liu_runs.append(pi_l)

        # CP-trained run (re-seed so init matches Liu run for clean comparison)
        np.random.seed(seed); torch.manual_seed(seed)
        _, _, _, pi_c = train_and_log(X, Y, criterion='cp')
        pi_cp_runs.append(pi_c)

        alpha = 0.4 if run > 0 else 0.85
        lw = 1.0 if run > 0 else 1.6

        # Panel (a) Empirical MMD: collapses to 0 under J_Liu training
        ax_mmd.plot(steps, mmd_l, color=MMD_COLOR, alpha=alpha, lw=lw,
                    label=r'$\widehat{\gamma}^2_{k,u}$' if run == 0 else None)
        # Panel (b) J_Liu: diverges
        ax_jliu.plot(steps, j_liu_l, color=LIU_COLOR, alpha=alpha, lw=lw,
                     label=r'$J_{\mathrm{Liu}}$' if run == 0 else None)
        # Panel (c) J_CP: stays negative
        ax_jcp.plot(steps, j_cp_l, color=CP_COLOR, alpha=alpha, lw=lw,
                    label=r'$J_{\mathrm{CP}}$' if run == 0 else None)

    # Panel (b): spectral-norm product Pi^(t), J_Liu vs J_CP ascent.
    # Both panel-(b) traces use dashed line style so 'solid' globally
    # encodes 'criterion value' (panels c, d) and 'dashed' encodes
    # 'spectral-norm trajectory' (panel b). This avoids the color
    # collision where panel (d)'s blue solid J_CP line was the same
    # visual encoding as a J_CP-ascent Pi trajectory.
    pi_liu_arr = np.array(pi_liu_runs)  # (N_RUNS, N_STEPS)
    pi_cp_arr = np.array(pi_cp_runs)
    for run in range(N_RUNS):
        alpha = 0.4 if run > 0 else 0.85
        lw = 1.0 if run > 0 else 1.6
        ax_pi.plot(steps, pi_liu_arr[run], color=PI_PLAIN_COLOR, alpha=alpha,
                   lw=lw, ls='--')
        ax_pi.plot(steps, pi_cp_arr[run], color=PI_CP_COLOR, alpha=alpha,
                   lw=lw, ls='--')

    # ----- Formatting -----
    # Per-panel legends are dropped: (a)-(c) y-axis labels already identify
    # the metric; the only line-encoding info that is not already in the
    # axes is panel (d)'s training-method distinction, which is conveyed
    # by the unified figure-level legend strip placed below the 2x2 grid.

    # (a) Empirical MMD
    ax_mmd.set_xlabel('Training step')
    ax_mmd.set_ylabel(r'$\widehat{\gamma}^2_{k,u}$')
    ax_mmd.set_title(r'(a) Empirical MMD under $H_0$')

    # (b) Spectral-norm product Pi (mechanism panel) with direct labels
    # at line right endpoints (no legend box).
    ax_pi.set_xlabel('Training step')
    ax_pi.set_ylabel(r'$\Pi^{(t)}$')
    ax_pi.set_title(r'(b) Spectral-norm product: plateau vs. divergence')
    ax_pi.set_yscale('log')
    # Endpoint labels: place text just past the right end of the primary
    # (run 0) trace for each ascent procedure, color-matched.
    last_x = N_STEPS - 1
    ax_pi.annotate(r'$J_{\mathrm{Liu}}$',
                   xy=(last_x, pi_liu_arr[0, -1]),
                   xytext=(6, 0), textcoords='offset points',
                   color=PI_PLAIN_COLOR, fontsize=12, va='center',
                   annotation_clip=False)
    ax_pi.annotate(r'$J_{\mathrm{CP}}$',
                   xy=(last_x, pi_cp_arr[0, -1]),
                   xytext=(6, 0), textcoords='offset points',
                   color=PI_CP_COLOR, fontsize=12, va='center',
                   annotation_clip=False)

    # (c) J_Liu
    ax_jliu.set_xlabel('Training step')
    ax_jliu.set_ylabel(r'$J_{\mathrm{Liu}}$')
    ax_jliu.set_title(r'(c) $J_{\mathrm{Liu}}$ diverges to $\infty$')
    ax_jliu.set_yscale('symlog', linthresh=10)

    # (d) J_CP -- keep y=0 line; J_CP < 0 is the structural claim
    ax_jcp.set_xlabel('Training step')
    ax_jcp.set_ylabel(r'$J_{\mathrm{CP}}$')
    ax_jcp.set_title(r'(d) $J_{\mathrm{CP}}$ stays strictly negative')
    ax_jcp.axhline(y=0, color=THRESHOLD_COLOR, ls=':', lw=0.9)

    # No unified bottom legend: each panel's content is identified by
    # its y-axis label and title; panel (b)'s two traces are tagged by
    # direct color-matched endpoint labels (placed above).
    fig.tight_layout()
    fig.suptitle('', y=0.99)  # caption supplied by paper.tex

    outpath = os.path.join(FIGURES_DIR, 'collapse_trajectory.pdf')
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.savefig(outpath.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"Saved {outpath}")
    print(f"Saved {outpath.replace('.pdf', '.png')}")


if __name__ == '__main__':
    main()
