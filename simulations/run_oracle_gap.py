"""
Oracle-gap validation experiment for the JMLR paper.

Validates Theorem 2 (oracle inequality) by plotting the empirical oracle gap
gamma^2(h*) - gamma^2(h_hat) against the theoretical bound 2*B_N(delta)
across sample sizes.

Also validates the SRM corollary by selecting across model classes of
varying network width and checking that CP-MMD picks the optimal class.
"""

import os, sys, warnings, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.optimize import minimize_scalar
from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, train_discriminator,
    _to_tensor, _sq_euclidean_distances, _DEVICE
)
from calibration_engine import compute_S

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'oracle_gap')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DIM = 20
DELTA_SHIFT = 0.5
FEATURE_DIM = 10
N_STEPS = 100
LR = 0.005
N_REPS = 50
ALPHA = 0.05
CONFIDENCE_DELTA = 0.05


def generate_data(n, d, delta, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = rng.randn(n, d) + delta
    return X, Y


def population_mmd2_isotropic(sigma, mu_diff_sq, d):
    # k(u,u') = exp(-||u-u'||^2 / sigma^2) on P=N(0,I_d), Q=N(mu,I_d);
    # the cross-term integral reduces to (sigma^2/(sigma^2+4))^{d/2} times an exp factor.
    a = (sigma ** 2) / (sigma ** 2 + 4.0)
    return 2.0 * (a ** (d / 2.0)) * (1.0 - np.exp(-mu_diff_sq / (sigma ** 2 + 4.0)))


def train_deep_kernel(X, Y, d, width, C1=None):
    """Train a deep kernel and return MMD and complexity."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(d, FEATURE_DIM, width).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)

        if C1 is not None:
            spec_prod = torch.tensor(1.0, device=_DEVICE)
            for name, p in D.named_parameters():
                if 'weight' in name:
                    spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
            Z = np.concatenate([X, Y], axis=0)
            data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
            G_h = spec_prod * data_geom / sigma
            J = mmd2 - C1 * G_h
            (-J).backward()
        else:
            (-mmd2).backward()

        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()

    # Compute G_spectral
    spec_prod = 1.0
    with torch.no_grad():
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod *= torch.linalg.matrix_norm(p, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    G = spec_prod * np.linalg.norm(Z) / (np.sqrt(len(Z)) * sigma_val)

    return mmd_val, G, sigma_val, D


def experiment_oracle_gap():
    """Continuous-bandwidth oracle-inequality validation matching paper section 5.1."""
    print("=" * 60)
    print("EXPERIMENT 1: Oracle gap vs sample size (continuous Gaussian bandwidth)")
    print("=" * 60)

    sample_sizes = [50, 100, 200, 500]
    sigma_search = (0.1, 30.0)
    mu_diff_sq = (DELTA_SHIFT ** 2) * DIM

    pop_objective = lambda s: -population_mmd2_isotropic(s, mu_diff_sq, DIM)
    star_res = minimize_scalar(pop_objective, bounds=sigma_search, method='bounded',
                               options={'xatol': 1e-6})
    sigma_star = float(star_res.x)
    gamma_star = -float(star_res.fun)
    print(f"\n  Population-best bandwidth sigma* = {sigma_star:.4f}")
    print(f"  Population MMD^2 at sigma*       = {gamma_star:.6f}")

    sigma_min_search = sigma_search[0]
    M_Lip_worst = (2.0 / sigma_min_search) * np.sqrt(2.0 / np.e)

    results = {n: {'gaps': [], 'bounds': []} for n in sample_sizes}

    for n in sample_sizes:
        print(f"\n  n = {n}")
        for rep in range(N_REPS):
            seed = rep * 1000 + n
            np.random.seed(seed)

            X, Y = generate_data(n, DIM, DELTA_SHIFT, seed)
            m_tr = n // 2
            X_tr, Y_tr = X[:m_tr], Y[:m_tr]

            hat_res = optimize_bandwidth(X_tr, Y_tr, 'gaussian', search_space=sigma_search)
            sigma_hat = float(hat_res['optimal_sigma'])
            gamma_hat = population_mmd2_isotropic(sigma_hat, mu_diff_sq, DIM)
            gap = abs(gamma_star - gamma_hat)

            X_te, Y_te = X[m_tr:], Y[m_tr:]
            Z = np.concatenate([X_te, Y_te], axis=0)
            D_frob = np.linalg.norm(Z)
            N_eff = len(X_te) + len(Y_te)
            # B_N(delta): Lipschitz-envelope x data-geometry over the bandwidth interval,
            # plus McDiarmid for the bounded-differences tail. Lipschitz envelope is
            # achieved at the smallest searched sigma.
            G_class = M_Lip_worst * D_frob / N_eff
            bound = 2 * (G_class / np.sqrt(N_eff)
                         + 8 * np.sqrt(np.log(2 / CONFIDENCE_DELTA) / N_eff))

            results[n]['gaps'].append(gap)
            results[n]['bounds'].append(bound)

        mean_gap = np.mean(results[n]['gaps'])
        mean_bound = np.mean(results[n]['bounds'])
        print(f"    Mean gap: {mean_gap:.6f}, Mean bound: {mean_bound:.6f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ns = np.array(sample_sizes)
    mean_gaps = np.array([np.mean(results[n]['gaps']) for n in ns])
    std_gaps = np.array([np.std(results[n]['gaps']) for n in ns])
    mean_bounds = np.array([np.mean(results[n]['bounds']) for n in ns])

    ax.errorbar(ns, mean_gaps, yerr=std_gaps, fmt='o-', color='#1f77b4',
                label=r'Empirical gap $|\gamma_k^2(h^*) - \gamma_k^2(\hat{h})|$', capsize=3)
    ax.plot(ns, mean_bounds, 's--', color='#d62728',
            label=r'Theoretical bound $2B_N(\delta)$')
    ax.set_xlabel(r'Sample size $n$')
    ax.set_ylabel(r'Gap / Bound')
    ax.set_title('Squared-MMD concentration check')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    outpath = os.path.join(FIGURES_DIR, 'oracle_gap.pdf')
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.savefig(outpath.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"\nSaved {outpath}")

    csv_path = os.path.join(RESULTS_DIR, 'oracle_gap.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'mean_gap', 'std_gap', 'mean_bound'])
        for n_, mg, sg, mb in zip(ns, mean_gaps, std_gaps, mean_bounds):
            writer.writerow([int(n_), float(mg), float(sg), float(mb)])
    print(f"Saved {csv_path}")

    return results


def experiment_srm():
    """Validate SRM corollary: CP-MMD selects the right model class."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: SRM model class selection")
    print("=" * 60)

    widths = [20, 50, 100, 200, 400]
    n = 200
    C1 = 0.01

    selection_counts = {w: 0 for w in widths}

    for rep in range(N_REPS):
        seed = rep * 1000
        np.random.seed(seed)
        torch.manual_seed(seed)

        X, Y = generate_data(n, DIM, DELTA_SHIFT, seed)
        m_tr = n // 2
        X_tr, Y_tr = X[:m_tr], Y[:m_tr]

        best_jcp = -np.inf
        best_width = None

        for w in widths:
            torch.manual_seed(seed + w)
            mmd_val, G_val, sigma_val, _ = train_deep_kernel(X_tr, Y_tr, DIM, w, C1=C1)
            jcp = mmd_val - C1 * G_val
            if jcp > best_jcp:
                best_jcp = jcp
                best_width = w

        selection_counts[best_width] = selection_counts.get(best_width, 0) + 1
        if (rep + 1) % 10 == 0:
            print(f"  Rep {rep+1}/{N_REPS}: selected width={best_width}, J_CP={best_jcp:.4f}")

    print("\nSRM selection frequencies:")
    for w in widths:
        freq = selection_counts[w] / N_REPS
        print(f"  Width {w:4d}: {selection_counts[w]:3d}/{N_REPS} ({freq:.1%})")

    # Save
    filepath = os.path.join(RESULTS_DIR, 'srm_selection.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['width', 'count', 'frequency'])
        for w in widths:
            writer.writerow([w, selection_counts[w], selection_counts[w] / N_REPS])
    print(f"Saved {filepath}")

    return selection_counts


def main():
    experiment_oracle_gap()
    experiment_srm()
    print("\nDone.")


if __name__ == '__main__':
    main()
