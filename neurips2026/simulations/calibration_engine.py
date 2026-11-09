"""Complexity utilities for CP-MMD kernel selection.

Provides reusable building blocks for the trajectory-adaptive proxy
\widetilde G(h) used inside J_CP = \widehat\gamma^2_{k,u}(h) - \widehat C_1 \widetilde G(h)
(Eq. 9 / Lemma L.14 of the paper):

  - compute_S        : concentration term \sqrt{\ln(2/\delta)/(m+n)}.
  - compute_G        : Monte Carlo Gaussian complexity for a generic
                       optimisation objective (used for both bandwidth
                       optimisation and learned representations).
  - compute_G_at_model / compute_G_bound : Gaussian complexity at a fixed
                       trained model (empirical and Bartlett-style spectral
                       bound, respectively).
  - make_bandwidth_objective / make_deep_objective : factories that build
                       the inner objective handed to compute_G.

Per-kernel-class \widehat C_1 calibration itself is performed inline by
each experiment script as the (1-\alpha)-quantile of null-permutation
ratios \widehat\gamma^2_{k,u}(\widehat h_\pi) / \widetilde G(\widehat h_\pi)
(Proposition J.11), so it lives next to its accompanying optimisation loop
rather than in this module.
"""

import numpy as np


# ==================================================================
# G term  -  Gaussian complexity
# ==================================================================
def compute_G(X, Y, objective_fn, n_mc=200, d_out=None):
    """Monte Carlo estimate of the Gaussian complexity term.

    G = E_xi[ max_theta  f(Z, xi, theta) ] / (m + n)

    Parameters
    ----------
    X : ndarray (m, d)
    Y : ndarray (n, d)
    objective_fn : callable(Z, xi) -> float
        Given Z (N, d_in) and one draw xi (N, d_out), return the
        maximised inner quantity (already optimised over theta).
    n_mc : int
        Monte Carlo draws for the expectation.
    d_out : int or None
        Dimension of xi.  Defaults to X.shape[1] (matches input dim).
        Set to feature_dim for deep discriminator case where
        xi must match D(Z) output dimension.
    """
    N = len(X) + len(Y)
    d = d_out if d_out is not None else X.shape[1]
    Z = np.concatenate([X, Y], axis=0)

    vals = np.empty(n_mc)
    for k in range(n_mc):
        xi = np.random.randn(N, d)
        vals[k] = objective_fn(Z, xi)

    return np.mean(vals) / N


def make_bandwidth_objective(search_space):
    """Return an *objective_fn* for ``compute_G`` that maximises over sigma.

    Computes  max_{sigma in [sigma_l, sigma_h]} (sum_i xi_i^T Z_i) / sigma
    = max(s/sigma_l, s/sigma_h).
    """
    sigma_l, sigma_h = search_space

    def _objective(Z, xi):
        s = np.sum(xi * Z)
        return max(s / sigma_l, s / sigma_h)

    return _objective


def make_deep_objective(hidden_width, x_dim, feature_dim,
                        lr=0.005, weight_decay=0.0, n_steps=150,
                        grad_clip=5.0):
    """Return an *objective_fn* for ``compute_G`` with deep discriminator.

    Trains D to maximise  sum_i xi_i^T D(Z_i)  using the **same**
    optimizer and hyperparameters as the MMD discriminator training,
    so G tracks the function class actually explored by the optimiser.

    Parameters
    ----------
    hidden_width, x_dim, feature_dim : int
        Network architecture.
    lr, weight_decay, n_steps, grad_clip : float
        Must match the MMD training setup for consistency.
    """
    import torch
    import torch.nn as nn
    from mmd_core import DeepDiscriminator, _DEVICE

    def _objective(Z, xi):
        Z_t = torch.from_numpy(Z.astype(np.float32)).to(_DEVICE)
        xi_t = torch.from_numpy(xi.astype(np.float32)).to(_DEVICE)

        D = DeepDiscriminator(x_dim, feature_dim, hidden_width).to(_DEVICE)
        opt = torch.optim.Adam(D.parameters(), lr=lr,
                               weight_decay=weight_decay)

        for _ in range(n_steps):
            opt.zero_grad()
            out = D(Z_t)
            loss = -(xi_t * out).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip)
            opt.step()

        with torch.no_grad():
            return (xi_t * D(Z_t)).sum().item()

    return _objective


def compute_G_at_model(X, Y, model, n_mc=200):
    """Empirical Gaussian complexity at a FIXED trained model D*.

    G = E_xi[ || sum_i xi_i * D*(Z_i) || ] / N

    Parameters
    ----------
    model : torch.nn.Module   Trained discriminator (eval mode).
    n_mc : int                MC draws for the expectation.
    """
    import torch
    from mmd_core import _DEVICE

    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    Z_t = torch.from_numpy(Z.astype(np.float32)).to(_DEVICE)

    with torch.no_grad():
        f_z = model(Z_t)  # (N, feature_dim)

    norms = []
    for _ in range(n_mc):
        xi = torch.randn(N, 1).to(_DEVICE)
        weighted_sum = (f_z * xi).sum(dim=0)
        norms.append(weighted_sum.norm().item())

    return np.mean(norms) / N


def compute_G_bound(X, Y, model, sigma=None):
    """
    Theoretical Gaussian complexity upper bound for an MLP.

    Reference: Bartlett et al. (2017), "Spectrally-normalized margin bounds for neural networks."
    G = (\prod ||W_i||_2) * ||Z||_F / sqrt(N)
    """
    import torch

    spectral_product = 1.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            # ord=2 computes the spectral norm (largest singular value) for matrices
            spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()

    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    data_geom = np.linalg.norm(Z) / np.sqrt(N)

    G = spectral_product * data_geom
    if sigma is not None:
        G = G / sigma
    return G


# ==================================================================
# S term  -  concentration / tail
# ==================================================================
def compute_S(delta, m, n):
    """Concentration term:  sqrt( ln(2/delta) / (m + n) ).

    Parameters
    ----------
    delta : float   Confidence parameter (e.g. 0.05).
    m, n  : int     Sample sizes of the two groups.
    """
    return np.sqrt(np.log(2.0 / delta) / (m + n))
