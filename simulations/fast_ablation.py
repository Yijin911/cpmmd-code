"""Fast vectorized helpers for the §5.2 calibration ablation.

Two speedups over the original per-permutation `compute_mmd_u_statistic`:
  (a) precompute the pooled gram matrix once per test, reshuffle labels
      with binary masks (vectorized across permutations);
  (b) J_CP bandwidth optimization that subtracts the per-h penalty
      C1 * ||D||_F / (sigma * N) from the empirical squared MMD.

Both run on `_DEVICE` (cuda when available, else cpu).
"""

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from mmd_core import _DEVICE, _to_tensor, _gaussian_gram, _laplacian_gram


_KERNEL_GRAM = {
    'gaussian': _gaussian_gram,
    'laplacian': _laplacian_gram,
}


def pooled_gram(X, Y, kernel_type, sigma):
    """Return the (m+n, m+n) gram matrix of pooled samples on _DEVICE."""
    X = _to_tensor(X).to(_DEVICE)
    Y = _to_tensor(Y).to(_DEVICE)
    Z = torch.cat([X, Y], dim=0)
    return _KERNEL_GRAM[kernel_type](Z, Z, sigma)


def mmd_from_gram(K, m, n):
    """Unbiased MMD^2 U-statistic from a pooled gram matrix K (m+n, m+n).

    First m rows/cols are X, next n are Y.
    """
    K_xx = K[:m, :m]
    K_yy = K[m:, m:]
    K_xy = K[:m, m:]
    xx = (K_xx.sum() - torch.diagonal(K_xx).sum()) / (m * (m - 1))
    yy = (K_yy.sum() - torch.diagonal(K_yy).sum()) / (n * (n - 1))
    xy = K_xy.sum() / (m * n)
    return (xx + yy - 2.0 * xy).item()


def batched_mmd_from_gram(K, labels, m, n):
    """Batched unbiased MMD^2 across many label vectors.

    Parameters
    ----------
    K : tensor (N, N)        pooled gram matrix on _DEVICE
    labels : tensor (B, N)   each row in {0,1}, sum to n (i.e. n ones = Y)
    m, n : int               sample sizes

    Returns
    -------
    tensor (B,)              MMD^2 for each labelling
    """
    N = m + n
    K_off = K - torch.diag(torch.diagonal(K))
    E_y = labels.to(K.dtype)
    E_x = 1.0 - E_y
    xx = ((E_x @ K_off) * E_x).sum(dim=1) / (m * (m - 1))
    yy = ((E_y @ K_off) * E_y).sum(dim=1) / (n * (n - 1))
    xy = ((E_x @ K) * E_y).sum(dim=1) / (m * n)
    return xx + yy - 2.0 * xy


def permutation_pvalue(X, Y, sigma, kernel_type='gaussian',
                       n_perm=200, alpha=0.05, rng=None,
                       batch_size=200):
    """Permutation p-value at a fixed kernel.

    Precomputes the pooled gram matrix once; permutations are applied to
    binary label vectors and processed in batches on _DEVICE.

    Returns
    -------
    reject : int (0/1)
    """
    if rng is None:
        rng = np.random.default_rng()
    m, n = len(X), len(Y)
    N = m + n

    K = pooled_gram(X, Y, kernel_type, sigma)
    obs_labels = torch.zeros(1, N, device=_DEVICE)
    obs_labels[0, m:] = 1.0
    obs = batched_mmd_from_gram(K, obs_labels, m, n).item()

    null_vals = []
    base = np.zeros(N, dtype=np.float32)
    base[m:] = 1.0
    for start in range(0, n_perm, batch_size):
        b = min(batch_size, n_perm - start)
        batch = np.empty((b, N), dtype=np.float32)
        for i in range(b):
            batch[i] = base[rng.permutation(N)]
        L = torch.from_numpy(batch).to(_DEVICE)
        vals = batched_mmd_from_gram(K, L, m, n)
        null_vals.append(vals.detach().cpu().numpy())
    null_arr = np.concatenate(null_vals)
    threshold = float(np.percentile(null_arr, (1 - alpha) * 100))
    return int(obs > threshold)


def optimize_bandwidth_jcp(X, Y, C1, kernel_type, search_space,
                           data_norm_F=None):
    """Bandwidth that maximizes J_CP(sigma) = MMD^2(sigma) - C1 * ||D||_F/(sigma*N).

    ``data_norm_F``: optional pre-computed ||D||_F of the pooled data (in the
    feature space that h acts on). If None, computed from raw X,Y.
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = torch.cat([X_t, Y_t], dim=0)
    N = Z.shape[0]
    if data_norm_F is None:
        data_norm_F = float(torch.linalg.norm(Z).item())
    m, n = X_t.shape[0], Y_t.shape[0]

    def neg_jcp(sigma):
        K = _KERNEL_GRAM[kernel_type](Z, Z, sigma)
        mmd = mmd_from_gram(K, m, n)
        penalty = C1 * data_norm_F / (sigma * N)
        return -(mmd - penalty)

    res = minimize_scalar(neg_jcp, bounds=search_space, method='bounded')
    return {
        'optimal_sigma': float(res.x),
        'jcp_value': -float(res.fun),
        'data_norm_F': data_norm_F,
    }


def optimize_bandwidth_plain(X, Y, kernel_type, search_space):
    """Plain MMD^2 argmax over sigma (used inside calibration). GPU path."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    Z = torch.cat([X_t, Y_t], dim=0)
    m, n = X_t.shape[0], Y_t.shape[0]

    def neg_mmd(sigma):
        K = _KERNEL_GRAM[kernel_type](Z, Z, sigma)
        return -mmd_from_gram(K, m, n)

    res = minimize_scalar(neg_mmd, bounds=search_space, method='bounded')
    return {
        'optimal_sigma': float(res.x),
        'max_mmd': -float(res.fun),
    }
