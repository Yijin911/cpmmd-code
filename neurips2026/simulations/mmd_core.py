"""Core MMD computation and bandwidth optimization using PyTorch.

Uses the vectorized expanded-form distance computation (x^2 + y^2 - 2xy)
from the existing codebase, which benchmarks ~2.4x faster than scipy.cdist
on CPU and supports GPU acceleration.

Also provides a deep discriminator for learned-representation MMD testing.
"""

import copy

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize_scalar


class BestIterate:
    """Keep the iterate with the largest criterion value along a trajectory.

    Implements the selection rule h_hat = argmax_{0 <= t <= T} J_CP(h^(t)):
    call update() with J_CP at every iterate t = 0, ..., T and restore() after
    the loop. Extra trainable tensors (e.g. log bandwidths) are passed after the
    model. Ties keep the earliest iterate; NaN values are ignored.
    """

    def __init__(self):
        self.value = -float('inf')
        self.state = None
        self.params = None

    def update(self, value, model, *params):
        if value > self.value:
            self.value = value
            self.state = copy.deepcopy(model.state_dict())
            self.params = [p.detach().clone() for p in params]

    def restore(self, model, *params):
        if self.state is None:
            return
        model.load_state_dict(self.state)
        with torch.no_grad():
            for p, v in zip(params, self.params):
                p.copy_(v)


def _sq_euclidean_distances(X, Y):
    """Compute pairwise squared Euclidean distances via expanded form."""
    X_sq = X.pow(2).sum(1).view(-1, 1)
    Y_sq = Y.pow(2).sum(1).view(1, -1)
    return (X_sq + Y_sq - 2 * X.mm(Y.t())).clamp(min=0.0)


def _l1_distances(X, Y):
    """Compute pairwise L1 (Manhattan) distances."""
    return torch.cdist(X, Y, p=1)


def _gaussian_gram(X, Y, sigma):
    """Gaussian RBF kernel: exp(-||x-y||^2 / sigma^2)."""
    return torch.exp(-_sq_euclidean_distances(X, Y) / (sigma ** 2))


def _laplacian_gram(X, Y, sigma):
    """Laplacian kernel: exp(-||x-y||_1 / sigma)."""
    return torch.exp(-_l1_distances(X, Y) / sigma)


_KERNELS = {
    'gaussian': _gaussian_gram,
    'laplacian': _laplacian_gram,
}


def _to_tensor(arr):
    """Convert numpy array to float torch tensor if needed."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    return arr.float()


def compute_mmd_u_statistic(X, Y, kernel_type='gaussian', sigma=1.0):
    """Compute the unbiased U-statistic estimator of MMD^2.

    Parameters
    ----------
    X, Y : ndarray or Tensor of shape (n, d)
        Two samples to compare.
    kernel_type : str
        'gaussian' or 'laplacian'.
    sigma : float
        Bandwidth parameter.

    Returns
    -------
    float
        Unbiased MMD^2 U-statistic estimate.
    """
    if kernel_type not in _KERNELS:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")

    X = _to_tensor(X)
    Y = _to_tensor(Y)
    kernel_fn = _KERNELS[kernel_type]

    m = X.shape[0]
    n = Y.shape[0]

    K_xx = kernel_fn(X, X, sigma)
    K_yy = kernel_fn(Y, Y, sigma)
    K_xy = kernel_fn(X, Y, sigma)

    # Exclude diagonal for unbiased estimator
    xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    xy = K_xy.sum() / (m * n)

    return (xx + yy - 2.0 * xy).item()


def optimize_bandwidth(X, Y, kernel_type='gaussian', search_space=(0.1, 10.0)):
    """Find the bandwidth that maximizes the MMD U-statistic.

    Parameters
    ----------
    X, Y : ndarray or Tensor of shape (n, d)
        Two samples to compare.
    kernel_type : str
        'gaussian' or 'laplacian'.
    search_space : tuple of (float, float)
        Bounds for the bandwidth search.

    Returns
    -------
    dict
        {'max_mmd': float, 'optimal_sigma': float}
    """
    X = _to_tensor(X)
    Y = _to_tensor(Y)

    def neg_mmd(sigma):
        return -compute_mmd_u_statistic(X, Y, kernel_type=kernel_type, sigma=sigma)

    result = minimize_scalar(neg_mmd, bounds=search_space, method='bounded')

    return {
        'max_mmd': -result.fun,
        'optimal_sigma': result.x,
    }


# ==================================================================
# Deep discriminator for learned-representation MMD
# ==================================================================
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepDiscriminator(nn.Module):
    """3-layer MLP: x_dim -> width -> width -> feature_dim."""

    def __init__(self, x_dim, feature_dim, hidden_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, feature_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


def mmd_loss_torch(f_x, f_y, sigma):
    """Differentiable MMD^2 U-statistic for discriminator training."""
    m, n = f_x.size(0), f_y.size(0)

    def gram(A, B):
        A_sq = A.pow(2).sum(1).view(-1, 1)
        B_sq = B.pow(2).sum(1).view(1, -1)
        dist = (A_sq + B_sq - 2 * A.mm(B.t())).clamp(min=0.0)
        return torch.exp(-dist / (sigma ** 2))

    K_xx = gram(f_x, f_x)
    K_yy = gram(f_y, f_y)
    K_xy = gram(f_x, f_y)

    xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    xy = K_xy.sum() / (m * n)
    return xx + yy - 2 * xy


def train_discriminator(X, Y, hidden_width, sigma, x_dim, feature_dim,
                        lr=0.005, weight_decay=0.0, n_steps=150,
                        grad_clip=5.0, n_restarts=3, learn_sigma=True):
    """Train D to maximise MMD(D(X), D(Y); sigma).

    Uses multiple random restarts to escape DC local optima and
    optionally learns sigma jointly (sigma variation is a linear
    scaling absorbable into D's layers).

    Parameters
    ----------
    X, Y : ndarray or Tensor
    hidden_width : int
    sigma : float   Initial kernel bandwidth (learned if learn_sigma).
    x_dim, feature_dim : int   Network architecture.
    n_restarts : int
        Number of independent random initialisations.
    learn_sigma : bool
        If True, jointly optimise log(sigma) alongside D.

    Returns
    -------
    dict: {'max_mmd': float, 'model': DeepDiscriminator,
           'optimal_sigma': float}
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)

    best_mmd = -float('inf')
    best_D = None
    best_sigma = float(sigma)

    for _ in range(n_restarts):
        D = DeepDiscriminator(x_dim, feature_dim, hidden_width).to(_DEVICE)

        params = list(D.parameters())
        if learn_sigma:
            log_sigma = torch.nn.Parameter(
                torch.tensor(np.log(float(sigma)), device=_DEVICE))
            params.append(log_sigma)

        optimizer = torch.optim.Adam(params, lr=lr,
                                     weight_decay=weight_decay)

        for _ in range(n_steps):
            optimizer.zero_grad()
            cur_sigma = log_sigma.exp() if learn_sigma else sigma
            loss = -mmd_loss_torch(D(X_t), D(Y_t), cur_sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

        with torch.no_grad():
            cur_sigma_val = log_sigma.exp().item() if learn_sigma else float(sigma)
            mmd_val = abs(mmd_loss_torch(D(X_t), D(Y_t), cur_sigma_val).item())

        if mmd_val > best_mmd:
            best_mmd = mmd_val
            best_D = D
            best_sigma = cur_sigma_val

    best_D.eval()
    return {'max_mmd': best_mmd, 'model': best_D,
            'optimal_sigma': best_sigma}


def train_discriminator_dca(X, Y, hidden_width, sigma, x_dim, feature_dim,
                            lr=0.01, n_outer=10, n_inner=30,
                            n_restarts=3, learn_sigma=True):
    """Maximise MMD via Stochastic DCA with multiple restarts.

    DCA decomposes MMD = g(D) - h(D) where:
      g(D) = within-class kernel similarity  (K_xx + K_yy terms)
      h(D) = cross-class kernel similarity   (K_xy term)

    Each outer iteration:
      1. Linearise h at current D_t  (freeze cross-term gradient)
      2. Maximise  g(D) - <nabla h(D_t), D>  for n_inner Adam steps

    Parameters
    ----------
    n_outer : int   DCA outer iterations.
    n_inner : int   Adam steps per DCA subproblem.
    n_restarts : int   Independent random initialisations.
    learn_sigma : bool  Jointly optimise log(sigma).

    Returns
    -------
    dict: {'max_mmd': float, 'model': DeepDiscriminator,
           'optimal_sigma': float}
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    m, n = X_t.size(0), Y_t.size(0)

    best_mmd = -float('inf')
    best_D = None
    best_sigma = float(sigma)

    for _ in range(n_restarts):
        D = DeepDiscriminator(x_dim, feature_dim, hidden_width).to(_DEVICE)

        params = list(D.parameters())
        if learn_sigma:
            log_sigma = torch.nn.Parameter(
                torch.tensor(np.log(float(sigma)), device=_DEVICE))
            params.append(log_sigma)

        for outer in range(n_outer):
            cur_sigma = log_sigma.exp() if learn_sigma else sigma

            # --- Step 1: linearise h (cross-term) at current D ---
            D.zero_grad()
            f_x = D(X_t)
            f_y = D(Y_t)
            K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) /
                             (cur_sigma ** 2))
            h_val = (2.0 / (m * n)) * K_xy.sum()
            h_val.backward()
            # Store nabla_h as detached copies
            h_grads = [p.grad.detach().clone() for p in params]
            D.zero_grad()

            # --- Step 2: maximise g(D) - <nabla_h, D> ---
            optimizer = torch.optim.Adam(params, lr=lr)
            for inner in range(n_inner):
                optimizer.zero_grad()
                cur_sigma = log_sigma.exp() if learn_sigma else sigma
                f_x = D(X_t)
                f_y = D(Y_t)

                # g(D): within-class terms
                K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) /
                                 (cur_sigma ** 2))
                K_yy = torch.exp(-_sq_euclidean_distances(f_y, f_y) /
                                 (cur_sigma ** 2))
                g_val = ((K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1)) +
                         (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1)))

                # Linear correction: <nabla_h(D_t), D>
                linear_corr = sum((p * hg).sum()
                                  for p, hg in zip(params, h_grads))

                loss = -(g_val - linear_corr)
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

        # Evaluate final MMD
        with torch.no_grad():
            cur_sigma_val = log_sigma.exp().item() if learn_sigma else float(sigma)
            mmd_val = abs(mmd_loss_torch(D(X_t), D(Y_t), cur_sigma_val).item())

        if mmd_val > best_mmd:
            best_mmd = mmd_val
            best_D = D
            best_sigma = cur_sigma_val

    best_D.eval()
    return {'max_mmd': best_mmd, 'model': best_D,
            'optimal_sigma': best_sigma}
