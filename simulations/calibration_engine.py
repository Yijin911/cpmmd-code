"""Data-driven calibration engine for MMD-based two-sample testing.

Learns bound coefficients C1 and C2 **separately**:

    tau = C1 * G  +  C2 * S

C1 captures how the *mean* of the optimised MMD scales with the Gaussian
complexity G (a property of the optimisation / function-class).

C2 captures how the *tail fluctuation* of the MMD statistic (at a fixed
model) relates to the concentration term S = sqrt(ln(2/delta) / (m+n)),
which is a function of the desired significance level and sample sizes.

G and S are public, reusable functions that work with arbitrary
optimisation problems (bandwidth, learned representations, etc.).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mmd_core import optimize_bandwidth, compute_mmd_u_statistic


# ==================================================================
# G term  –  Gaussian complexity
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
    r"""
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
# S term  –  concentration / tail
# ==================================================================
def compute_S(delta, m, n):
    """Concentration term:  sqrt( ln(2/delta) / (m + n) ).

    Parameters
    ----------
    delta : float   Confidence parameter (e.g. 0.05).
    m, n  : int     Sample sizes of the two groups.
    """
    return np.sqrt(np.log(2.0 / delta) / (m + n))


# ==================================================================
# Stage 1 — fit C1 (mean behaviour)
# ==================================================================
def fit_C1(X, Y, kernel_type=None, search_space=None,
           n_reps=5, n_mc=200, objective_fn=None,
           optimize_fn=None, d_out=None, compute_g_fn=None,
           use_shared_g_ref=True):
    """Learn C1 from permutation-null shuffles at the full sample size.

    The theoretical relationship (Prop 4.23 in the journal) is
    C_1* = r*(H_T) / G_T^{ref} + small, where C_1* is identified with
    the population (1-alpha)-quantile of (max_theta MMD)/G under the
    null permutation distribution. For Prop 4.23's proof to go through,
    the denominator G must be permutation-INVARIANT — otherwise the
    permutation distribution of the ratio is not exchangeable.

    With ``use_shared_g_ref=True`` (default, journal procedure):
        Compute G ONCE on the original (un-permuted) data, and reuse the
        same G_ref for every permutation. Each permutation only varies
        the numerator (max_theta MMD on permuted data). The resulting
        empirical (1-alpha)-quantile is the calibrated C_1.

    With ``use_shared_g_ref=False`` (legacy / conference procedure):
        Compute G separately on each permuted sample. This recovers the
        original behaviour and is preserved for the empirical sanity
        check that verifies the two procedures give similar C_1 values
        on the experiments of S5.9.

    Parameters
    ----------
    kernel_type, search_space : used for bandwidth case (default).
    n_reps : int
        Number of permutation shuffles (default 5).
    n_mc : int
        MC draws for each G estimate.
    optimize_fn : callable(X, Y) -> dict with 'max_mmd', or None.
        If None, uses optimize_bandwidth with kernel_type/search_space.
    d_out : int or None
        Passed to compute_G for xi dimension.
    compute_g_fn : callable(X, Y, opt_result) -> float, or None.
        If provided, used instead of compute_G with objective_fn.
        Receives the optimisation result (with 'model', 'optimal_sigma')
        so G can be computed from the trained model's parameters.
    use_shared_g_ref : bool, default True
        If True (journal procedure), compute G_ref once on the
        un-permuted data and reuse it across all permutations. If False
        (legacy procedure), recompute G on each permuted sample.

    Returns
    -------
    C1 : float
        Average of per-rep ratios  mmd / G.
    diagnostics : dict
        Per-rep mmds and Gs for inspection. Under the shared-G_ref
        procedure, ``Gs`` is a list of identical values (G_ref).
    """
    if objective_fn is None and compute_g_fn is None:
        objective_fn = make_bandwidth_objective(search_space)
    if optimize_fn is None:
        def optimize_fn(X_sub, Y_sub):
            return optimize_bandwidth(X_sub, Y_sub, kernel_type, search_space)

    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)
    rng = np.random.default_rng()

    mmds = []
    Gs = []

    if use_shared_g_ref:
        opt_ref = optimize_fn(X, Y)
        if compute_g_fn is not None:
            G_ref = compute_g_fn(X, Y, opt_ref)
        else:
            G_ref = compute_G(X, Y, objective_fn, n_mc, d_out=d_out)

    for _ in range(n_reps):
        perm = rng.permutation(N)
        X_new = pooled[perm[:m]]
        Y_new = pooled[perm[m:m + n]]

        opt_result = optimize_fn(X_new, Y_new)
        mmds.append(opt_result['max_mmd'])
        if use_shared_g_ref:
            Gs.append(G_ref)
        elif compute_g_fn is not None:
            Gs.append(compute_g_fn(X_new, Y_new, opt_result))
        else:
            Gs.append(compute_G(X_new, Y_new, objective_fn,
                                 n_mc, d_out=d_out))

    G_arr = np.array(Gs)
    mmd_arr = np.array(mmds)
    ratios = mmd_arr / np.where(np.abs(G_arr) > 1e-12, G_arr, np.nan)
    C1 = float(np.nanmean(ratios))

    return C1, {'mmds': mmds, 'Gs': Gs,
                'use_shared_g_ref': use_shared_g_ref}


# ==================================================================
# Stage 2 — fit C2 (tail behaviour)
# ==================================================================
def fit_C2(X, Y, kernel_type=None, optimal_sigma=None,
           n_permutations=200,
           alphas=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
           null_mmd_fn=None):
    """Learn C2 by matching empirical null quantiles to S(alpha).

    Build the permutation-null distribution of MMD at a fixed model
    (no re-optimisation), then match the (1-alpha) quantiles to S(alpha).

    Parameters
    ----------
    kernel_type, optimal_sigma : used for bandwidth case (default).
    n_permutations : int
    alphas : sequence of float
    null_mmd_fn : callable(X_perm, Y_perm) -> float, or None.
        If None, uses compute_mmd_u_statistic at fixed sigma.
        For deep case, pass a function that applies fixed D then
        computes MMD on the features.

    Returns
    -------
    C2 : float
    r2 : float
    diagnostics : dict
    """
    if null_mmd_fn is None:
        def null_mmd_fn(Xp, Yp):
            return compute_mmd_u_statistic(Xp, Yp, kernel_type,
                                           sigma=optimal_sigma)

    m, n = len(X), len(Y)
    N = m + n
    pooled = np.concatenate([X, Y], axis=0)
    rng = np.random.default_rng()

    null_mmds = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(N)
        X_perm = pooled[perm[:m]]
        Y_perm = pooled[perm[m:m + n]]
        null_mmds[i] = null_mmd_fn(X_perm, Y_perm)

    quantiles = []
    S_vals = []
    for a in alphas:
        q = np.percentile(null_mmds, (1.0 - a) * 100)
        quantiles.append(q)
        S_vals.append(compute_S(a, m, n))

    S_arr = np.array(S_vals).reshape(-1, 1)
    q_arr = np.array(quantiles)
    reg = LinearRegression(fit_intercept=False).fit(S_arr, q_arr)
    C2 = reg.coef_[0]
    r2 = r2_score(q_arr, reg.predict(S_arr))

    return C2, r2, {'alphas': list(alphas),
                     'quantiles': quantiles,
                     'S_vals': S_vals,
                     'null_mmds': null_mmds}


# ==================================================================
# Combined threshold
# ==================================================================
def get_calibrated_threshold(X, Y, kernel_type=None, search_space=None,
                             alpha=0.05,
                             n_reps=5,
                             n_permutations=200,
                             n_mc=200,
                             objective_fn=None,
                             optimize_fn=None,
                             make_null_fn=None,
                             d_out=None,
                             compute_g_fn=None,
                             alphas_for_C2=(0.01, 0.02, 0.05, 0.10,
                                            0.15, 0.20),
                             use_shared_g_ref=True):
    """Two-stage calibration: fit C1 (mean) then C2 (tail), return threshold.

    Generic interface that works for both bandwidth optimisation and
    deep discriminator.  For the bandwidth case (default), pass
    kernel_type and search_space.  For the deep case, pass
    optimize_fn, make_null_fn, and compute_g_fn.

    Parameters
    ----------
    optimize_fn : callable(X, Y) -> dict with 'max_mmd' (and
        optionally 'model', 'optimal_sigma', etc.), or None.
    make_null_fn : callable(opt_result) -> callable(X_perm, Y_perm) -> float
        Factory that takes the optimisation result and returns a
        function to compute MMD on permuted data at the fixed model.
    d_out : int or None
        xi dimension for compute_G (bandwidth case).
    compute_g_fn : callable(X, Y, opt_result) -> float, or None.
        If provided, computes G from the trained model's parameters
        (e.g. spectral norm bound).  Used in both fit_C1 and for
        the final threshold.

    Returns
    -------
    dict with keys: tau, C1, C2, R2_C2, test_stat, G, opt_result,
        C1_diagnostics, and optimal_sigma (bandwidth case only).
    """
    if objective_fn is None and search_space is not None and compute_g_fn is None:
        objective_fn = make_bandwidth_objective(search_space)
    if optimize_fn is None:
        def optimize_fn(X_sub, Y_sub):
            return optimize_bandwidth(X_sub, Y_sub, kernel_type, search_space)

    m, n = len(X), len(Y)

    # Optimise on original data
    opt = optimize_fn(X, Y)
    test_stat = opt['max_mmd']

    # Stage 1: fit C1
    C1, c1_diag = fit_C1(X, Y, optimize_fn=optimize_fn,
                          objective_fn=objective_fn,
                          n_reps=n_reps,
                          n_mc=n_mc, d_out=d_out,
                          compute_g_fn=compute_g_fn,
                          use_shared_g_ref=use_shared_g_ref)

    # Stage 2: fit C2
    if make_null_fn is not None:
        null_mmd_fn = make_null_fn(opt)
    else:
        null_mmd_fn = None

    C2, r2_C2, c2_diag = fit_C2(
        X, Y, kernel_type=kernel_type,
        optimal_sigma=opt.get('optimal_sigma'),
        null_mmd_fn=null_mmd_fn,
        n_permutations=n_permutations,
        alphas=alphas_for_C2)

    # Variance correction: scale C2 by the ratio
    #   std(gap from n_reps optimisations) / std(null MMD at fixed D)
    # so that C2 accounts for optimisation-induced variance.
    std_opt = np.std(c1_diag['mmds']) if len(c1_diag['mmds']) > 1 else 0.0
    std_null = np.std(c2_diag['null_mmds']) if len(c2_diag['null_mmds']) > 1 else 1.0
    var_ratio = std_opt / std_null if std_null > 1e-12 else 1.0
    C2_corrected = C2 * var_ratio

    # Threshold: G from trained model or from objective_fn
    if compute_g_fn is not None:
        G_orig = compute_g_fn(X, Y, opt)
    else:
        G_orig = compute_G(X, Y, objective_fn, n_mc, d_out=d_out)
    S_orig = compute_S(alpha, m, n)
    tau = C1 * G_orig + C2_corrected * S_orig

    result = {
        'tau': tau,
        'C1': C1,
        'C2': C2_corrected,
        'C2_raw': C2,
        'var_ratio': var_ratio,
        'R2_C2': r2_C2,
        'test_stat': test_stat,
        'G': G_orig,
        'C1_diagnostics': c1_diag,
        'opt_result': opt,
    }
    if 'optimal_sigma' in opt:
        result['optimal_sigma'] = opt['optimal_sigma']
    return result
