"""
CIFAR-10 vs CIFAR-10.1 two-sample testing experiment.

P = CIFAR-10 test set (10,000 images)
Q = CIFAR-10.1 v6 (2,021 images)

This experiment compares kernel SELECTION criteria (Step 1) on real data.
All learned-kernel methods use the same Step 2 (PermSplit_2) to isolate
the effect of the selection criterion.

Methods compared:
  Deep kernel selection (Step 1) + PermSplit_2 (Step 2):
    CPMMD_1:  J_CP = MMD^2 - C1*G_h (complexity-penalised)
    Liu_1:    J_Liu  = MMD^2 / sigma_hat (variance-normalised)
    Plain_1:  max MMD^2 (no regularisation)

  Fixed-kernel baselines (permutation on full data):
    Median_1: median heuristic + permutation test
    MMDAgg:   kernel aggregation + permutation test
    MMD-FUSE: kernel fusion + permutation test

Features: PCA-reduced to 500 dimensions from raw pixels (3072-dim).
The defaults below are the setting of App. K: n = 200, 50 reps.
"""

import os, sys, time, csv, warnings
import numpy as np
import torch
import torch.nn as nn
import pickle

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    compute_mmd_u_statistic, optimize_bandwidth,
    DeepDiscriminator, mmd_loss_torch, _to_tensor,
    _sq_euclidean_distances, _DEVICE, BestIterate, liu_tau_hat
)
from calibration_engine import compute_S

from sklearn.decomposition import PCA
from mmdagg import mmdagg as mmdagg_fn
from mmdfuse import mmdfuse as mmdfuse_fn
import jax.random

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'cifar_pca500')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.expanduser('~/data')

ALPHA = 0.05
N_REPS = 50          # repetitions per setting
N_PERM = 200         # permutations for threshold
PCA_DIM = 500        # PCA dimensionality reduction
SAMPLE_SIZES = [200]

# Deep kernel hyperparameters (matching kernel_criterion.tex)
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = open(LOG_FILE, 'w', buffering=1)


def log(msg):
    print(msg)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ======================================================================
# Data loading
# ======================================================================
def load_cifar10_test():
    """Load CIFAR-10 test set from pickle file."""
    test_path = os.path.join(DATA_DIR, 'cifar-10-batches-py', 'test_batch')
    with open(test_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']  # (10000, 3072) uint8
    labels = np.array(batch[b'labels'])
    return data.astype(np.float32) / 255.0, labels


def load_cifar101():
    """Load CIFAR-10.1 v6 dataset."""
    data_path = os.path.join(DATA_DIR, 'cifar10.1', 'cifar10.1_v6_data.npy')
    labels_path = os.path.join(DATA_DIR, 'cifar10.1', 'cifar10.1_v6_labels.npy')
    data = np.load(data_path)   # (2021, 32, 32, 3) uint8
    labels = np.load(labels_path)
    data = data.reshape(len(data), -1).astype(np.float32) / 255.0
    return data, labels


def prepare_features(cifar10_data, cifar101_data, pca_dim=PCA_DIM):
    """Fit PCA on pooled data, return transformed features."""
    pooled = np.concatenate([cifar10_data, cifar101_data], axis=0)
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(pooled)
    log(f"  PCA: {pca.explained_variance_ratio_.sum():.3f} variance explained with {pca_dim} components")
    return pca.transform(cifar10_data), pca.transform(cifar101_data), pca


# ======================================================================
# Helpers
# ======================================================================
def median_heuristic(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z)
    dists = _sq_euclidean_distances(Z_t, Z_t)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    return max(torch.sqrt(dists[mask]).median().item(), 1e-6)


def compute_G_spectral(model, X, Y, sigma):
    """Spectral norm bound on Gaussian complexity for deep kernel."""
    spectral_product = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                spectral_product *= torch.linalg.matrix_norm(param, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return spectral_product * np.linalg.norm(Z) / (np.sqrt(N) * sigma)


def permutation_test(X, Y, sigma, n_perm=N_PERM):
    """Full permutation test at fixed bandwidth."""
    m, n = len(X), len(Y)
    test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
    pooled = np.concatenate([X, Y], axis=0)
    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m + n)
        null_mmds[i] = compute_mmd_u_statistic(
            pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma)
    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return test_stat > threshold


# ======================================================================
# Step 1: Deep kernel selection criteria
# ======================================================================
def step1_cpmmd(X, Y, dim, C1):
    """J_CP = MMD^2 - C1*G_h (complexity-penalised deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    Z = np.concatenate([X, Y], axis=0)
    data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
    best = BestIterate()  # h_hat = argmax_{0<=t<=T} J_CP(h^(t))
    for step in range(N_STEPS + 1):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        spec_prod = torch.tensor(1.0, device=_DEVICE)
        for name, p in D.named_parameters():
            if 'weight' in name:
                spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
        G_h = spec_prod * data_geom / sigma
        J = mmd2 - C1 * G_h
        best.update(J.item(), D, log_sigma)
        if step == N_STEPS:
            break
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    best.restore(D, log_sigma)
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


def step1_liu(X, Y, dim):
    """J_Liu = MMD^2 / sigma_hat (variance-normalised deep kernel)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        m = f_x.shape[0]
        K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
        K_yy = torch.exp(-_sq_euclidean_distances(f_y, f_y) / (sigma**2))
        K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
        sigma_hat = liu_tau_hat(K_xx, K_yy, K_xy) / np.sqrt(m)
        J = mmd2 / sigma_hat
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


def step1_plain(X, Y, dim):
    """Plain: max MMD^2 (no regularisation)."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)
        (-mmd2).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    D.eval()
    with torch.no_grad():
        sigma_val = log_sigma.exp().item()
        f_x, f_y = D(X_t), D(Y_t)
        mmd_val = mmd_loss_torch(f_x, f_y, sigma_val).item()
    return {'sigma': sigma_val, 'mmd': mmd_val, 'model': D}


# ======================================================================
# Step 2: PermSplit_2 for deep kernel methods
# ======================================================================
def step2_perm_split_deep(X, Y, step1_fn, dim, n_perm=N_PERM):
    """Train deep kernel on first half, permutation test on second half.

    step1_fn: callable(X_tr, Y_tr, dim) -> {'model', 'sigma', 'mmd'}
    """
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
    X_te, Y_te = X[m_tr:], Y[n_tr:]

    result = step1_fn(X_tr, Y_tr, dim)
    model, sigma = result['model'], result['sigma']

    # Compute features for test half
    te_all = np.concatenate([X_te, Y_te], axis=0)
    te_t = _to_tensor(te_all).to(_DEVICE)
    with torch.no_grad():
        f_all = model(te_t)

    m_te, n_te = len(X_te), len(Y_te)
    test_stat = mmd_loss_torch(f_all[:m_te], f_all[m_te:], sigma).item()

    null_mmds = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(m_te + n_te)
        null_mmds[i] = mmd_loss_torch(f_all[perm[:m_te]], f_all[perm[m_te:]], sigma).item()

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return test_stat > threshold


# ======================================================================
# Calibration for C1 (deep kernel, from null permutations)
# ======================================================================
def calibrate_C1(dim, X_null, Y_null, n_reps=10):
    """Calibrate C1 = 95th percentile of MMD/G over null permutations."""
    m, n = len(X_null), len(Y_null)
    N = m + n
    pooled = np.concatenate([X_null, Y_null], axis=0)

    ratios = []
    for rep in range(n_reps):
        perm = np.random.permutation(N)
        X_p, Y_p = pooled[perm[:m]], pooled[perm[m:]]
        res = step1_plain(X_p, Y_p, dim)
        G = compute_G_spectral(res['model'], X_p, Y_p, res['sigma'])
        if abs(G) > 1e-12:
            ratios.append(res['mmd'] / G)

    return float(np.percentile(ratios, 95)) if ratios else 0.01


# ======================================================================
# Main experiment
# ======================================================================
def run_cifar_experiment(cifar10_feat, cifar101_feat, n_samples, n_reps=N_REPS):
    """Run all methods on CIFAR-10 vs CIFAR-10.1 at given sample size."""
    dim = cifar10_feat.shape[1]
    n_cifar101 = len(cifar101_feat)
    n_per_group = min(n_samples, n_cifar101)

    log(f"\n{'='*70}")
    log(f"CIFAR experiment: n_samples={n_per_group}, n_reps={n_reps}")
    log(f"{'='*70}")

    # Calibrate C1 from CIFAR-10 null (both samples from CIFAR-10)
    log("  Calibrating C1 from CIFAR-10 null (deep kernel)...")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(cifar10_feat))
    X_cal = cifar10_feat[idx[:n_per_group]]
    Y_cal = cifar10_feat[idx[n_per_group:2*n_per_group]]
    C1 = calibrate_C1(dim, X_cal, Y_cal, n_reps=10)
    log(f"  C1={C1:.6f}")

    methods = [
        'CPMMD1+PermSplit2',
        'Liu1+PermSplit2',
        'Plain1+PermSplit2',
        'Median+Perm',
        'MMDAgg+Perm',
        'MMDFUSE+Perm',
    ]
    results = []

    for hypothesis, label in [('H1', 'CIFAR-10 vs CIFAR-10.1'),
                               ('H0', 'CIFAR-10 vs CIFAR-10')]:
        log(f"\n  --- {hypothesis}: {label} (n={n_per_group}) ---")
        counts = {k: 0 for k in methods}

        for rep in range(n_reps):
            seed = rep + (1000 if hypothesis == 'H1' else 5000)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if hypothesis == 'H1':
                idx_p = np.random.choice(len(cifar10_feat), n_per_group, replace=False)
                idx_q = np.random.choice(n_cifar101, n_per_group, replace=False)
                X = cifar10_feat[idx_p]
                Y = cifar101_feat[idx_q]
            else:
                idx_all = np.random.permutation(len(cifar10_feat))
                X = cifar10_feat[idx_all[:n_per_group]]
                Y = cifar10_feat[idx_all[n_per_group:2*n_per_group]]

            # Deep kernel methods (Step 1) + PermSplit_2 (Step 2)
            try:
                reject = step2_perm_split_deep(
                    X, Y, lambda Xt, Yt, d: step1_cpmmd(Xt, Yt, d, C1), dim)
                counts['CPMMD1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [CPMMD1] rep {rep} failed: {e}")

            try:
                reject = step2_perm_split_deep(X, Y, step1_liu, dim)
                counts['Liu1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [Liu1] rep {rep} failed: {e}")

            try:
                reject = step2_perm_split_deep(X, Y, step1_plain, dim)
                counts['Plain1+PermSplit2'] += int(reject)
            except Exception as e:
                log(f"    [Plain1] rep {rep} failed: {e}")

            # Median + full permutation test
            sigma_med = median_heuristic(X, Y)
            reject = permutation_test(X, Y, sigma_med)
            counts['Median+Perm'] += int(reject)

            # MMDAgg (permutation mode)
            try:
                result = mmdagg_fn(
                    X, Y, alpha=ALPHA,
                    kernel='laplace_gaussian',
                    number_bandwidths=10,
                    permutations_same_sample_size=True,
                    seed=seed,
                )
                counts['MMDAgg+Perm'] += int(result)
            except Exception as e:
                log(f"    [MMDAgg] rep {rep} failed: {e}")

            # MMD-FUSE (permutation-based)
            try:
                key = jax.random.PRNGKey(seed)
                result = mmdfuse_fn(
                    X, Y, key,
                    alpha=ALPHA,
                    number_bandwidths=10,
                    number_permutations=N_PERM,
                )
                counts['MMDFUSE+Perm'] += int(result)
            except Exception as e:
                log(f"    [MMDFUSE] rep {rep} failed: {e}")

            if (rep + 1) % 50 == 0:
                log(f"    rep {rep+1}/{n_reps}")

        metric = 'power' if hypothesis == 'H1' else 'type_I'
        for method in methods:
            val = counts[method] / n_reps
            log(f"    {method:25s}: {metric.upper()} = {val:.3f} ({counts[method]}/{n_reps})")
            results.append({
                'experiment': f'{hypothesis}_CIFAR10',
                'n_samples': n_per_group,
                'method': method,
                'metric': metric,
                'value': val,
                'n_reps': n_reps,
            })

    return results


# ======================================================================
# Entry point
# ======================================================================
if __name__ == '__main__':
    log(f"CIFAR-10 vs CIFAR-10.1 Experiment")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {_DEVICE}")

    # Load data
    log("\nLoading CIFAR-10 test set...")
    cifar10_data, cifar10_labels = load_cifar10_test()
    log(f"  Shape: {cifar10_data.shape}")

    log("Loading CIFAR-10.1 v6...")
    cifar101_data, cifar101_labels = load_cifar101()
    log(f"  Shape: {cifar101_data.shape}")

    # PCA reduction
    log(f"\nApplying PCA to {PCA_DIM} dimensions...")
    cifar10_feat, cifar101_feat, pca = prepare_features(cifar10_data, cifar101_data, PCA_DIM)
    log(f"  CIFAR-10 features: {cifar10_feat.shape}")
    log(f"  CIFAR-10.1 features: {cifar101_feat.shape}")

    all_results = []
    for n in SAMPLE_SIZES:
        results = run_cifar_experiment(cifar10_feat, cifar101_feat, n)
        all_results.extend(results)

    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, 'cifar_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'n_samples', 'method',
                                                'metric', 'value', 'n_reps'])
        writer.writeheader()
        writer.writerows(all_results)

    log(f"\nResults saved to {csv_path}")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()
