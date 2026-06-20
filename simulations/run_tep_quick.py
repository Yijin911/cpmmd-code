"""
Quick TEP experiment: covariance-shift fault detection.
Reduced: 30 reps, 100 permutations, deep methods only (skip MMDAgg).
"""
import os, sys, time, csv
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmd_core import (DeepDiscriminator, mmd_loss_torch, _to_tensor,
                       _sq_euclidean_distances, _DEVICE, compute_mmd_u_statistic)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'tep')
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG = os.path.join(RESULTS_DIR, 'tep_quick_log.txt')
log_fh = open(LOG, 'w', buffering=1)
def log(msg):
    print(msg); log_fh.write(msg + '\n'); log_fh.flush()

ALPHA = 0.05
N_REPS = 50
N_PERM = 200
N_SAMPLES = 200
DIM = 52
FEATURE_DIM = 10
HIDDEN_WIDTH = 200
N_STEPS = 100
LR = 0.005

def generate_tep_data(n, fault_type=None, seed=None):
    """Generate synthetic TEP-like data with d=52 sensors."""
    if seed is not None:
        np.random.seed(seed)
    # Normal covariance: moderate correlations
    rng = np.random.RandomState(42)
    A = rng.randn(DIM, DIM) * 0.1
    Sigma_normal = np.eye(DIM) + A @ A.T
    # Ensure well-conditioned
    Sigma_normal = 0.5 * (Sigma_normal + Sigma_normal.T)
    eigvals = np.linalg.eigvalsh(Sigma_normal)
    if eigvals.min() < 0.1:
        Sigma_normal += (0.1 - eigvals.min()) * np.eye(DIM)

    X = np.random.multivariate_normal(np.zeros(DIM), Sigma_normal, size=n)

    if fault_type is None:
        Y = np.random.multivariate_normal(np.zeros(DIM), Sigma_normal, size=n)
    else:
        Sigma_fault = Sigma_normal.copy()
        mu_fault = np.zeros(DIM)
        if fault_type == 3:
            for i in range(5):
                for j in range(5, 10):
                    Sigma_fault[i, j] = 0.6
                    Sigma_fault[j, i] = 0.6
        elif fault_type == 9:
            mu_fault[:10] = 0.1
            for i in range(5):
                for j in range(5, 10):
                    Sigma_fault[i, j] = 0.4
                    Sigma_fault[j, i] = 0.4
        elif fault_type == 15:
            for i in range(20, 31):
                Sigma_fault[i, i] *= 1.5
                for j in range(DIM):
                    if j != i:
                        Sigma_fault[i, j] *= np.sqrt(1.5)
                        Sigma_fault[j, i] *= np.sqrt(1.5)
        # Ensure PD
        Sigma_fault = 0.5 * (Sigma_fault + Sigma_fault.T)
        eigvals = np.linalg.eigvalsh(Sigma_fault)
        if eigvals.min() < 0.01:
            Sigma_fault += (0.01 - eigvals.min()) * np.eye(DIM)
        Y = np.random.multivariate_normal(mu_fault, Sigma_fault, size=n)
    return X, Y

def compute_G_spectral(model, X, Y, sigma):
    spec_prod = 1.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name:
                spec_prod *= torch.linalg.matrix_norm(p, ord=2).item()
    Z = np.concatenate([X, Y], axis=0)
    N = len(Z)
    return spec_prod * np.linalg.norm(Z) / (np.sqrt(N) * sigma)

def median_heuristic(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    Z_t = _to_tensor(Z)
    dists = _sq_euclidean_distances(Z_t, Z_t)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    return max(torch.sqrt(dists[mask]).median().item(), 1e-6)

def train_deep(X, Y, criterion='plain', C1=0.01):
    """Train deep kernel, return model + sigma + mmd."""
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(DIM, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma]
    optimizer = torch.optim.Adam(params, lr=LR)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_loss_torch(f_x, f_y, sigma)

        if criterion == 'cpmmd':
            spec_prod = torch.tensor(1.0, device=_DEVICE)
            for name, p in D.named_parameters():
                if 'weight' in name:
                    spec_prod = spec_prod * torch.linalg.matrix_norm(p, ord=2)
            Z = np.concatenate([X, Y], axis=0)
            data_geom = np.linalg.norm(Z) / np.sqrt(len(Z))
            G_h = spec_prod * data_geom / sigma
            J = mmd2 - C1 * G_h
            (-J).backward()
        elif criterion == 'liu':
            m = f_x.shape[0]
            K_xx = torch.exp(-_sq_euclidean_distances(f_x, f_x) / (sigma**2))
            K_xy = torch.exp(-_sq_euclidean_distances(f_x, f_y) / (sigma**2))
            h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
            var_h = torch.var(h_x).clamp(min=1e-10)
            sigma_hat = torch.sqrt(4 * var_h / m)
            J = mmd2 / sigma_hat
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
    return D, sigma_val, mmd_val

def perm_split_test(X, Y, train_fn):
    """Train on first half, permutation test on second half."""
    m_tr = len(X) // 2
    n_tr = len(Y) // 2
    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
    X_te, Y_te = X[m_tr:], Y[n_tr:]

    model, sigma, _ = train_fn(X_tr, Y_tr)

    X_te_t = _to_tensor(X_te).to(_DEVICE)
    Y_te_t = _to_tensor(Y_te).to(_DEVICE)
    with torch.no_grad():
        f_all = model(torch.cat([X_te_t, Y_te_t]))
    m_te, n_te = len(X_te), len(Y_te)
    test_stat = mmd_loss_torch(f_all[:m_te], f_all[m_te:], sigma).item()

    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(m_te + n_te)
        null_mmds.append(mmd_loss_torch(f_all[perm[:m_te]], f_all[perm[m_te:]], sigma).item())

    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return int(test_stat > threshold)

def median_perm_test(X, Y):
    """Median heuristic + full permutation test."""
    sigma = median_heuristic(X, Y)
    m, n = len(X), len(Y)
    test_stat = compute_mmd_u_statistic(X, Y, 'gaussian', sigma=sigma)
    pooled = np.concatenate([X, Y], axis=0)
    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(m + n)
        null_mmds.append(compute_mmd_u_statistic(pooled[perm[:m]], pooled[perm[m:]], 'gaussian', sigma=sigma))
    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return int(test_stat > threshold)

def calibrate_C1(n_reps=10):
    """Calibrate C1 from null permutations."""
    X, Y = generate_tep_data(N_SAMPLES, fault_type=None, seed=999)
    pooled = np.concatenate([X, Y], axis=0)
    m, n = len(X), len(Y)
    N = m + n
    ratios = []
    for i in range(n_reps):
        perm = np.random.permutation(N)
        Xp, Yp = pooled[perm[:m]], pooled[perm[m:]]
        model, sigma, mmd_val = train_deep(Xp, Yp, criterion='plain')
        G = compute_G_spectral(model, Xp, Yp, sigma)
        if G > 1e-12:
            ratios.append(mmd_val / G)
    return float(np.percentile(ratios, 95)) if ratios else 0.01

def main():
    log(f"TEP Quick Experiment: N_REPS={N_REPS}, N_PERM={N_PERM}, N_SAMPLES={N_SAMPLES}")
    log(f"Device: {_DEVICE}")

    log("\nCalibrating C1...")
    C1 = calibrate_C1(n_reps=10)
    log(f"C1 = {C1:.6f}")

    results = []
    faults = [None, 3, 9, 15]

    for fault in faults:
        fault_label = f"Fault_{fault}" if fault else "Normal"
        metric = "TYPE_I" if fault is None else "POWER"
        log(f"\n{'='*50}")
        log(f"Condition: {fault_label} ({metric})")
        log(f"{'='*50}")

        methods = [
            ('CPMMD1+PermSplit2', lambda X, Y: perm_split_test(X, Y, lambda x, y: train_deep(x, y, 'cpmmd', C1))),
            ('Liu1+PermSplit2', lambda X, Y: perm_split_test(X, Y, lambda x, y: train_deep(x, y, 'liu'))),
            ('Plain1+PermSplit2', lambda X, Y: perm_split_test(X, Y, lambda x, y: train_deep(x, y, 'plain'))),
            ('Median+Perm', lambda X, Y: median_perm_test(X, Y)),
        ]

        for mname, mfn in methods:
            rejects = 0
            t0 = time.time()
            for rep in range(N_REPS):
                np.random.seed(rep * 1000 + (fault or 0))
                torch.manual_seed(rep * 1000 + (fault or 0))
                X, Y = generate_tep_data(N_SAMPLES, fault_type=fault)
                try:
                    rejects += mfn(X, Y)
                except Exception as e:
                    log(f"      Error in {mname} rep {rep}: {e}")
            dt = time.time() - t0
            rate = rejects / N_REPS
            log(f"  {mname:25s}: {metric}={rate:.3f} ({N_REPS} reps, {dt:.1f}s)")
            results.append({'fault': fault_label, 'method': mname, 'metric': metric,
                            'value': rate, 'n_reps': N_REPS})

    csv_path = os.path.join(RESULTS_DIR, 'tep_quick_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['fault', 'method', 'metric', 'value', 'n_reps'])
        w.writeheader()
        w.writerows(results)
    log(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    main()
