"""
Quick HDGM experiment: validates adaptive calibration fix.
Reduced scale: n=500 (not 2000), 100 permutations (not 200), 50 reps.
Skips MMDAgg (we have original paper numbers). Focus on poly1 vs poly2 vs poly2_adaptive.
"""
import os, sys, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmd_core import _to_tensor, _l1_distances, _DEVICE

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hdgm')
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG = os.path.join(RESULTS_DIR, 'hdgm_quick_log.txt')
log_fh = open(LOG, 'w', buffering=1)
def log(msg):
    print(msg); log_fh.write(msg + '\n'); log_fh.flush()

ALPHA = 0.05
N_REPS = 50
N_PERM = 100
N_SAMPLES = 500

def generate_hdgm(n, d, rho=0.0):
    """P = N(0,I), Q has Cov(Y1,Y2)=rho."""
    X = np.random.randn(n, d)
    cov = np.eye(d)
    if rho != 0:
        cov[0, 1] = cov[1, 0] = rho
    L = np.linalg.cholesky(cov)
    Y = np.random.randn(n, d) @ L.T
    return X, Y

def poly2_features(X):
    """Degree-2 polynomial: (x, x_i*x_j for i<=j)."""
    n, d = X.shape
    quads = []
    for i in range(d):
        for j in range(i, d):
            quads.append(X[:, i] * X[:, j])
    return np.column_stack([X] + quads)

def laplacian_mmd(X, Y, sigma):
    """Laplacian MMD U-statistic."""
    X_t, Y_t = _to_tensor(X), _to_tensor(Y)
    m, n = X_t.shape[0], Y_t.shape[0]
    import torch
    K_xx = torch.exp(-_l1_distances(X_t, X_t) / sigma)
    K_yy = torch.exp(-_l1_distances(Y_t, Y_t) / sigma)
    K_xy = torch.exp(-_l1_distances(X_t, Y_t) / sigma)
    xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    xy = K_xy.sum() / (m * n)
    return (xx + yy - 2 * xy).item()

def run_poly_method(X_raw, Y_raw, degree, quantile_level):
    """Run Laplacian+poly method with PermSplit."""
    m_tr, n_tr = len(X_raw) // 2, len(Y_raw) // 2
    X_tr, Y_tr = X_raw[:m_tr], Y_raw[:n_tr]
    X_te, Y_te = X_raw[m_tr:], Y_raw[n_tr:]

    # Apply features
    if degree == 2:
        X_tr_f = poly2_features(X_tr)
        Y_tr_f = poly2_features(Y_tr)
        X_te_f = poly2_features(X_te)
        Y_te_f = poly2_features(Y_te)
    else:
        X_tr_f, Y_tr_f = X_tr, Y_tr
        X_te_f, Y_te_f = X_te, Y_te

    # Calibrate C1 on training half
    pooled_tr = np.concatenate([X_tr_f, Y_tr_f], axis=0)
    N_tr = len(pooled_tr)
    m_h, n_h = len(X_tr_f), len(Y_tr_f)
    sigmas = np.logspace(-1, 2, 20)

    ratios = []
    for _ in range(10):
        perm = np.random.permutation(N_tr)
        Xp, Yp = pooled_tr[perm[:m_h]], pooled_tr[perm[m_h:]]
        best_mmd = -1e10
        best_sigma = 1.0
        for s in sigmas:
            v = laplacian_mmd(Xp, Yp, s)
            if v > best_mmd:
                best_mmd = v
                best_sigma = s
        Z = np.concatenate([Xp, Yp], axis=0)
        G = np.linalg.norm(Z) / (np.sqrt(N_tr) * best_sigma)
        if G > 1e-12:
            ratios.append(best_mmd / G)
    C1 = float(np.percentile(ratios, quantile_level * 100)) if ratios else 0.01

    # Select bandwidth on training half using J = MMD^2 - C1*G
    best_J = -1e10
    best_sigma = 1.0
    for s in sigmas:
        mmd_val = laplacian_mmd(X_tr_f, Y_tr_f, s)
        Z = np.concatenate([X_tr_f, Y_tr_f], axis=0)
        G = np.linalg.norm(Z) / (np.sqrt(N_tr) * s)
        J = mmd_val - C1 * G
        if J > best_J:
            best_J = J
            best_sigma = s

    # Test on held-out half
    test_stat = laplacian_mmd(X_te_f, Y_te_f, best_sigma)
    pooled_te = np.concatenate([X_te_f, Y_te_f], axis=0)
    m_te, n_te = len(X_te_f), len(Y_te_f)
    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(m_te + n_te)
        null_mmds.append(laplacian_mmd(pooled_te[perm[:m_te]], pooled_te[perm[m_te:]], best_sigma))
    threshold = np.percentile(null_mmds, (1 - ALPHA) * 100)
    return int(test_stat > threshold)

def main():
    log(f"HDGM Quick Experiment: N_REPS={N_REPS}, N_PERM={N_PERM}, N_SAMPLES={N_SAMPLES}")
    results = []

    for d in [10, 20]:
        d_prime = d + d * (d + 1) // 2
        q_standard = 1 - ALPHA
        q_adaptive = 1 - ALPHA * d / max(d, d_prime)
        log(f"\n{'='*60}")
        log(f"d={d}, d'={d_prime}, q_standard={q_standard:.4f}, q_adaptive={q_adaptive:.4f}")
        log(f"{'='*60}")

        for rho in [0.0, 0.5]:
            label = "TYPE_I" if rho == 0 else "POWER"
            log(f"\n  rho={rho} ({label})")

            methods = [
                ('Lap+poly1', 1, q_standard),
                ('Lap+poly2', 2, q_standard),
                ('Lap+poly2 (adaptive)', 2, q_adaptive),
            ]

            for name, degree, q_level in methods:
                rejects = 0
                t0 = time.time()
                for rep in range(N_REPS):
                    np.random.seed(rep * 100 + d)
                    X, Y = generate_hdgm(N_SAMPLES, d, rho=rho)
                    rejects += run_poly_method(X, Y, degree, q_level)
                dt = time.time() - t0
                rate = rejects / N_REPS
                log(f"    {name:25s}: {label}={rate:.3f} ({N_REPS} reps, {dt:.1f}s)")
                results.append({'d': d, 'd_prime': d_prime, 'rho': rho,
                                'method': name, 'metric': label,
                                'value': rate, 'n_reps': N_REPS})

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'hdgm_quick_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['d', 'd_prime', 'rho', 'method', 'metric', 'value', 'n_reps'])
        w.writeheader()
        w.writerows(results)
    log(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    main()
