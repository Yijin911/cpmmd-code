"""
Liu mixture ablation: criterion x kernel-ansatz factorial.

For section 5.4 of the paper. Compares six method combinations on the same
deep MLP class, isolating contributions of (a) the criterion and (b) the
kernel ansatz.

Methods (criterion x kernel form):
                          | pure-feature  k(phi(x), phi(y))  | Liu mixture k_eps  |
  CP-MMD criterion        |  A (Table 4)                     |  B (heuristic G)   |
  Liu criterion (J/sigma) |  D (Table 4 "Liu" row)           |  C (Liu published) |
  Plain criterion (J)     |  E (Table 4 "Plain" row)         |  F (NEW)           |

Liu's mixture kernel:
    k_eps(x, y) = [(1-eps) * kappa(phi(x), phi(y)) + eps] * q(x, y)
where kappa is Gaussian on features (bandwidth sigma_k), q is Gaussian on
raw inputs (bandwidth sigma_q), eps in (0,1) learnable.

By linearity of MMD,
    MMD^2(P, Q; k_eps) = (1-eps) * MMD^2(kappa*q) + eps * MMD^2(q).

PermSplit handled correctly: every method retrains on train half before
testing on held-out half.

Output: results/liu_mixture_ablation/results_*.csv
Env vars: PILOT=1 (3 reps, single cell), N_REPS=N (default 100), CELLS=...
"""
import os, sys, time, csv
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmd_core import (
    DeepDiscriminator, mmd_loss_torch,
    _to_tensor, _sq_euclidean_distances, _DEVICE, BestIterate,
)
from calibration_engine import compute_S
from data_generator import generate_data
from run_cross_comparison import (
    step1_cpmmd, step1_liu, step1_plain, step2_perm_split,
    calibrate_C1_C2, ALPHA, FEATURE_DIM, HIDDEN_WIDTH, N_STEPS, LR, N_PERM,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'liu_mixture_ablation'
)
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt')
log_fh = open(LOG_FILE, 'a', buffering=1)


def log(msg):
    print(msg, flush=True)
    log_fh.write(msg + '\n')
    log_fh.flush()


# ----------------------------------------------------------------------
# Mixture-kernel MMD computation
# ----------------------------------------------------------------------
def _gaussian_gram_torch(A, B, sigma):
    A_sq = A.pow(2).sum(1).view(-1, 1)
    B_sq = B.pow(2).sum(1).view(1, -1)
    dist = (A_sq + B_sq - 2 * A.mm(B.t())).clamp(min=0.0)
    return torch.exp(-dist / (sigma ** 2))


def mmd_mixture_torch(X_t, Y_t, f_x, f_y, eps, sigma_k, sigma_q):
    m, n = X_t.size(0), Y_t.size(0)

    K_xx_feat = _gaussian_gram_torch(f_x, f_x, sigma_k)
    K_yy_feat = _gaussian_gram_torch(f_y, f_y, sigma_k)
    K_xy_feat = _gaussian_gram_torch(f_x, f_y, sigma_k)

    Q_xx = _gaussian_gram_torch(X_t, X_t, sigma_q)
    Q_yy = _gaussian_gram_torch(Y_t, Y_t, sigma_q)
    Q_xy = _gaussian_gram_torch(X_t, Y_t, sigma_q)

    P_xx = K_xx_feat * Q_xx
    P_yy = K_yy_feat * Q_yy
    P_xy = K_xy_feat * Q_xy

    def ustat(Kxx, Kyy, Kxy):
        xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
        yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
        xy = Kxy.sum() / (m * n)
        return xx + yy - 2 * xy

    mmd_kq = ustat(P_xx, P_yy, P_xy)
    mmd_q = ustat(Q_xx, Q_yy, Q_xy)
    return (1 - eps) * mmd_kq + eps * mmd_q


def mmd_mixture_eval(X, Y, model, eps, sigma_k, sigma_q):
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    with torch.no_grad():
        f_x = model(X_t)
        f_y = model(Y_t)
        return mmd_mixture_torch(X_t, Y_t, f_x, f_y, eps, sigma_k, sigma_q).item()


def _spec_product(model):
    spec = torch.tensor(1.0, device=_DEVICE)
    for name, p in model.named_parameters():
        if 'weight' in name:
            spec = spec * torch.linalg.matrix_norm(p, ord=2)
    return spec


# ----------------------------------------------------------------------
# Step 1: training on Liu's mixture under different criteria
# ----------------------------------------------------------------------
def _train_mixture(X, Y, dim, criterion_fn, select_best=False):
    """Generic training loop on Liu's mixture kernel.

    criterion_fn(mmd2, model, eps, sigma_k, sigma_q, X_t, Y_t, f_x, f_y) -> J (to maximise)
    """
    X_t = _to_tensor(X).to(_DEVICE)
    Y_t = _to_tensor(Y).to(_DEVICE)
    D = DeepDiscriminator(dim, FEATURE_DIM, HIDDEN_WIDTH).to(_DEVICE)
    log_sigma_k = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    log_sigma_q = nn.Parameter(torch.tensor(0.0, device=_DEVICE))
    logit_eps = nn.Parameter(torch.tensor(-1.0, device=_DEVICE))
    params = list(D.parameters()) + [log_sigma_k, log_sigma_q, logit_eps]
    opt = torch.optim.Adam(params, lr=LR)

    # select_best=True selects argmax_{0<=t<=T} J (used for CP-MMD only).
    best = BestIterate() if select_best else None
    for step in range(N_STEPS + 1):
        opt.zero_grad()
        sigma_k = log_sigma_k.exp()
        sigma_q = log_sigma_q.exp()
        eps = torch.sigmoid(logit_eps)
        f_x, f_y = D(X_t), D(Y_t)
        mmd2 = mmd_mixture_torch(X_t, Y_t, f_x, f_y, eps, sigma_k, sigma_q)
        J = criterion_fn(mmd2, D, eps, sigma_k, sigma_q, X_t, Y_t, f_x, f_y)
        if best is not None:
            best.update(J.item(), D, log_sigma_k, log_sigma_q, logit_eps)
        if step == N_STEPS:
            break
        (-J).backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    if best is not None:
        best.restore(D, log_sigma_k, log_sigma_q, logit_eps)

    D.eval()
    with torch.no_grad():
        sigma_k_v = log_sigma_k.exp().item()
        sigma_q_v = log_sigma_q.exp().item()
        eps_v = torch.sigmoid(logit_eps).item()
        mmd_v = mmd_mixture_eval(X, Y, D, eps_v, sigma_k_v, sigma_q_v)
    return {
        'sigma_k': sigma_k_v, 'sigma_q': sigma_q_v, 'eps': eps_v,
        'mmd': mmd_v, 'model': D, 'mode': 'mixture',
    }


def step1_cpmmd_mixture(X, Y, dim, C1):
    """B: CP-MMD criterion on Liu's mixture (heuristic G)."""
    Z = np.concatenate([X, Y], axis=0)
    data_geom = float(np.linalg.norm(Z) / np.sqrt(len(Z)))

    def crit(mmd2, D, eps, sigma_k, sigma_q, X_t, Y_t, f_x, f_y):
        spec = _spec_product(D)
        G_feat = spec * data_geom / sigma_k
        G_raw = data_geom / sigma_q
        G_h = (1 - eps) * G_feat + eps * G_raw
        return mmd2 - C1 * G_h
    return _train_mixture(X, Y, dim, crit, select_best=True)


def step1_liu_mixture(X, Y, dim):
    """C: Liu's published criterion J = MMD^2 / sigma_hat on Liu's mixture."""
    m = len(X)

    def crit(mmd2, D, eps, sigma_k, sigma_q, X_t, Y_t, f_x, f_y):
        K_xx_feat = _gaussian_gram_torch(f_x, f_x, sigma_k)
        K_xy_feat = _gaussian_gram_torch(f_x, f_y, sigma_k)
        Q_xx = _gaussian_gram_torch(X_t, X_t, sigma_q)
        Q_xy = _gaussian_gram_torch(X_t, Y_t, sigma_q)
        K_xx = (1 - eps) * K_xx_feat * Q_xx + eps * Q_xx
        K_xy = (1 - eps) * K_xy_feat * Q_xy + eps * Q_xy
        h_x = K_xx.mean(dim=1) - K_xy.mean(dim=1)
        var_h = torch.var(h_x).clamp(min=1e-10)
        sigma_hat = torch.sqrt(4 * var_h / m)
        return mmd2 / sigma_hat
    return _train_mixture(X, Y, dim, crit)


def step1_plain_mixture(X, Y, dim):
    """F: Plain criterion (no penalty) on Liu's mixture."""
    def crit(mmd2, D, eps, sigma_k, sigma_q, X_t, Y_t, f_x, f_y):
        return mmd2
    return _train_mixture(X, Y, dim, crit)


# ----------------------------------------------------------------------
# PermSplit testing on the mixture kernel
# ----------------------------------------------------------------------
def step2_perm_split_mixture(X, Y, s1_split):
    """PermSplit: evaluate test stat and permutation null on second half
    using model already trained on train half (stored in s1_split)."""
    m, n = len(X), len(Y)
    m_tr, n_tr = m // 2, n // 2
    X_te, Y_te = X[m_tr:], Y[n_tr:]
    m_te, n_te = len(X_te), len(Y_te)

    model = s1_split['model']
    eps = s1_split['eps']
    sigma_k = s1_split['sigma_k']
    sigma_q = s1_split['sigma_q']

    test_stat = mmd_mixture_eval(X_te, Y_te, model, eps, sigma_k, sigma_q)

    pooled_te = np.concatenate([X_te, Y_te], axis=0)
    null_mmds = []
    for _ in range(N_PERM):
        perm = np.random.permutation(m_te + n_te)
        X_p = pooled_te[perm[:m_te]]
        Y_p = pooled_te[perm[m_te:]]
        null_mmds.append(mmd_mixture_eval(X_p, Y_p, model, eps, sigma_k, sigma_q))

    threshold = float(np.percentile(null_mmds, (1 - ALPHA) * 100))
    return {'reject': test_stat > threshold, 'test_stat': test_stat, 'threshold': threshold}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_one_setting(dim, n_samples, deltas, n_reps, exp_name):
    log(f"\n{'='*70}\nEXPERIMENT: {exp_name} (dim={dim}, n={n_samples}, reps={n_reps})\n{'='*70}")
    log("  Calibrating C1, C2 on pure-feature form (reused for mixture's heuristic G)...")
    C1, C2 = calibrate_C1_C2(dim, n_samples, dist_type='gaussian', mode='deep', n_reps=10)
    log(f"  C1={C1:.6f}, C2={C2:.4f}")

    # Each method: (s1 callable, mixture flag)
    methods = {
        'A_CPMMD_pure':  (lambda X, Y: step1_cpmmd(X, Y, dim, C1, mode='deep'), False),
        'B_CPMMD_mix':   (lambda X, Y: step1_cpmmd_mixture(X, Y, dim, C1),       True),
        'C_Liu_mix':     (lambda X, Y: step1_liu_mixture(X, Y, dim),             True),
        'D_Liu_pure':    (lambda X, Y: step1_liu(X, Y, dim, mode='deep'),        False),
        'E_Plain_pure':  (lambda X, Y: step1_plain(X, Y, dim, mode='deep'),      False),
        'F_Plain_mix':   (lambda X, Y: step1_plain_mixture(X, Y, dim),           True),
    }

    rows = []
    for delta in deltas:
        for name, (s1_fn, is_mix) in methods.items():
            rejects, valid = 0, 0
            t0 = time.time()
            for rep in range(n_reps):
                seed = rep * 1000 + int(delta * 100) + dim
                np.random.seed(seed)
                torch.manual_seed(seed)
                X, Y = generate_data(n_samples, dist_type='gaussian',
                                     shift_type='mean' if delta > 0 else 'none',
                                     delta=delta, dim=dim)
                try:
                    # PermSplit: train on first half, test on second half.
                    m_tr = len(X) // 2
                    n_tr = len(Y) // 2
                    X_tr, Y_tr = X[:m_tr], Y[:n_tr]
                    s1_split = s1_fn(X_tr, Y_tr)
                    if is_mix:
                        s2 = step2_perm_split_mixture(X, Y, s1_split)
                    else:
                        s2 = step2_perm_split(X, Y, s1_split, dim)
                    rejects += int(s2['reject'])
                    valid += 1
                except Exception as e:
                    log(f"    [warn] {name} rep {rep}: {type(e).__name__}: {e}")

            dt = time.time() - t0
            rate = rejects / max(valid, 1)
            label = 'TYPE_I' if delta == 0 else 'POWER'
            log(f"  delta={delta} {name:15s}: {label}={rate:.3f} ({valid}/{n_reps} valid, {dt:.1f}s)")
            rows.append({
                'experiment': exp_name, 'dim': dim, 'n_samples': n_samples,
                'delta': delta, 'method': name,
                'rejection_rate': rate, 'n_valid': valid, 'n_reps': n_reps,
                'time_s': dt,
            })
    return rows


def save_csv(rows, fname):
    fp = os.path.join(RESULTS_DIR, fname)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(fp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  saved {fp}")


def main():
    start = datetime.now()
    log(f"\n{'#'*70}\nLiu mixture ablation\nstart {start.strftime('%Y-%m-%d %H:%M:%S')}\ndevice {_DEVICE}\n{'#'*70}")

    if os.environ.get('PILOT', '0') == '1':
        settings = [{'exp_name': 'pilot', 'dim': 20, 'n_samples': 200,
                     'deltas': [0.0, 0.5], 'n_reps': int(os.environ.get('N_REPS', 5))}]
    else:
        n_reps = int(os.environ.get('N_REPS', 100))
        deltas_env = os.environ.get('DELTAS')
        if deltas_env:
            deltas = [float(x) for x in deltas_env.split(',')]
        else:
            deltas = [0.0, 0.5, 1.0]
        all_settings = {
            'd2_n200':    {'exp_name': 'd2_n200',   'dim': 2,   'n_samples': 200, 'deltas': deltas, 'n_reps': n_reps},
            'd20_n200':   {'exp_name': 'd20_n200',  'dim': 20,  'n_samples': 200, 'deltas': deltas, 'n_reps': n_reps},
            'd50_n100':   {'exp_name': 'd50_n100',  'dim': 50,  'n_samples': 100, 'deltas': deltas, 'n_reps': n_reps},
            'd100_n100':  {'exp_name': 'd100_n100', 'dim': 100, 'n_samples': 100, 'deltas': deltas, 'n_reps': n_reps},
            'd20_n50':    {'exp_name': 'd20_n50',   'dim': 20,  'n_samples': 50,  'deltas': deltas, 'n_reps': n_reps},
        }
        cells_env = os.environ.get('CELLS')
        if cells_env:
            keys = [k.strip() for k in cells_env.split(',')]
            settings = [all_settings[k] for k in keys if k in all_settings]
        else:
            settings = list(all_settings.values())

    all_rows = []
    for s in settings:
        rows = run_one_setting(**s)
        all_rows.extend(rows)
        save_csv(all_rows, 'results_partial.csv')

    save_csv(all_rows, 'results_final.csv')
    log(f"\nfinished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
