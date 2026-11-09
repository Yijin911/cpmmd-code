"""Selection cost of CP-MMD in the Fig. 3(b) setting (d = 28, n = 200 per group).

Selection follows the grid procedure of the Higgs linear and polynomial ansatze
(run_higgs_extra_gpu.py): C1_hat is the 0.95 quantile, over N_CAL = 30 null
permutations of the fitting sample, of max_sigma gamma_hat^2(sigma) / G_tilde(sigma),
and sigma_hat maximizes J_CP = gamma_hat^2 - C1_hat * G_tilde over a 24-point
geometric grid on the search interval of run_deployment_cost.py, with
G_tilde(sigma) = ||Psi(D)||_F / (sigma N) as in Eq. (10). The pooled
squared-distance matrix is computed once and shared by all bandwidths and
permutations; at each bandwidth, the statistics of the observed split and of the
N_CAL permutations come from one matrix product.

CP-MMD-ard (per-feature bandwidths) calibrates and selects with
run_higgs_extra_gpu.select_ard: Adam ascent on the log feature scales, N_CAL null
permutations for the plain-MMD maximizers and one J_CP ascent, ARD_STEPS steps each,
in PyTorch. Its deployed test is the linear one applied to the rescaled features.

Selection runs on one CPU thread. The per-test cost reuses the JAX test
of run_deployment_cost.py, with JIT compilation excluded as in Fig. 3(b).
K* = T_sel / (T_baseline - T_test) is the number of tests after which the
cumulative cost of CP-MMD falls below that of a baseline; the baseline and
CP-MMD per-test times are read from results/deployment_cost/deployment_cost.csv.

Usage: python run_selection_cost.py
Output: results/deployment_cost/selection_cost.csv
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU timing, as in Fig. 3(b)

import csv  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import torch  # noqa: E402

import run_deployment_cost as rdc  # noqa: E402
import run_higgs_extra_gpu as rx  # noqa: E402

torch.set_num_threads(1)
ARD_GRID = list(np.geomspace(2.0, 20.0, 24))     # start grid of CPMMD-ard (sigma_lin)

N_CAL = 30
N_GRID = 24
N_REPEATS = 5
GRIDS = {
    'linear': np.geomspace(rdc.SIGMA_RANGE[0], rdc.SIGMA_RANGE[1], N_GRID),
    'poly3': np.geomspace(rdc.SIGMA_RANGE[0], rdc.SIGMA_RANGE[1] * 10, N_GRID),
}


def mmd_u_batched(K, A, m, n):
    """Unbiased MMD^2 for every column of the 0/1 labelling matrix A (N x P)."""
    KA = K @ A
    KB = K.sum(axis=1)[:, None] - KA
    B = 1.0 - A
    diag = np.diag(K)
    s_xx = (A * KA).sum(axis=0) - A.T @ diag
    s_yy = (B * KB).sum(axis=0) - B.T @ diag
    s_xy = (B * KA).sum(axis=0)
    return s_xx / (m * (m - 1)) + s_yy / (n * (n - 1)) - 2.0 * s_xy / (m * n)


def select(kind, X, Y, seed):
    """Calibrate C1_hat and return (C1_hat, sigma_hat, boundary flag)."""
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)
    if kind == 'poly3':
        Z = rdc._poly_features(Z, 3)
    N, m = len(Z), len(X)
    sq_norm = (Z * Z).sum(axis=1)
    sq = np.maximum(sq_norm[:, None] + sq_norm[None, :] - 2.0 * (Z @ Z.T), 0.0)
    A = np.zeros((N, N_CAL + 1), dtype=Z.dtype)
    A[:m, 0] = 1.0                                   # observed split
    for p in range(N_CAL):
        A[rng.permutation(N)[:m], p + 1] = 1.0       # null permutations
    sigmas = GRIDS[kind]
    G = np.linalg.norm(Z) / (sigmas * N)
    stats = np.stack([mmd_u_batched(np.exp(-sq / s ** 2), A, m, N - m) for s in sigmas])
    C1 = float(np.percentile((stats[:, 1:] / G[:, None]).max(axis=0), 95))
    j = int(np.argmax(stats[:, 0] - C1 * G))
    return C1, float(sigmas[j]), j in (0, N_GRID - 1)


def select_ard(X, Y, seed):
    """Calibrate C1_hat and select the feature scales of CPMMD-ard; returns (C1_hat, scales)."""
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    C1, log_a, _, _ = rx.select_ard(t(X), t(Y), ARD_GRID, np.random.default_rng(seed))
    return C1, log_a.exp().numpy()


def per_test_times(model, test_pairs):
    model.per_test(*test_pairs[0], seed=-1)          # JIT warm-up
    times = []
    for i, (Xt, Yt) in enumerate(test_pairs):
        t0 = time.perf_counter()
        model.per_test(Xt, Yt, seed=i)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    X, Y = rdc.gen(rdc.N, rdc.DIM, rdc.DELTA, seed=0)
    test_pairs = [rdc.gen(rdc.N, rdc.DIM, rdc.DELTA, seed=1000 + i) for i in range(rdc.N_TRIALS)]
    with open(os.path.join(rdc.RESULTS_DIR, 'deployment_cost.csv')) as f:
        cost = list(csv.DictReader(f))
    t_cost = {(r['method'], int(r['B'])): float(r['per_test_median_s']) for r in cost}

    rows = []
    for kind in ('linear', 'poly3', 'ard'):
        run = (lambda r: select_ard(X, Y, seed=r)) if kind == 'ard' else (lambda r: select(kind, X, Y, seed=r))
        run(99)                                      # warm-up
        sel, out = [], None
        for r in range(N_REPEATS):
            t0 = time.perf_counter()
            out = run(r)
            sel.append(time.perf_counter() - t0)
        t_sel = float(np.median(sel))
        if kind == 'ard':                            # linear test on the rescaled features
            C1, scales = out
            sigma, boundary = float(np.exp(-np.log(scales).mean())), False
            model = rdc.CPMMDDeployedKernel(np.sqrt(2.0))
            pairs = [(Xt * scales, Yt * scales) for Xt, Yt in test_pairs]
            t_test = t_cost[('CP-MMD-linear', 1)]
        else:
            C1, sigma, boundary = out
            model = (rdc.CPMMDDeployedKernel(sigma) if kind == 'linear'
                     else rdc.CPMMDPolyDeployed(sigma, 3))
            pairs = test_pairs
            t_test = t_cost[(f'CP-MMD-{kind}', 1)]
        t_test_here = per_test_times(model, pairs)
        row = {'method': f'CP-MMD-{kind}', 'n_cal': N_CAL,
               'grid_points': N_GRID if kind != 'ard' else f'{len(ARD_GRID)} + {rx.ARD_STEPS} Adam steps',
               'selection_ms': round(1000 * t_sel, 1),
               'per_test_ms': round(1000 * t_test, 1),
               'per_test_ms_this_run': round(1000 * t_test_here, 1),
               'total_one_test_ms': round(1000 * (t_sel + t_test), 1),
               'C1_hat': f'{C1:.3g}', 'sigma_hat': round(sigma, 4),
               'sigma_on_boundary': boundary}
        for base in ('MMDAgg', 'MMDAgg (default)', 'MMD-FUSE'):
            for B in rdc.B_GRID:
                gap = t_cost[(base, B)] - t_test
                row[f'K*[{base}, B={B}]'] = round(t_sel / gap, 2) if gap > 0 else 'inf'
        rows.append(row)
        print(row, flush=True)

    out_path = os.path.join(rdc.RESULTS_DIR, 'selection_cost.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
