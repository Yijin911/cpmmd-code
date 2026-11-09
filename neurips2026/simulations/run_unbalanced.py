"""Deep CP-MMD with unequal sample sizes (m != n), Table 2 pipeline.

d = 20, mean shift Delta = 0.5, total N = m + n = 300, designs
(m, n) in {(150, 150), (200, 100), (100, 200)}. For each design, C1_hat is
calibrated once on an independent null sample of sizes (m, n), as in
run_deep_summary.py; each repetition trains on the first halves (m/2, n/2)
and runs the permutation test on the held-out halves. Reports the rejection
rate at Delta = 0 (Type-I) and Delta = 0.5 (power).

Usage:
  python run_unbalanced.py                      # all three designs
  python run_unbalanced.py --m 200 --n 100      # one design
Output: results/unbalanced/unbalanced_results.csv (or --out)
"""
import argparse
import csv
import os

import numpy as np
import torch

import run_cross_comparison as rcc

D, DELTA, N_CAL = 20, 0.5, 10
DESIGNS = [(150, 150), (200, 100), (100, 200)]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'unbalanced')


def calibrate(m, n, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((m, D)).astype(np.float32)
    Y0 = rng.standard_normal((n, D)).astype(np.float32)
    pooled = np.concatenate([X0, Y0], axis=0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ratios = []
    for _ in range(N_CAL):
        perm = rng.permutation(m + n)
        Xp, Yp = pooled[perm[:m]], pooled[perm[m:]]
        res = rcc.step1_plain(Xp, Yp, D, mode='deep')
        G = rcc.compute_G_spectral(res['model'], Xp, Yp, res['sigma'])
        if abs(G) > 1e-12:
            ratios.append(res['mmd'] / G)
    return float(np.percentile(ratios, 95))


def run_design(m, n, reps):
    C1 = calibrate(m, n)
    print(f'(m, n) = ({m}, {n}): C1_hat = {C1:.6f}', flush=True)
    rows = []
    for delta in (0.0, DELTA):
        rejects = 0
        for rep in range(reps):
            seed = rep * 1000 + int(delta * 100) + 7
            np.random.seed(seed)
            torch.manual_seed(seed)
            X = np.random.standard_normal((m, D)).astype(np.float32)
            Y = (np.random.standard_normal((n, D)) + delta).astype(np.float32)
            s1 = rcc.step1_cpmmd(X[:m // 2], Y[:n // 2], D, C1, mode='deep')
            rejects += int(rcc.step2_perm_split(X, Y, s1, D)['reject'])
        rate = rejects / reps
        print(f'  delta={delta}: rejection rate {rate:.3f}', flush=True)
        rows.append({'m': m, 'n': n, 'rho_star': round(max((m + n) / m, (m + n) / n), 3),
                     'C1_hat': round(C1, 6), 'delta': delta, 'rate': rate, 'reps': reps})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int)
    ap.add_argument('--n', type=int)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'unbalanced_results.csv'))
    args = ap.parse_args()
    designs = [(args.m, args.n)] if args.m and args.n else DESIGNS
    rows = [r for m, n in designs for r in run_design(m, n, args.reps)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
