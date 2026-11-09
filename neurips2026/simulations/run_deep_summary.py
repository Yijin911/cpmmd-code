"""
Deep-kernel summary experiment for Table 2 (tab:deep-summary).

Reproduces the (d, n) sweep that the deleted run_cross_comparison.py drove for
the body's deep-kernel head-to-head: CP-MMD / Liu / Plain on the same MLP
under the sample-split protocol. Imports the step1_* and step2_perm_split
helpers from the recovered run_cross_comparison.py so the calibration and
training pipelines match the script that produced the original numbers.

Cells: {(2,200), (20,200), (50,100), (100,100), (20,50)} at delta=0 (type-I)
and delta=0.5 (power). 200 reps each.

Output: results/deep_summary/deep_summary_results.csv
"""

import os, sys, time, csv, warnings
import numpy as np
import torch

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cross_comparison import (
    step1_cpmmd, step1_liu, step1_plain, step2_perm_split, calibrate_C1_C2,
)
from data_generator import generate_data
from mmd_core import _DEVICE

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'deep_summary')
os.makedirs(RESULTS_DIR, exist_ok=True)

CELLS = [(2, 200), (20, 200), (50, 100), (100, 100), (20, 50)]
DELTAS = [0.0, 0.5]
N_REPS = 200

CRITERIA = {
    'CPMMD': lambda X, Y, dim, C1: step1_cpmmd(X, Y, dim, C1, mode='deep'),
    'Liu':   lambda X, Y, dim, C1: step1_liu(X, Y, dim, mode='deep'),
    'Plain': lambda X, Y, dim, C1: step1_plain(X, Y, dim, mode='deep'),
}


def run_cell(dim, n_samples, delta, C1, criterion_name, criterion_fn, log_fh):
    rejects = 0
    valid = 0
    t0 = time.time()
    for rep in range(N_REPS):
        seed = rep * 1000 + int(delta * 100)
        np.random.seed(seed)
        torch.manual_seed(seed)
        X, Y = generate_data(n_samples, dist_type='gaussian',
                             shift_type='mean' if delta > 0 else 'none',
                             delta=delta, dim=dim)
        try:
            m_tr = len(X) // 2
            n_tr = len(Y) // 2
            X_tr, Y_tr = X[:m_tr], Y[:n_tr]
            s1 = criterion_fn(X_tr, Y_tr, dim, C1)
            s2 = step2_perm_split(X, Y, s1, dim)
            rejects += int(s2['reject'])
            valid += 1
        except Exception as e:
            log_fh.write(f"  rep {rep} failed: {e}\n")
    dt = time.time() - t0
    rate = rejects / max(valid, 1)
    return rate, valid, dt


def main():
    log_path = os.path.join(RESULTS_DIR, 'run_log.txt')
    log_fh = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg)
        log_fh.write(msg + '\n')

    log(f"Device: {_DEVICE}")
    log(f"N_REPS = {N_REPS}")
    log(f"Cells: {CELLS}")
    log(f"Deltas: {DELTAS}\n")

    csv_path = os.path.join(RESULTS_DIR, 'deep_summary_results.csv')
    fieldnames = ['dim', 'n_samples', 'delta', 'criterion',
                  'rejection_rate', 'n_valid', 'time_s', 'C1']
    rows = []

    for dim, n_samples in CELLS:
        log(f"\n{'='*70}")
        log(f"Cell (d={dim}, n={n_samples})")
        log(f"{'='*70}")
        log("Calibrating C1 ...")
        t0 = time.time()
        C1, _ = calibrate_C1_C2(dim, n_samples, dist_type='gaussian',
                                mode='deep', n_reps=10)
        log(f"  C1 = {C1:.6f}  ({time.time()-t0:.1f}s)")

        for delta in DELTAS:
            label = "TYPE_I" if delta == 0 else "POWER"
            log(f"\n  delta={delta} ({label})")
            for cname, cfn in CRITERIA.items():
                rate, valid, dt = run_cell(dim, n_samples, delta, C1,
                                           cname, cfn, log_fh)
                log(f"    {cname:6s}: {rate:.3f}  ({valid}/{N_REPS} valid, {dt:.0f}s)")
                rows.append({
                    'dim': dim, 'n_samples': n_samples, 'delta': delta,
                    'criterion': cname, 'rejection_rate': rate,
                    'n_valid': valid, 'time_s': dt, 'C1': C1,
                })
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)

    log(f"\nDone. Wrote {csv_path}  ({len(rows)} rows)")
    log_fh.close()


if __name__ == '__main__':
    main()
