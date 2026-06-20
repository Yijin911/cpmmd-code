"""
Run the high d/n deep kernel settings where CP-MMD >> Plain >> Liu.
Uses the same code as run_cross_comparison.py.
Settings: d=50 n=100, d=100 n=100, d=20 n=50.
200 reps.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from run_cross_comparison
from run_cross_comparison import (
    run_experiment, save_results, log, RESULTS_DIR
)
from datetime import datetime

def main():
    log(f"{'='*70}")
    log(f"HIGH d/n DEEP KERNEL EXPERIMENTS")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*70}")

    all_results = []
    deltas = [0, 0.5, 1.0]

    res = run_experiment('C_deep_d50_n100', dim=50, dist_type='gaussian',
                         deltas=deltas, n_samples=100, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_high_dn_1.csv')

    res = run_experiment('C_deep_d100_n100', dim=100, dist_type='gaussian',
                         deltas=deltas, n_samples=100, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_high_dn_2.csv')

    res = run_experiment('C_deep_d20_n50', dim=20, dist_type='gaussian',
                         deltas=deltas, n_samples=50, mode='deep')
    all_results.extend(res)
    save_results(all_results, 'results_high_dn_final.csv')

    log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
