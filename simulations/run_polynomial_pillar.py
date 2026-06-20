"""
Polynomial pillar: CP-MMD+poly2 vs MMDAgg, SNR sweep on HDGM at d=20.

The existing run_hdgm_revised.py covered rho in {0.0, 0.5}, where 0.5 is
a strong cross-correlation that even MMDAgg+raw can detect via bandwidth
aggregation. At d=20 the previous run had:
    Lap+poly2 (adaptive)  : POWER=0.34
    MMDAgg                : POWER=0.58
i.e., MMDAgg won because the signal was so strong it didn't need
polynomial features.

This script reruns at lower SNR (rho in {0.1, 0.15, 0.2, 0.3}) where the
cross-correlation is too weak for raw-feature aggregation but still
detectable via degree-2 polynomial features. CP-MMD+poly2 should be at
least competitive with MMDAgg in this regime.

Compared to run_hdgm_revised.py: only the rho_values list and the dim
list differ.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_hdgm_revised as base
from datetime import datetime

# Override the dimensions and rho values for the SNR sweep.
# We focus on d=20 since that's where the previous run lost.
def main():
    base.log(f"{'='*70}")
    base.log(f"POLYNOMIAL PILLAR: HDGM SNR sweep at d=20")
    base.log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    base.log(f"Device: {base._DEVICE}")
    base.log(f"N_REPS={base.N_REPS}, N_PERM={base.N_PERM}, "
             f"N_SAMPLES={base.N_SAMPLES}")
    base.log(f"ALPHA={base.ALPHA}")
    base.log(f"{'='*70}")

    all_results = []

    # H0 + multiple H1 SNR levels
    rho_values = [0.0, 0.10, 0.15, 0.20, 0.30, 0.40]

    for dim in [20]:
        d_prime = base.get_poly_dim(dim, degree=2)
        adaptive_q = 1.0 - base.ALPHA * dim / max(dim, d_prime)
        base.log(f"\n{'='*70}")
        base.log(f"Dimension d={dim}, poly2 dim d'={d_prime}")
        base.log(f"Standard quantile: 0.95, Adaptive quantile: {adaptive_q:.4f}")
        base.log(f"{'='*70}")

        results = base.run_hdgm_experiment(dim, rho_values)
        all_results.extend(results)

        base.save_results(all_results, f'hdgm_snr_d{dim}.csv')

    base.save_results(all_results, 'hdgm_snr_results.csv')

    # Summary table
    base.log(f"\n{'='*70}")
    base.log(f"Summary")
    base.log(f"{'='*70}")
    base.log(f"{'Method':30s} {'dim':>4s} {'rho':>5s} {'Rate':>8s}")
    base.log(f"{'-'*50}")
    for r in all_results:
        base.log(f"{r['method']:30s} {r['dim']:4d} {r['rho']:5.2f} "
                 f"{r['rejection_rate']:8.3f}")

    base.log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
