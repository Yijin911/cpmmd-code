"""
Polynomial pillar — CP-MMD methods only, no MMDAgg.

The slow part of run_polynomial_pillar.py is MMDAgg (47 min per rho cell).
This script runs the same SNR sweep but with only the Lap+poly methods,
which take ~17 min per rho cell. Used to get the CP-MMD numbers fast,
in parallel with the full run.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_hdgm_revised as base
from datetime import datetime


def run_hdgm_cp_only(dim, rho_values):
    """Copy of base.run_hdgm_experiment but with MMDAgg removed."""
    all_results = []

    for rho in rho_values:
        is_null = (rho == 0.0)
        label = "TYPE_I" if is_null else "POWER"
        base.log(f"\n  rho={rho} ({label})")

        # Only the Lap+poly methods, no MMDAgg.
        method_list = [
            ('Lap+poly1', base.run_lap_poly1),
            ('Lap+poly2', base.run_lap_poly2),
            ('Lap+poly2 (adaptive)', base.run_lap_poly2_adaptive),
        ]

        import time
        for method_name, method_fn in method_list:
            t0 = time.time()
            rejections = 0
            valid = 0
            for rep in range(base.N_REPS):
                import numpy as np
                np.random.seed(rep * 5555 + int(rho * 1000) + dim)
                import torch
                torch.manual_seed(rep * 5555 + int(rho * 1000) + dim)

                X, Y = base.generate_hdgm_data(base.N_SAMPLES, dim, rho=rho)
                try:
                    rejected = method_fn(X, Y, dim=dim)
                    rejections += int(rejected)
                    valid += 1
                except Exception as e:
                    pass
            dt = time.time() - t0
            rate = rejections / valid if valid > 0 else 0
            base.log(f"    {method_name:25s}: {label}={rate:.3f} ({valid}/{base.N_REPS} valid, {dt:.1f}s)")
            all_results.append({
                'dim': dim, 'rho': rho,
                'hypothesis': 'H0' if is_null else 'H1',
                'method': method_name,
                'rejection_rate': rate,
                'n_valid': valid,
                'n_reps': base.N_REPS,
                'n_samples': base.N_SAMPLES,
                'time_s': dt,
            })

    return all_results


def main():
    base.log(f"{'='*70}")
    base.log(f"POLYNOMIAL PILLAR (CP-only): HDGM SNR sweep at d=20")
    base.log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    base.log(f"Device: {base._DEVICE}")
    base.log(f"N_REPS={base.N_REPS}, N_PERM={base.N_PERM}, "
             f"N_SAMPLES={base.N_SAMPLES}")
    base.log(f"{'='*70}")

    rho_values = [0.0, 0.10, 0.15, 0.20, 0.30, 0.40]

    all_results = []
    for dim in [20]:
        results = run_hdgm_cp_only(dim, rho_values)
        all_results.extend(results)
        base.save_results(all_results, f'hdgm_snr_cponly_d{dim}.csv')

    base.save_results(all_results, 'hdgm_snr_cponly_results.csv')

    base.log(f"\n{'='*70}\nSummary\n{'='*70}")
    base.log(f"{'Method':30s} {'rho':>5s} {'Rate':>8s}")
    for r in all_results:
        base.log(f"{r['method']:30s} {r['rho']:5.2f} {r['rejection_rate']:8.3f}")

    base.log(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
