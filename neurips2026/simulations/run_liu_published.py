"""Rerun the Liu baseline with Liu et al.'s published variance estimator.

Every deep-kernel driver computes Liu's ratio criterion
J_Liu = sqrt(n) * MMD^2 / tau_hat with mmd_core.liu_tau_hat, the H1 variance
estimate of Liu et al. (2020) and of their code. This script reruns only the
Liu rows of the paper's tables and figures. Each job reuses its driver's data
generation, seeds, training loop and held-out permutation test:

  table2 --cell 20x50          Table 2 (run_deep_summary), delta in {0, 0.5}
  table7 --cell d20_n50 --method C_Liu_mix|D_Liu_pure
                               Table 7 (run_liu_mixture_ablation), delta in {0, 0.5, 1}
  higgs  --n 200               Fig. 3(a) (run_higgs), H1 power and H0 Type-I
  tep                          App. K (run_tep), no fault and faults 3, 9, 15
  cifar                        App. K (run_cifar), PCA to 500 dimensions, n = 200
  merge                        writes the Liu rows into the drivers' result CSVs

Each job writes results/liu_published/<job>.csv.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, '..', 'results')
OUT_DIR = os.path.join(RESULTS, 'liu_published')
os.makedirs(OUT_DIR, exist_ok=True)

TABLE7_CELLS = {'d2_n200': (2, 200), 'd20_n200': (20, 200), 'd50_n100': (50, 100),
                'd100_n100': (100, 100), 'd20_n50': (20, 50)}


def write_rows(name, rows):
    path = os.path.join(OUT_DIR, f'{name}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}', flush=True)


def table2(cell):
    import run_deep_summary as rds
    dim, n = (int(v) for v in cell.split('x'))
    rows = []
    with open(os.path.join(OUT_DIR, f'table2_{cell}.log'), 'w') as log_fh:
        for delta in rds.DELTAS:
            rate, valid, dt = rds.run_cell(dim, n, delta, None, 'Liu', rds.CRITERIA['Liu'], log_fh)
            print(f'table2 d={dim} n={n} delta={delta}: Liu {rate:.3f} ({valid} valid, {dt:.0f}s)', flush=True)
            rows.append({'dim': dim, 'n_samples': n, 'delta': delta, 'criterion': 'Liu',
                         'rejection_rate': rate, 'n_valid': valid, 'time_s': dt})
    write_rows(f'table2_{cell}', rows)


def table7(cell, method, n_reps):
    import run_liu_mixture_ablation as rla
    from data_generator import generate_data
    dim, n_samples = TABLE7_CELLS[cell]
    step1, is_mix = {
        'C_Liu_mix': (lambda X, Y: rla.step1_liu_mixture(X, Y, dim), True),
        'D_Liu_pure': (lambda X, Y: rla.step1_liu(X, Y, dim, mode='deep'), False),
    }[method]
    rows = []
    for delta in (0.0, 0.5, 1.0):
        rejects = valid = 0
        t0 = time.time()
        for rep in range(n_reps):
            seed = rep * 1000 + int(delta * 100) + dim   # as rla.run_one_setting
            np.random.seed(seed)
            torch.manual_seed(seed)
            X, Y = generate_data(n_samples, dist_type='gaussian',
                                 shift_type='mean' if delta > 0 else 'none',
                                 delta=delta, dim=dim)
            try:
                s1 = step1(X[:len(X) // 2], Y[:len(Y) // 2])
                s2 = (rla.step2_perm_split_mixture(X, Y, s1) if is_mix
                      else rla.step2_perm_split(X, Y, s1, dim))
                rejects += int(s2['reject'])
                valid += 1
            except Exception as e:
                print(f'  [warn] {method} rep {rep}: {type(e).__name__}: {e}', flush=True)
        dt = time.time() - t0
        rate = rejects / max(valid, 1)
        print(f'table7 {cell} {method} delta={delta}: {rate:.3f} ({valid} valid, {dt:.0f}s)', flush=True)
        rows.append({'experiment': cell, 'dim': dim, 'n_samples': n_samples, 'delta': delta,
                     'method': method, 'rejection_rate': rate, 'n_valid': valid,
                     'n_reps': n_reps, 'time_s': dt})
    write_rows(f'table7_{cell}_{method}', rows)


def _count_rejections(label, draw, test, n_reps):
    rejects = 0
    for rep in range(n_reps):
        X, Y = draw(rep)
        try:
            rejects += int(test(X, Y))
        except Exception as e:
            print(f'  [warn] {label} rep {rep}: {type(e).__name__}: {e}', flush=True)
    return rejects / n_reps


def higgs(n, n_reps):
    import run_higgs as rh
    rh.log_fh = open(os.path.join(OUT_DIR, f'higgs_n{n}.log'), 'w', buffering=1)
    P, Q = rh.load_higgs(max_rows=rh.MAX_ROWS)   # P background, Q signal

    def draw(hypothesis):
        def _draw(rep):   # as rh.run_higgs_experiment
            seed = rep + (1000 if hypothesis == 'H1' else 5000)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hypothesis == 'H1':
                idx_p = np.random.choice(len(P), n, replace=False)
                idx_q = np.random.choice(len(Q), n, replace=False)
                return P[idx_p], Q[idx_q]
            idx_all = np.random.permutation(len(P))
            return P[idx_all[:n]], P[idx_all[n:2 * n]]
        return _draw

    test = lambda X, Y: rh.step2_perm_split_deep(X, Y, rh.step1_liu, rh.INPUT_DIM)
    rows = []
    for hypothesis, metric in (('H1', 'power'), ('H0', 'type_I')):
        value = _count_rejections(f'higgs n={n} {hypothesis}', draw(hypothesis), test, n_reps)
        print(f'higgs n={n} {hypothesis}: Liu {metric} {value:.3f}', flush=True)
        rows.append({'experiment': f'{hypothesis}_Higgs', 'n_samples': n,
                     'method': 'Liu1+PermSplit2', 'metric': metric, 'value': value,
                     'n_reps': n_reps})
    write_rows(f'higgs_n{n}', rows)


def tep(n_reps):
    import run_tep as rt
    rt.log_fh = open(os.path.join(OUT_DIR, 'tep.log'), 'w', buffering=1)
    n = rt.N_SAMPLES

    def draw(fault_type, offset):
        def _draw(rep):   # as rt.run_tep_experiment
            seed = rep + offset
            np.random.seed(seed)
            torch.manual_seed(seed)
            return rt.generate_tep_data(n, fault_type, rng=np.random.default_rng(seed))
        return _draw

    test = lambda X, Y: rt.step2_perm_split_deep(X, Y, rt.step1_liu, rt.INPUT_DIM)
    rows = []
    for fault_type, offset in (('none', 0), ('fault3', 1000), ('fault9', 2000), ('fault15', 3000)):
        hypothesis, metric = ('H0', 'type_I') if fault_type == 'none' else ('H1', 'power')
        value = _count_rejections(f'tep {fault_type}', draw(fault_type, offset), test, n_reps)
        print(f'tep {fault_type}: Liu {metric} {value:.3f}', flush=True)
        rows.append({'experiment': f'{hypothesis}_{fault_type}', 'fault_type': fault_type,
                     'n_samples': n, 'method': 'Liu1+PermSplit2', 'metric': metric,
                     'value': value, 'n_reps': n_reps})
    write_rows('tep', rows)


def cifar(pca_dim, n, n_reps):
    import run_cifar as rc
    c10, _ = rc.load_cifar10_test()
    c101, _ = rc.load_cifar101()
    f10, f101, _ = rc.prepare_features(c10, c101, pca_dim)
    n_per = min(n, len(f101))

    def draw(hypothesis):
        def _draw(rep):   # as rc.run_cifar_experiment
            seed = rep + (1000 if hypothesis == 'H1' else 5000)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hypothesis == 'H1':
                idx_p = np.random.choice(len(f10), n_per, replace=False)
                idx_q = np.random.choice(len(f101), n_per, replace=False)
                return f10[idx_p], f101[idx_q]
            idx_all = np.random.permutation(len(f10))
            return f10[idx_all[:n_per]], f10[idx_all[n_per:2 * n_per]]
        return _draw

    test = lambda X, Y: rc.step2_perm_split_deep(X, Y, rc.step1_liu, f10.shape[1])
    rows = []
    for hypothesis, label, metric in (('H1', 'H1_CIFAR10', 'power'), ('H0', 'H0_CIFAR10', 'type_I')):
        value = _count_rejections(f'cifar {hypothesis}', draw(hypothesis), test, n_reps)
        print(f'cifar {hypothesis}: Liu {metric} {value:.3f}', flush=True)
        rows.append({'experiment': label, 'n_samples': n_per, 'method': 'Liu1+PermSplit2',
                     'metric': metric, 'value': value, 'n_reps': n_reps})
    write_rows('cifar', rows)


def _merge_into(path, new_rows, is_liu, key, fields):
    """Overwrite `fields` of the Liu rows of `path` with the matching new rows."""
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    new = {key(r): r for r in new_rows}
    liu_rows = [r for r in rows if is_liu(r)]
    missing = [key(r) for r in liu_rows if key(r) not in new]
    assert not missing, f'{path}: no new result for {missing}'
    for r in liu_rows:
        for field in fields:
            r[field] = new[key(r)][field]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'merged {len(liu_rows)} Liu rows into {path}', flush=True)


def merge():
    def load(pattern):
        rows = []
        for path in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
            with open(path) as f:
                rows.extend(csv.DictReader(f))
        return rows

    _merge_into(os.path.join(RESULTS, 'deep_summary', 'deep_summary_results.csv'),
                load('table2_*.csv'), lambda r: r['criterion'] == 'Liu',
                lambda r: (int(r['dim']), int(r['n_samples']), float(r['delta'])),
                ('rejection_rate', 'n_valid', 'time_s'))
    _merge_into(os.path.join(RESULTS, 'liu_mixture_ablation', 'results_final.csv'),
                load('table7_*.csv'), lambda r: r['method'] in ('C_Liu_mix', 'D_Liu_pure'),
                lambda r: (r['experiment'], float(r['delta']), r['method']),
                ('rejection_rate', 'n_valid', 'n_reps', 'time_s'))
    _merge_into(os.path.join(RESULTS, 'higgs', 'higgs_results.csv'),
                load('higgs_n*.csv'), lambda r: r['method'] == 'Liu1+PermSplit2',
                lambda r: (r['experiment'], int(r['n_samples'])), ('value', 'n_reps'))
    _merge_into(os.path.join(RESULTS, 'tep', 'tep_results.csv'),
                load('tep.csv'), lambda r: r['method'] == 'Liu1+PermSplit2',
                lambda r: r['experiment'], ('value', 'n_reps'))
    _merge_into(os.path.join(RESULTS, 'cifar_pca500', 'cifar_results.csv'),
                load('cifar.csv'), lambda r: r['method'] == 'Liu1+PermSplit2',
                lambda r: (r['experiment'], int(r['n_samples'])), ('value', 'n_reps'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', choices=['table2', 'table7', 'higgs', 'tep', 'cifar', 'merge'])
    parser.add_argument('--cell')
    parser.add_argument('--method', choices=['C_Liu_mix', 'D_Liu_pure'])
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--pca-dim', type=int, default=500)
    args = parser.parse_args()
    if args.job == 'table2':
        table2(args.cell)
    elif args.job == 'table7':
        table7(args.cell, args.method, args.reps or 100)
    elif args.job == 'higgs':
        higgs(args.n, args.reps or 50)
    elif args.job == 'tep':
        tep(args.reps or 50)
    elif args.job == 'cifar':
        cifar(args.pca_dim, args.n, args.reps or 50)
    else:
        merge()


if __name__ == '__main__':
    main()
