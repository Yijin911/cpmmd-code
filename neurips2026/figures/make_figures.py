#!/usr/bin/env python
"""Generate figures for CP-MMD paper main text.

Outputs:
  figures/class_richness.pdf  -- two-panel line plot replacing Table 1
  figures/deployment_cost.pdf -- single-panel log-log line plot replacing Table 3
"""
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)


def colours():
    # ColorBrewer Set1 — colour-blind safe enough for 3 classes; line styles add redundancy
    return {
        'CP-MMD': '#377EB8',  # blue
        'Liu': '#E41A1C',     # red
        'Plain': '#FF7F00',   # orange
        'MMDAgg': '#E41A1C',
        'MMD-FUSE': '#FF7F00',
    }


def _load_srm_width():
    path = os.path.join(PROJECT_ROOT, 'results', 'srm_width', 'srm_width_results.csv')
    width, p_plain, p_liu, p_cp, s_plain, s_liu, s_cp = [], [], [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            width.append(int(row['width']))
            p_plain.append(float(row['power_plain']))
            p_liu.append(float(row['power_liu']))
            p_cp.append(float(row['power_cpmmd']))
            s_plain.append(float(row['spec_plain_mean']))
            s_liu.append(float(row['spec_liu_mean']))
            s_cp.append(float(row['spec_cpmmd_mean']))
    return width, (p_plain, p_liu, p_cp), (s_plain, s_liu, s_cp)


def fig_class_richness():
    """Single-panel main-text figure: power vs network width w (three failure modes)."""
    width, (p_plain, p_liu, p_cp), _ = _load_srm_width()
    c = colours()
    n_reps = 50
    p_cp_arr = np.array(p_cp)
    p_liu_arr = np.array(p_liu)
    p_plain_arr = np.array(p_plain)
    cp_se = np.sqrt(p_cp_arr * (1 - p_cp_arr) / n_reps)
    liu_se = np.sqrt(p_liu_arr * (1 - p_liu_arr) / n_reps)
    plain_se = np.sqrt(p_plain_arr * (1 - p_plain_arr) / n_reps)
    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    ax.fill_between(width, p_cp_arr - cp_se, p_cp_arr + cp_se,
                    color=c['CP-MMD'], alpha=0.18, zorder=1)
    ax.fill_between(width, p_liu_arr - liu_se, p_liu_arr + liu_se,
                    color=c['Liu'], alpha=0.18, zorder=1)
    ax.fill_between(width, p_plain_arr - plain_se, p_plain_arr + plain_se,
                    color=c['Plain'], alpha=0.18, zorder=1)
    h1, = ax.plot(width, p_cp, 'o-', color=c['CP-MMD'], linewidth=2.2, markersize=6,
                  label='CP-MMD (ours)', zorder=3)
    h2, = ax.plot(width, p_liu, 's--', color=c['Liu'], linewidth=1.5, markersize=5,
                  label='Liu (ratio)', zorder=2)
    h3, = ax.plot(width, p_plain, '^:', color=c['Plain'], linewidth=1.5, markersize=5,
                  label='Plain max', zorder=2)
    ax.set_xscale('log')
    ax.set_xticks(width)
    ax.set_xticklabels([str(w) for w in width])
    ax.set_xlabel(r'Network hidden width $w$')
    ax.set_ylabel('Power')
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.annotate('Liu collapses\n(variance collapse)', xy=(200, 0.50), xytext=(95, 0.25),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=c['Liu'], lw=0.8))
    ax.annotate('Plain collapses\n(over-fits training)', xy=(500, 0.0), xytext=(380, 0.18),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=c['Plain'], lw=0.8))
    ax.legend(loc='center right', fontsize=8, frameon=False, handlelength=1.5,
              handletextpad=0.4, borderaxespad=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    out = os.path.join(ROOT, 'class_richness.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}')
    plt.close(fig)


def fig_spectral_brake_trajectory():
    """Appendix figure: spectral-norm product Π vs network width w; the empirical
    companion to Proposition spectral-bound. CP-MMD's brake holds Π near a
    fixed point Π*∝1/Ĉ₁; Liu and Plain let Π blow up unboundedly."""
    width, _, (s_plain, s_liu, s_cp) = _load_srm_width()
    c = colours()
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    h1, = ax.plot(width, s_cp, 'o-', color=c['CP-MMD'], linewidth=2.2, markersize=6,
                  label='CP-MMD (ours)', zorder=3)
    h2, = ax.plot(width, s_liu, 's--', color=c['Liu'], linewidth=1.5, markersize=5,
                  label='Liu (ratio)', zorder=2)
    h3, = ax.plot(width, s_plain, '^:', color=c['Plain'], linewidth=1.5, markersize=5,
                  label='Plain max', zorder=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(width)
    ax.set_xticklabels([str(w) for w in width])
    ax.set_xlabel(r'Network hidden width $w$')
    ax.set_ylabel(r'Spectral-norm product $\Pi=\prod_j\|W_j\|_2$')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.axhspan(1.5, 2.2, alpha=0.15, color=c['CP-MMD'], zorder=1)
    ax.annotate(r'CP-MMD brake at $\Pi^*\propto 1/\widehat C_1$', xy=(900, 1.7),
                xytext=(60, 8), fontsize=7, color=c['CP-MMD'], ha='left', va='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=c['CP-MMD'], lw=0.8))
    ax.legend(loc='upper left', fontsize=8, frameon=False, handlelength=1.5,
              handletextpad=0.4, borderaxespad=0.3)
    plt.tight_layout()
    out = os.path.join(ROOT, 'spectral_brake_trajectory.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}')
    plt.close(fig)


def fig_pillar_power():
    """Side-by-side line plots replacing Tables 2 (linear δ-sweep) and 3 (poly df-sweep).
    Each dot annotated with exact value for reviewer-quotable numbers; shaded
    bands show ±1 binomial SE (linear: n_reps=100; poly: n_reps=200)."""
    # Linear regime: hard-coded from results/linear_pillar/linear_pillar_results.csv
    # (rows ordered by delta in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30})
    deltas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    cp_lin = [0.030, 0.060, 0.270, 0.570, 0.840, 0.990]
    med_lin = [0.060, 0.070, 0.310, 0.410, 0.400, 0.530]
    n_reps_lin = 100
    # Polynomial regime: hard-coded from results/polynomial_200reps/kurtosis_results.csv
    # (CP-MMD (Lap+poly4) with J_CP; MMDAgg / Median rows at df in {5, 8, 12, 20})
    dfs = [5, 8, 12, 20]
    cp_poly = [1.000, 1.000, 0.965, 0.600]
    agg_poly = [1.000, 1.000, 0.960, 0.370]
    med_poly = [1.000, 0.455, 0.190, 0.050]
    n_reps_poly = 200

    def binomial_se(p_list, n_reps):
        return [np.sqrt(p * (1 - p) / n_reps) for p in p_list]

    cp_lin_se = binomial_se(cp_lin, n_reps_lin)
    med_lin_se = binomial_se(med_lin, n_reps_lin)
    cp_poly_se = binomial_se(cp_poly, n_reps_poly)
    agg_poly_se = binomial_se(agg_poly, n_reps_poly)
    med_poly_se = binomial_se(med_poly, n_reps_poly)

    c = colours()
    cp_color = c['CP-MMD']
    median_color = '#4DAF4A'
    agg_color = '#984EA3'

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.0),
                            gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel (a): linear regime
    ax = axs[0]
    ax.fill_between(deltas, np.array(cp_lin) - np.array(cp_lin_se),
                    np.array(cp_lin) + np.array(cp_lin_se),
                    color=cp_color, alpha=0.18, zorder=1)
    ax.fill_between(deltas, np.array(med_lin) - np.array(med_lin_se),
                    np.array(med_lin) + np.array(med_lin_se),
                    color=median_color, alpha=0.18, zorder=1)
    h_cp_a, = ax.plot(deltas, cp_lin, 'o-', color=cp_color, linewidth=2.2, markersize=6,
                       label='CP-MMD', zorder=3)
    h_med, = ax.plot(deltas, med_lin, 's--', color=median_color, linewidth=1.6, markersize=5,
                      label='Median heuristic', zorder=2)
    for x, y in zip(deltas, cp_lin):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=6.5, color=cp_color)
    for x, y in zip(deltas, med_lin):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, -10),
                    ha='center', fontsize=6.5, color=median_color)
    ax.set_xlabel(r'Mean shift $\Delta$')
    ax.set_ylabel('Power')
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(deltas)
    ax.set_title(r'(a) Linear: CP-MMD bandwidth selection, $n{=}200$', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel (b): polynomial regime
    ax = axs[1]
    ax.fill_between(dfs, np.array(cp_poly) - np.array(cp_poly_se),
                    np.array(cp_poly) + np.array(cp_poly_se),
                    color=cp_color, alpha=0.18, zorder=1)
    ax.fill_between(dfs, np.array(agg_poly) - np.array(agg_poly_se),
                    np.array(agg_poly) + np.array(agg_poly_se),
                    color=agg_color, alpha=0.18, zorder=1)
    ax.fill_between(dfs, np.array(med_poly) - np.array(med_poly_se),
                    np.array(med_poly) + np.array(med_poly_se),
                    color=median_color, alpha=0.18, zorder=1)
    ax.plot(dfs, cp_poly, 'o-', color=cp_color, linewidth=2.2, markersize=6, zorder=3)
    h_agg, = ax.plot(dfs, agg_poly, 'D--', color=agg_color, linewidth=1.6, markersize=5,
                      label='MMDAgg', zorder=2)
    ax.plot(dfs, med_poly, 's--', color=median_color, linewidth=1.6, markersize=5, zorder=2)
    for x, y in zip(dfs, cp_poly):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=6.5, color=cp_color)
    for x, y in zip(dfs, agg_poly):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, -10),
                    ha='center', fontsize=6.5, color=agg_color)
    for x, y in zip(dfs, med_poly):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, -10),
                    ha='center', fontsize=6.5, color=median_color)
    ax.set_xlabel(r'Degrees of freedom (df)')
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(dfs)
    ax.set_title(r'(b) Polynomial: Lap+poly$^{(4)}$ CP-MMD, $n{=}500$', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Shared legend at the bottom of the whole figure
    fig.legend(handles=[h_cp_a, h_agg, h_med], loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8, frameon=False,
               handlelength=1.8, handletextpad=0.4, columnspacing=2.0)

    plt.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(ROOT, 'pillar_power.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}')
    plt.close(fig)
    plt.close(fig)


def fig_deployment_cost():
    """Single-panel: per-test wall-clock vs aggregation grid size B.

    Shows multiple CP-MMD ansatz reference lines (linear, poly3) so the
    comparison covers more than one kernel family. MMDAgg is run with its
    standard 'laplace_gaussian' configuration (Schrab et al.'s default),
    giving 2*B effective kernels. Colors of CP-MMD ansätze match Fig 3's
    panel (a) so the legend can be shared."""
    path = os.path.join(PROJECT_ROOT, 'results', 'deployment_cost', 'deployment_cost.csv')
    rows = list(csv.DictReader(open(path)))
    by_method = {}
    for r in rows:
        m = r['method']
        by_method.setdefault(m, {'B': [], 'ms': []})
        by_method[m]['B'].append(int(r['B']))
        by_method[m]['ms'].append(float(r['per_test_median_s']) * 1000)

    fig, ax = plt.subplots(figsize=(2.8, 1.8))

    # CP-MMD blues match Fig 3 (a)'s ansatz palette
    cp_lin_color = '#9ECAE1'    # CP-MMD linear (light blue)
    cp_p3_color = '#377EB8'     # CP-MMD poly3 (strong blue, contrast with linear)
    fuse_color = '#A65628'      # brown
    agg_color = '#984EA3'       # purple, matches MMDAgg in pillar_power and Fig 3

    Bs_full = [5, 10, 20, 50, 100]

    # CP-MMD reference lines for two ansätze
    if 'CP-MMD-linear' in by_method:
        cp_lin_ms = by_method['CP-MMD-linear']['ms'][0]
    else:
        cp_lin_ms = by_method['CP-MMD']['ms'][0]
    h_cp_lin = ax.axhline(cp_lin_ms, color=cp_lin_color, linewidth=1.8, linestyle='-',
                          label='CP-MMD linear')

    if 'CP-MMD-poly3' in by_method:
        cp_p3_ms = by_method['CP-MMD-poly3']['ms'][0]
        h_cp_p3 = ax.axhline(cp_p3_ms, color=cp_p3_color, linewidth=1.8, linestyle='-',
                             label='CP-MMD poly3')
    else:
        h_cp_p3 = None

    # Aggregation methods
    Bs_agg = by_method['MMDAgg']['B']
    ms_agg = by_method['MMDAgg']['ms']
    Bs_fuse = by_method['MMD-FUSE']['B']
    ms_fuse = by_method['MMD-FUSE']['ms']
    h_agg, = ax.plot(Bs_agg, ms_agg, 's--', color=agg_color, linewidth=1.5, markersize=5,
                     label='MMDAgg')
    h_fuse, = ax.plot(Bs_fuse, ms_fuse, '^:', color=fuse_color, linewidth=1.5, markersize=5,
                      label='MMD-FUSE')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(Bs_full)
    ax.set_xticklabels([str(b) for b in Bs_full])
    ax.set_xlabel(r'Aggregation grid size $B$')
    ax.set_ylabel('Wall-clock (ms)')
    # Auto-scale y, but ensure CP-MMD references visible
    all_ms = list(ms_agg) + list(ms_fuse) + [cp_lin_ms]
    if h_cp_p3 is not None:
        all_ms.append(cp_p3_ms)
    ax.set_ylim(min(all_ms) * 0.5, max(all_ms) * 1.5)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [h_cp_lin]
    if h_cp_p3 is not None:
        handles.append(h_cp_p3)
    handles.extend([h_agg, h_fuse])
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7, frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(ROOT, 'deployment_cost.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.18)
    print(f'wrote {out}')
    plt.close(fig)


def fig_real_data():
    """Bar chart: power on Higgs at n=200 and n=500 for all three CP-MMD variants
    (linear, polynomial, deep) plus the regime-specific competitors (Median, MMDAgg)
    and deep-criterion competitors (Liu, Plain).

    Story: all three CP-MMD variants are functional on real data; the deep variant is
    the strongest CP-MMD; on this single-scale signal, fixed-bandwidth Median/MMDAgg
    happen to be competitive overall."""
    higgs_path = os.path.join(PROJECT_ROOT, 'results', 'higgs', 'higgs_results.csv')
    extra_v2_path = os.path.join(PROJECT_ROOT, 'results', 'higgs_extra_v2', 'higgs_extra_v2_results.csv')

    def read_power(path, n_target):
        """Read deep-CP-MMD/Liu/Plain/Median/MMDAgg from higgs_path (no mode column)."""
        out = {}
        for r in csv.DictReader(open(path)):
            if 'experiment' not in r or r['experiment'] != 'H1_Higgs':
                continue
            if int(r['n_samples']) != n_target or r['metric'] != 'power':
                continue
            out[r['method']] = float(r['value'])
        return out

    def read_power_v2(path, n_target, mode_filter='permsplit'):
        """Read CP-MMD-linear/poly2/poly3 from v2 CSV (has mode column).
        Use full-sample protocol to match Median/MMDAgg's full-sample evaluation."""
        out = {}
        for r in csv.DictReader(open(path)):
            if r['experiment'] != 'H1_Higgs':
                continue
            if int(r['n_samples']) != n_target or r['metric'] != 'power':
                continue
            if r['mode'] != mode_filter:
                continue
            out[r['method']] = float(r['value'])
        return out

    h200 = read_power(higgs_path, 200)
    h500 = read_power(higgs_path, 500)
    h200.update(read_power_v2(extra_v2_path, 200, 'permsplit'))  # sample-split protocol of Sec. 3.3
    h500.update(read_power_v2(extra_v2_path, 500, 'permsplit'))

    # Four CP-MMD variants on the left (linear, poly2, poly3, deep — our framework),
    # two deep-criterion competitors in the middle (Liu, Plain — same MLP as CP-MMD-deep),
    # two fixed-scale baselines on the right (Median, MMDAgg).
    # Protocol: linear/poly{2,3} use full sample (matching Median/MMDAgg); deep uses
    # PermSplit (matching Liu/Plain).
    methods = [
        'CPMMD-linear', 'CPMMD-poly2', 'CPMMD-poly3', 'CPMMD1+PermSplit2',  # our four variants
        'Liu1+PermSplit2', 'Plain1+PermSplit2',                              # deep-criterion competitors
        'Median+Perm', 'MMDAgg+Perm',                                        # fixed-scale baselines
    ]
    method_label = {
        'CPMMD-linear': 'CP-MMD linear',
        'CPMMD-poly2': 'CP-MMD poly2',
        'CPMMD-poly3': 'CP-MMD poly3',
        'CPMMD1+PermSplit2': 'CP-MMD deep',
        'Liu1+PermSplit2': 'Liu',
        'Plain1+PermSplit2': 'Plain',
        'Median+Perm': 'Median',
        'MMDAgg+Perm': 'MMDAgg',
    }
    c = colours()
    # Four shades of CP-MMD blue for the four variants; competitors as before.
    cp_blue = c['CP-MMD']
    method_color = {
        'CPMMD-linear': '#C6DBEF',          # lightest blue
        'CPMMD-poly2': '#9ECAE1',           # light blue
        'CPMMD-poly3': '#6BAED6',           # mid blue
        'CPMMD1+PermSplit2': cp_blue,       # deep CP-MMD = strong blue
        'Liu1+PermSplit2': c['Liu'],
        'Plain1+PermSplit2': c['Plain'],
        'Median+Perm': '#4DAF4A',
        'MMDAgg+Perm': '#984EA3',
    }
    method_hatch = {
        'CPMMD-linear': '',
        'CPMMD-poly2': '',
        'CPMMD-poly3': '',
        'CPMMD1+PermSplit2': '',
        'Liu1+PermSplit2': '',
        'Plain1+PermSplit2': '',
        'Median+Perm': '//',
        'MMDAgg+Perm': '//',
    }

    datasets = [
        ('Higgs, $n{=}200$', h200),
        ('Higgs, $n{=}500$', h500),
    ]
    n_methods = len(methods)
    width = 0.11
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    width = 0.10
    for i, m in enumerate(methods):
        vals = [d.get(m, 0.0) for _, d in datasets]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=method_label[m],
               color=method_color[m], edgecolor='white', linewidth=0.3,
               hatch=method_hatch[m])

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=8)
    ax.set_ylabel('Power', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Legend below the plot, two rows of four
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4,
              frameon=False, columnspacing=0.8, handletextpad=0.3, fontsize=6.5,
              handlelength=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out = os.path.join(ROOT, 'real_data.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}')
    plt.close(fig)


def fig_higgs_deployment():
    """Combined Fig 3: (a) Higgs power bar chart, (b) deployment cost line plot,
    with a shared legend at bottom listing all methods (union of panel methods).
    Replaces the two-PDF / two-legend layout with one PDF / one legend."""
    higgs_path = os.path.join(PROJECT_ROOT, 'results', 'higgs', 'higgs_results.csv')
    extra_v2_path = os.path.join(PROJECT_ROOT, 'results', 'higgs_extra_v2', 'higgs_extra_v2_results.csv')

    def read_power(path, n_target):
        out = {}
        for r in csv.DictReader(open(path)):
            if 'experiment' not in r or r['experiment'] != 'H1_Higgs':
                continue
            if int(r['n_samples']) != n_target or r['metric'] != 'power':
                continue
            out[r['method']] = float(r['value'])
        return out

    def read_power_v2(path, n_target, mode_filter='permsplit'):
        out = {}
        for r in csv.DictReader(open(path)):
            if r['experiment'] != 'H1_Higgs':
                continue
            if int(r['n_samples']) != n_target or r['metric'] != 'power':
                continue
            if r['mode'] != mode_filter:
                continue
            out[r['method']] = float(r['value'])
        return out

    h200 = read_power(higgs_path, 200)
    h500 = read_power(higgs_path, 500)
    h200.update(read_power_v2(extra_v2_path, 200, 'permsplit'))  # sample-split protocol of Sec. 3.3
    h500.update(read_power_v2(extra_v2_path, 500, 'permsplit'))

    cost_path = os.path.join(PROJECT_ROOT, 'results', 'deployment_cost', 'deployment_cost.csv')
    cost_rows = list(csv.DictReader(open(cost_path)))
    by_method = {}
    for r in cost_rows:
        m = r['method']
        by_method.setdefault(m, {'B': [], 'ms': []})
        by_method[m]['B'].append(int(r['B']))
        by_method[m]['ms'].append(float(r['per_test_median_s']) * 1000)

    # Unified colours: methods in BOTH panels share colour; panel-unique methods get their own.
    color = {
        'CPMMD-linear': '#9ECAE1',
        'CPMMD-ard':    '#17BECF',
        'CPMMD-poly2':  '#6BAED6',
        'CPMMD-poly3':  '#377EB8',
        'CPMMD-deep':   '#08519C',
        'Liu':          '#E41A1C',
        'Plain':        '#FF7F00',
        'Median':       '#4DAF4A',
        'MMDAgg':       '#984EA3',
        'MMD-FUSE':     '#A65628',
    }
    label = {
        'CPMMD-linear': 'CP-MMD linear',
        'CPMMD-ard':    'CP-MMD ARD',
        'CPMMD-poly2':  'CP-MMD poly2',
        'CPMMD-poly3':  'CP-MMD poly3',
        'CPMMD-deep':   'CP-MMD deep',
        'Liu':          'Liu',
        'Plain':        'Plain',
        'Median':       'Median',
        'MMDAgg':       'MMDAgg',
        'MMD-FUSE':     'MMD-FUSE',
    }

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.8),
                            gridspec_kw={'width_ratios': [1.5, 1]})

    # ---- Panel (a): Higgs bar chart ----
    ax = axs[0]
    higgs_methods = ['CPMMD-linear', 'CPMMD-ard', 'CPMMD-poly2', 'CPMMD-poly3', 'CPMMD-deep',
                     'Liu', 'Plain', 'Median', 'MMDAgg']
    csv_key = {
        'CPMMD-linear': 'CPMMD-linear',
        'CPMMD-ard': 'CPMMD-ard',
        'CPMMD-poly2': 'CPMMD-poly2',
        'CPMMD-poly3': 'CPMMD-poly3',
        'CPMMD-deep': 'CPMMD1+PermSplit2',
        'Liu': 'Liu1+PermSplit2',
        'Plain': 'Plain1+PermSplit2',
        'Median': 'Median+Perm',
        'MMDAgg': 'MMDAgg+Perm',
    }
    hatch = {m: ('//' if m in ('Median', 'MMDAgg') else '') for m in higgs_methods}
    datasets = [('Higgs, $n{=}200$', h200), ('Higgs, $n{=}500$', h500)]
    n_meth = len(higgs_methods)
    bw = 0.10
    x = np.arange(len(datasets))
    panel_a_handles = {}
    for i, m in enumerate(higgs_methods):
        vals = [d.get(csv_key[m], 0.0) for _, d in datasets]
        offset = (i - (n_meth - 1) / 2) * bw
        bar = ax.bar(x + offset, vals, bw, color=color[m], edgecolor='white',
                     linewidth=0.3, hatch=hatch[m])
        panel_a_handles[m] = bar[0]
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=8)
    ax.set_ylabel('Power', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_title('(a) Higgs power', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Panel (b): Deployment cost line plot ----
    ax = axs[1]
    cp_lin_ms = by_method['CP-MMD-linear']['ms'][0]
    cp_p3_ms = by_method['CP-MMD-poly3']['ms'][0]
    Bs_full = [5, 10, 20, 50, 100]
    panel_b_handles = {}
    panel_b_handles['CPMMD-linear'] = ax.axhline(
        cp_lin_ms, color=color['CPMMD-linear'], linewidth=1.8, linestyle='-')
    panel_b_handles['CPMMD-poly3'] = ax.axhline(
        cp_p3_ms, color=color['CPMMD-poly3'], linewidth=1.8, linestyle='-')
    h_agg, = ax.plot(by_method['MMDAgg']['B'], by_method['MMDAgg']['ms'],
                     's--', color=color['MMDAgg'], linewidth=1.5, markersize=5)
    h_fuse, = ax.plot(by_method['MMD-FUSE']['B'], by_method['MMD-FUSE']['ms'],
                      '^:', color=color['MMD-FUSE'], linewidth=1.5, markersize=5)
    panel_b_handles['MMDAgg'] = h_agg
    panel_b_handles['MMD-FUSE'] = h_fuse
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(Bs_full)
    ax.set_xticklabels([str(b) for b in Bs_full])
    ax.set_xlabel(r'Aggregation grid size $B$', fontsize=8)
    ax.set_ylabel('Wall-clock (ms)', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    all_ms = list(by_method['MMDAgg']['ms']) + list(by_method['MMD-FUSE']['ms']) + \
             [cp_lin_ms, cp_p3_ms]
    ax.set_ylim(min(all_ms) * 0.5, max(all_ms) * 1.5)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_title('(b) Deployment cost', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Shared legend at bottom (union of methods) ----
    legend_order = ['CPMMD-linear', 'CPMMD-ard', 'CPMMD-poly2', 'CPMMD-poly3', 'CPMMD-deep',
                    'Liu', 'Plain', 'Median', 'MMDAgg', 'MMD-FUSE']
    handles = []
    for m in legend_order:
        if m in panel_a_handles:
            handles.append(panel_a_handles[m])
        elif m in panel_b_handles:
            handles.append(panel_b_handles[m])
    fig.legend(handles=handles, labels=[label[m] for m in legend_order],
               loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=5,
               frameon=False, fontsize=7, columnspacing=1.0,
               handletextpad=0.3, handlelength=1.4)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(ROOT, 'higgs_deployment.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.10)
    print(f'wrote {out}')
    plt.close(fig)


def fig_c1_ablation():
    """Log-log: Π* vs Ĉ1, empirical points + empirical-fit line on the productive range.
    Power on right axis."""
    path = os.path.join(PROJECT_ROOT, 'results', 'c1_ablation', 'c1_ablation_results.csv')
    rows = list(csv.DictReader(open(path)))
    C1_vals, pi_vals, pow_vals = [], [], []
    for r in rows:
        c1 = float(r['C1'])
        if c1 <= 0:
            continue  # skip C1=0 (log-undefined)
        C1_vals.append(c1)
        pi_vals.append(float(r['mean_pi']))
        pow_vals.append(float(r['power']))

    c = colours()
    fig, ax = plt.subplots(figsize=(4.6, 2.7))

    # Empirical Pi
    h1, = ax.loglog(C1_vals, pi_vals, 'o', color=c['CP-MMD'], markersize=7, markerfacecolor=c['CP-MMD'],
                    markeredgecolor='white', markeredgewidth=0.6, label=r'Empirical $\Pi^*$', zorder=3)

    # Productive-range mask (saturation excluded at both endpoints)
    productive_mask = [i for i, c1v in enumerate(C1_vals) if 1e-3 <= c1v <= 1e-1]

    # Empirical fit on the productive range (least-squares slope on log-log).
    xs = np.array([np.log(C1_vals[i]) for i in productive_mask])
    ys = np.array([np.log(pi_vals[i]) for i in productive_mask])
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope_emp, intercept_emp = np.linalg.lstsq(A, ys, rcond=None)[0]
    C1_prod_grid = np.logspace(-3, -1, 50)
    fit_line = np.exp(intercept_emp) * C1_prod_grid ** slope_emp
    h3, = ax.loglog(C1_prod_grid, fit_line, ':', color='#377EB8', linewidth=1.7,
                    label=fr'Empirical fit (productive range, slope ${slope_emp:.2f}$)',
                    zorder=2, alpha=0.9)

    ax.set_xlabel(r'Calibrated $\widehat C_1$')
    ax.set_ylabel(r'$\Pi^*$ at convergence')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Power on right axis
    ax2 = ax.twinx()
    h4, = ax2.plot(C1_vals, pow_vals, 's', color='#666666', markersize=5, alpha=0.7,
                   label='Power (right axis)')
    ax2.set_ylabel('Power', color='#666666')
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(axis='y', labelcolor='#666666')
    ax2.axhline(1.0, color='#666666', alpha=0.2, linestyle=':')

    # Legend below the plot, outside the panel (3 entries)
    fig.legend(handles=[h1, h3, h4], loc='lower center',
               bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=7,
               columnspacing=1.2, handletextpad=0.4)

    plt.tight_layout(rect=(0, 0.14, 1, 1))
    out = os.path.join(ROOT, 'c1_ablation.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}')
    plt.close(fig)


if __name__ == '__main__':
    fig_class_richness()
    fig_pillar_power()
    fig_deployment_cost()
    fig_real_data()
    fig_higgs_deployment()
    fig_c1_ablation()
    fig_spectral_brake_trajectory()
