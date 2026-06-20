"""
Make the paper figure for the linear pillar.

Two-panel figure:
  (a) Power vs delta — CP-MMD bandwidth selection vs Median heuristic
  (b) Selected sigma vs delta — showing CP-MMD finds the within-cluster
      scale (~0.2) while Median is stuck at the inter-cluster gap (~2.5)
"""

import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT, 'results', 'linear_pillar', 'linear_pillar_results.csv')

deltas, cp_rate, md_rate, cp_sig, md_sig = [], [], [], [], []
seen_d = set()
with open(csv_path) as f:
    rdr = csv.DictReader(f)
    rows = list(rdr)

for d in sorted(set(float(r['delta']) for r in rows)):
    deltas.append(d)
    for r in rows:
        if float(r['delta']) != d: continue
        if r['method'] == 'CPMMD_bw_select':
            cp_rate.append(float(r['rate']))
            cp_sig.append(float(r['median_sigma']))
        elif r['method'] == 'Median_heuristic':
            md_rate.append(float(r['rate']))
            md_sig.append(float(r['median_sigma']))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(deltas, cp_rate, 'o-', color='C0', label='CP-MMD bandwidth selection', linewidth=2)
ax.plot(deltas, md_rate, 's--', color='C1', label='Median heuristic', linewidth=2)
ax.axhline(0.05, color='gray', linestyle=':', alpha=0.6, label=r'nominal $\alpha = 0.05$')
ax.set_xlabel(r'mean shift $\delta$')
ax.set_ylabel('rejection rate (power; H0 = Type I)')
ax.set_title('(a) Linear pillar: power vs. shift')
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right', fontsize=9, frameon=False)
ax.grid(alpha=0.3)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.text(0.0, 1.02, 'H0', ha='center', fontsize=8, color='gray')

ax = axes[1]
ax.plot(deltas, cp_sig, 'o-', color='C0', label='CP-MMD selected $\\sigma$', linewidth=2)
ax.plot(deltas, md_sig, 's--', color='C1', label='Median pairwise distance', linewidth=2)
ax.axhline(0.1, color='C2', linestyle=':', label='within-cluster scale (0.1)')
ax.axhline(3.0, color='C3', linestyle=':', label='inter-cluster gap (3.0)')
ax.set_yscale('log')
ax.set_xlabel(r'mean shift $\delta$')
ax.set_ylabel('selected bandwidth $\\sigma$')
ax.set_title('(b) Selected bandwidth')
ax.legend(loc='center right', fontsize=8, frameon=False)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out = os.path.join(ROOT, 'figures', 'linear_pillar.pdf')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"Saved {out}")
