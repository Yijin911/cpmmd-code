"""Presentation schematic for the degree-4 polynomial feature map Psi_4.

Lifting diagram: input x in R^d  -->  Psi_4  -->  feature blocks (deg 1..4).
Bar length is proportional to the number of monomials at each degree, so the
combinatorial explosion (10 -> 55 -> 220 -> 715, total 1000 at d=10) is visible.
Outputs vector PDF + PNG into ../figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.size': 15,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
})

d = 10
degrees = [1, 2, 3, 4]
counts = [10, 55, 220, 715]          # C(d+p-1, p) for d=10
total = sum(counts)                   # 1000 = C(14,4) - 1

# sequential blues, light (deg 1) -> dark (deg 4); colour-blind safe
colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']

examples = {
    1: r'$x_1,\ x_2,\ \dots,\ x_{10}$',
    2: r'$x_1^2,\ x_1 x_2,\ \dots,\ x_{10}^2$',
    3: r'$x_1^3,\ x_1^2 x_2,\ \dots,\ x_{10}^3$',
    4: r'$x_1^4,\ x_1^2 x_2^2,\ \dots,\ x_{10}^4$',
}
formula = {1: r'$\binom{10}{1}$', 2: r'$\binom{11}{2}$',
           3: r'$\binom{12}{3}$', 4: r'$\binom{13}{4}$'}

fig, ax = plt.subplots(figsize=(13.5, 6.9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ---- title ----
ax.text(8.0, 8.55,
        r'Degree-4 polynomial feature map  '
        r'$\Psi_4:\ \mathbb{R}^{d}\rightarrow\mathbb{R}^{D}$',
        ha='center', va='center', fontsize=20, fontweight='bold')

# ---- input vector ----
inx, iny, inw, inh = 0.6, 2.3, 1.35, 3.6
ax.add_patch(FancyBboxPatch((inx, iny), inw, inh,
             boxstyle="round,pad=0.08,rounding_size=0.18",
             fc='#f2f2f2', ec='#444444', lw=1.6))
entries = [r'$x_1$', r'$x_2$', r'$x_3$', r'$\vdots$', r'$x_{10}$']
ys = np.linspace(iny + inh - 0.55, iny + 0.55, len(entries))
for e, y in zip(entries, ys):
    ax.text(inx + inw / 2, y, e, ha='center', va='center', fontsize=15)
ax.text(inx + inw / 2, iny - 0.45, r'$x\in\mathbb{R}^{10}$',
        ha='center', va='top', fontsize=15)

# ---- arrow + operator ----
ax.annotate('', xy=(4.55, 4.1), xytext=(2.25, 4.1),
            arrowprops=dict(arrowstyle='-|>', lw=3.2, color='#333333'))
ax.text(3.4, 4.95, r'$\Psi_4$', ha='center', va='center',
        fontsize=24, color='#08519c')
ax.text(3.4, 3.35, 'all monomials\n' + r'of degree $\leq 4$',
        ha='center', va='center', fontsize=12, color='#555555')

# ---- feature blocks (horizontal bars) ----
x0 = 5.05
maxlen = 7.4
scale = maxlen / max(counts)
row_h = 1.5
for i, (p, c) in enumerate(zip(degrees, counts)):
    yb = 1.55 + i * row_h                      # deg 1 at bottom
    ax.text(x0, yb + 0.92, r'degree %d:   %s' % (p, examples[p]),
            ha='left', va='center', fontsize=14.5)
    w = max(scale * c, 0.12)
    ax.add_patch(FancyBboxPatch((x0, yb), w, 0.5,
                 boxstyle="round,pad=0.0,rounding_size=0.05",
                 fc=colors[i], ec='none'))
    ax.text(x0 + w + 0.18, yb + 0.25,
            r'%s $=\,%d$' % (formula[p], c),
            ha='left', va='center', fontsize=13.5, color='#222222')

# ---- bottom annotations ----
ax.text(8.0, 0.78,
        r'Total dimension  $D=\binom{d+4}{4}-1=\mathbf{1000}$  for $d=10$',
        ha='center', va='center', fontsize=15.5)
ax.text(8.0, 0.22,
        r'blocks include cross terms ($x_i x_j,\ x_i^2 x_j^2,\dots$),'
        r' not only per-coordinate powers',
        ha='center', va='center', fontsize=11.5, color='#777777', style='italic')

fig.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(outdir, exist_ok=True)
pdf = os.path.join(outdir, 'psi4_feature_map.pdf')
png = os.path.join(outdir, 'psi4_feature_map.png')
fig.savefig(pdf, bbox_inches='tight')
fig.savefig(png, dpi=200, bbox_inches='tight')
print('wrote', pdf)
print('wrote', png)
