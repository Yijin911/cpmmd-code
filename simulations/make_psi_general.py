"""Minimal one-line schematic for the general degree-p polynomial feature map."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(11.8, 2.4))
ax.set_xlim(0, 11.8); ax.set_ylim(0, 2.4); ax.axis('off')

ax.text(1.7, 1.55, r'$(x_1,\dots,x_d)$', ha='center', va='center', fontsize=21)
ax.annotate('', xy=(4.1, 1.55), xytext=(3.2, 1.55),
            arrowprops=dict(arrowstyle='-|>', lw=2.6, color='#333333'))
ax.text(3.65, 2.0, r'$\Psi_p$', ha='center', va='center', fontsize=20, color='#08519c')
ax.text(8.0, 1.55,
        r'$(\,x_i,\ \ x_i x_j,\ \ \cdots,\ \ x_{i_1} x_{i_2}\cdots x_{i_p}\,)$',
        ha='center', va='center', fontsize=21)
ax.text(5.9, 0.45,
        r'all monomials of degree $\leq p$:'
        r'$\quad\mathbb{R}^{d}\rightarrow\mathbb{R}^{\binom{d+p}{p}-1}$',
        ha='center', va='center', fontsize=16, color='#444444')

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
fig.savefig(os.path.join(outdir, 'psi_general.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'psi_general.png'), dpi=200, bbox_inches='tight')
print('done')
