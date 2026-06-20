"""Minimal one-line schematic for the degree-4 polynomial feature map."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(11.5, 2.4))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 2.4); ax.axis('off')

ax.text(1.9, 1.55, r'$(x_1,\dots,x_{10})$', ha='center', va='center', fontsize=21)
ax.annotate('', xy=(4.4, 1.55), xytext=(3.5, 1.55),
            arrowprops=dict(arrowstyle='-|>', lw=2.6, color='#333333'))
ax.text(3.95, 2.0, r'$\Psi_4$', ha='center', va='center', fontsize=20, color='#08519c')
ax.text(7.95, 1.55,
        r'$(\,x_i,\ \ x_i x_j,\ \ x_i x_j x_k,\ \ x_i x_j x_k x_l\,)$',
        ha='center', va='center', fontsize=21)
ax.text(5.75, 0.45,
        r'all monomials of degree $\leq 4$:'
        r'$\quad\mathbb{R}^{10}\rightarrow\mathbb{R}^{1000}$',
        ha='center', va='center', fontsize=16, color='#444444')

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
fig.savefig(os.path.join(outdir, 'psi4_brief.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'psi4_brief.png'), dpi=200, bbox_inches='tight')
print('done')
