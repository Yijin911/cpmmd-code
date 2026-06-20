"""Proxy overview slide: (1) theory chain, (2) function set, (3) three instances.

Zone 1: the named results that license G_tilde (UCI -> monotone -> envelope -> calib).
Zone 2: the class H optimized over (composite kernel, trajectory).
Zone 3: linear / polynomial / deep instantiations of the envelope L(h).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({'font.size': 13, 'mathtext.fontset': 'cm', 'font.family': 'serif'})

BLUE = '#08519c'
LIGHT = '#eef3f9'
HILITE = '#fdebd0'

fig, ax = plt.subplots(figsize=(13.2, 8.6))
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 8.6)
ax.axis('off')


def box(x, y, w, h, text, fc=LIGHT, ec='#9bb0c7', fs=12, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def harrow(x0, x1, y, label=''):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', lw=2, color='#444444'))
    if label:
        ax.text((x0 + x1) / 2, y + 0.16, label, ha='center', va='bottom',
                fontsize=9.5, color='#666666')


# ===== Zone 1: theory chain =====
ax.text(0.2, 8.32, '1.  Theory that licenses the proxy',
        ha='left', va='center', fontsize=15, color=BLUE)
yb = 6.7
box(0.2, yb, 3.05, 1.2,
    'UCI  (Thm 1)\n' r'$\sup_h|\widehat\gamma^2-\gamma^2|\leq C_1 G(\mathcal{H})+C_2\sqrt{\frac{\ln(2/\delta)}{N}}$',
    fs=10.5)
harrow(3.3, 4.1, yb + 0.6, r'$\mathcal{H}_T\subseteq\mathcal{H}$' + '\nmonotone')
box(4.15, yb, 2.95, 1.2,
    'envelope bound  (Prop 1)\n' r'$G(\mathcal{H}_T)\leq A_T\,L^\ast\,\dfrac{\|\mathbf{D}\|_F}{N}$',
    fs=10.5)
harrow(7.15, 7.95, yb + 0.6, 'differentiable\nsurrogate')
box(8.0, yb, 2.6, 1.2, r'$\widetilde G_T(h)=L(h)\,\dfrac{\|\mathbf{D}\|_F}{N}$',
    fc=HILITE, ec='#e0a96d', fs=12.5, lw=1.9)
harrow(10.65, 11.4, yb + 0.6, 'calibration\n(Thm 2)')
box(11.45, yb, 1.65, 1.2,
    r'$\widehat C_1\sim r^\ast(\mathcal{H}_T)$' + '\nabsorbs gap', fs=10)
ax.text(6.65, 6.42,
        r'$\widehat C_1$ tracks the local Rademacher fixed point; $A_T$ and the global-to-local gap fold into it',
        ha='center', va='top', fontsize=9.5, color='#888888', style='italic')

# ===== Zone 2: function set =====
ax.text(0.2, 5.75, '2.  What we optimize over',
        ha='left', va='center', fontsize=15, color=BLUE)
ax.add_patch(FancyBboxPatch((0.2, 4.55), 12.8, 0.95,
             boxstyle="round,pad=0.06,rounding_size=0.12",
             fc='#f6f6f6', ec='#cccccc', lw=1.4))
ax.text(6.6, 5.02,
        r'class  $\mathcal{H}=\{\,h:\mathbb{R}^d\to\mathcal{U}\ \mathrm{measurable}\,\}$,'
        r'   composite kernel  $k_h(x,x{\prime})=k(h(x),h(x{\prime}))$,'
        r'   trajectory  $\mathcal{H}_T=\{h^{(0)},\dots,h^{(T)}\}\subseteq\mathcal{H}$',
        ha='center', va='center', fontsize=12.5)

# ===== Zone 3: three instantiations =====
ax.text(0.2, 3.95, '3.  Three instantiations',
        ha='left', va='center', fontsize=15, color=BLUE)
ax.text(4.55, 3.95, r'(only the envelope $L(h)$ changes)',
        ha='left', va='center', fontsize=12, color='#888888', style='italic')

cols_x = [0.35, 3.9, 7.5, 10.7]
headers = ['regime', r'class $\mathcal{H}$', r'envelope $L(h)$', r'complexity $V$']
rows = [
    ['linear / bandwidth', r'$\{\,x\mapsto x/\sigma\,\}$', r'$1/\sigma$', r'$1$'],
    ['polynomial (deg $p$)', r'$\{\,x\mapsto \Psi_p(x)/\sigma\,\}$', r'$L_{\Psi_p}/\sigma$', r'$\leq\binom{d+p}{p}$'],
    ['deep  (MLP)', r'$\{\,h_\theta:\theta\in\Theta\,\}$', r'$\prod_{j}\|W_j\|_2$', r'$\sim D$'],
]
ytop = 3.2
rh = 0.82
for cx, hd in zip(cols_x, headers):
    ax.text(cx, ytop, hd, ha='left', va='center', fontsize=12.5, color=BLUE)
ax.plot([0.3, 13.0], [ytop - 0.42, ytop - 0.42], color='#9bb0c7', lw=1.2)
for r, row in enumerate(rows):
    yr = ytop - 0.42 - (r + 0.6) * rh
    if r % 2 == 0:
        ax.add_patch(Rectangle((0.3, yr - rh / 2 + 0.04), 12.7, rh - 0.08,
                     fc='#f5f8fc', ec='none', zorder=0))
    for cx, cell in zip(cols_x, row):
        ax.text(cx, yr, cell, ha='left', va='center', fontsize=13)

fig.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
fig.savefig(os.path.join(outdir, 'proxy_overview.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'proxy_overview.png'), dpi=200, bbox_inches='tight')
print('done')
