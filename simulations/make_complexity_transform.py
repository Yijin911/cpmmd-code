"""Concept figure: global set complexity G(H)  -->  pointwise proxy G_tilde(h).

Left  : the feasible class H (a set) carries ONE scalar G(H).
Right : the trajectory proxy G_tilde(h) = L(h) ||D||_F / N is a CURVE over h,
        capped by the worst-case envelope; the gap is absorbed into C1_hat.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

plt.rcParams.update({'font.size': 14, 'mathtext.fontset': 'cm', 'font.family': 'serif'})

ORANGE = '#e6820a'
BLUE = '#08519c'

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 4.9),
                               gridspec_kw={'width_ratios': [1.0, 1.25]})

# ---------------- Left: the feasible set + trajectory ----------------
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
axL.add_patch(Ellipse((5, 6), 8.2, 6.2, fc='#eef2f7', ec='#9bb0c7', lw=1.8))
axL.text(5, 8.7, r'$\mathcal{H}$  (all candidate kernels)', ha='center', fontsize=14, color='#5a6b80')

traj = np.array([(2.6, 4.6), (3.5, 5.6), (4.5, 6.3), (5.5, 6.5), (6.4, 6.0), (7.0, 5.2)])
axL.plot(traj[:, 0], traj[:, 1], '-', color='#2f4f6f', lw=1.8, zorder=2)
axL.plot(traj[:-1, 0], traj[:-1, 1], 'o', color='#2f4f6f', ms=7, zorder=3)
axL.plot(traj[-1, 0], traj[-1, 1], 'o', color=ORANGE, ms=11, zorder=4)
axL.annotate('', xy=traj[-1], xytext=traj[-2],
             arrowprops=dict(arrowstyle='-|>', color='#2f4f6f', lw=1.8))
axL.text(traj[0, 0] - 0.2, traj[0, 1] - 0.7, r'$h^{(0)}$', ha='center', fontsize=12, color='#2f4f6f')
axL.text(traj[-1, 0] + 0.3, traj[-1, 1] - 0.7, r'$\widehat h$', ha='center', fontsize=13, color=ORANGE)
axL.text(4.6, 4.0, r'$\mathcal{H}_T$ (optimizer path)', ha='center', fontsize=12, color='#2f4f6f')
axL.text(5, 1.7, r'$G(\mathcal{H})=\mathbb{E}\,[\mathcal{G}(\mathcal{H}(\mathbf{D}))]/N$',
         ha='center', fontsize=15)
axL.text(5, 0.85, 'one scalar for the whole class  (intractable)',
         ha='center', fontsize=11.5, color='#777777', style='italic')
axL.set_title('(a) global complexity of the feasible set', fontsize=14.5, pad=8)

# ---------------- transformation arrow between panels ----------------
fig.text(0.477, 0.55, r'$\Longrightarrow$', ha='center', va='center', fontsize=26, color='#333333')
fig.text(0.477, 0.43, 'restrict\n+ envelope', ha='center', va='center',
         fontsize=9.5, color='#666666')

# ---------------- Right: pointwise proxy curve ----------------
sigma = np.logspace(np.log10(0.3), np.log10(10), 300)
c = np.sqrt(10)                       # ||D||_F / sqrt(N) for unit-variance d=10
Gt = c / sigma                        # G_tilde(sigma) = L(sigma) * geometry, L = 1/sigma
ceiling = c / 0.3                     # worst-case envelope L* ||D||_F/N at sigma_l
sig_hat = 1.7
Gt_hat = c / sig_hat

axR.fill_between(sigma, Gt, ceiling, color='#fbe3c4', alpha=0.7, zorder=1)
axR.text(3.2, ceiling * 0.62, r'gap absorbed' + '\n' + r'into $\widehat C_1$',
         ha='center', va='center', fontsize=11.5, color='#c47713')
axR.plot(sigma, Gt, color=BLUE, lw=3, zorder=3)
axR.text(0.55, 4.2, r'$\widetilde G_T(h)=L(h)\,\dfrac{\|\mathbf{D}\|_F}{N}$',
         ha='left', va='bottom', fontsize=14, color=BLUE)
axR.axhline(ceiling, ls='--', color='#888888', lw=1.8, zorder=2)
axR.text(0.32, ceiling - 0.35,
         r'global envelope  $L^\ast\|\mathbf{D}\|_F/N \geq G(\mathcal{H}_T)$',
         ha='left', va='top', fontsize=11, color='#666666')
axR.plot([sig_hat], [Gt_hat], 'o', color=ORANGE, ms=12, zorder=5)
axR.plot([sig_hat, sig_hat], [0, Gt_hat], ':', color=ORANGE, lw=1.6, zorder=2)
axR.annotate(r'$\widetilde G_T(\widehat h)$ paid here',
             xy=(sig_hat, Gt_hat), xytext=(2.6, Gt_hat + 2.4),
             fontsize=12, color=ORANGE,
             arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.6))
axR.text(sig_hat, -1.5, r'$\widehat\sigma$', ha='center', color=ORANGE, fontsize=13)

axR.set_xscale('log')
axR.set_xlim(0.3, 10); axR.set_ylim(0, ceiling * 1.08)
axR.set_xlabel(r'candidate $h$  (here: bandwidth $\sigma$)', fontsize=13)
axR.set_ylabel(r'complexity penalty', fontsize=13)
axR.set_title(r'(b) pointwise proxy: a function of the $h$ you optimize', fontsize=14.5, pad=8)
axR.spines[['top', 'right']].set_visible(False)

# per-class envelope reminder, as a footer strip
fig.text(0.5, -0.02,
         r'across classes only the envelope $L(h)$ changes:   '
         r'bandwidth $\frac{1}{\sigma}$    $\cdot$    polynomial $\frac{L_{\Psi_p}}{\sigma}$'
         r'    $\cdot$    deep $\frac{\prod_l\|W_l\|_2}{\sigma}$',
         ha='center', va='top', fontsize=12, color='#333333')

fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.subplots_adjust(wspace=0.34)
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
fig.savefig(os.path.join(outdir, 'complexity_transform.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'complexity_transform.png'), dpi=200, bbox_inches='tight')
print('done')
