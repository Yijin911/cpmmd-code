# CP-MMD: NeurIPS 2026 code

Code and results for *Kernel Selection is Model Selection: A Unified Complexity-Penalized Approach for MMD Two-Sample Tests* (Yijin Ni and Xiaoming Huo, NeurIPS 2026). This folder is self-contained and reproduces the NeurIPS paper; the rest of the repository is the code release of the journal version.

## How this code differs from the journal code

- $\widehat C_1$ is calibrated on null permutations of the training half, one ratio per permutation, as in Algorithm 1 of the paper. The journal's shared-denominator calibration is not used.
- The deep CP-MMD trainers return the iterate with the largest training objective, $\widehat h = \arg\max_{0 \le t \le T} J_{\mathrm{CP}}(h^{(t)})$ (`mmd_core.BestIterate`); the Liu and plain-maximization baselines keep their final iterate.
- In the linear and polynomial experiments every method receives the same $n$ samples per class; CP-MMD calibrates and selects on one half and tests on the other.
- The Liu baseline uses the $H_1$ variance estimate of Liu et al. (2020) and of their code: $\hat\tau_h^2 = 4\,\mathrm{Var}_i(r_i) + 10^{-8}$, where $r_i$ is the $i$-th row mean of $H = K_{xx} + K_{yy} - K_{xy} - K_{yx}$ (`mmd_core.liu_tau_hat`).
- MMDAgg runs only through the pinned `mmdagg` package; the drivers no longer fall back silently to a Bonferroni test when it is missing.

## Layout

```
simulations/              experiment drivers (run_*.py) and helper modules
  mmd_core.py             MMD estimator, permutation null, MLP feature map, best-iterate tracking
  data_generator.py       synthetic data generators
  calibration_engine.py   complexity proxies and objective factories
results/                  CSV outputs behind the paper's figures and tables
figures/make_figures.py   renders the paper's figures from results/
```

## Setup

```bash
pip install -r requirements.txt
# GPU (recommended for the deep and Higgs experiments):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Tested with Python 3.11, PyTorch 2.4, and JAX 0.4.30. MMDAgg and MMD-FUSE are installed from pinned GitHub commits listed in `requirements.txt`. The deployment-cost timings run on CPU (`JAX_PLATFORMS=cpu`); the deep-kernel experiments use one GPU when available.

## Reproducing the figures and tables

Run each script from `simulations/`, e.g. `cd simulations && python run_linear_pillar.py`. Every output listed below is committed under `results/` except where the notes say otherwise.

| Paper artifact | Script | Output | Notes |
|---|---|---|---|
| Fig. 1 (class-richness sweep, §4.1) | `run_srm_width.py` | `results/srm_width/srm_width_results.csv` | MLP widths $w \in \{50,\ldots,1000\}$, $50$ reps |
| Fig. 2(a) (linear, §4.2) | `run_linear_pillar.py` | `results/linear_pillar/linear_pillar_results.csv` | multi-scale 2D mixture, $n=200$ per class, $100$ reps |
| Fig. 2(b) (polynomial, §4.2) | `run_polynomial_200reps.py` | `results/polynomial_200reps/kurtosis_results.csv` | kurtosis shift, $n=500$ per class, $200$ reps, $\sigma \in [0.01, 100]$ |
| Table 2 (deep, §4.2) | `run_deep_summary.py` (uses `run_cross_comparison.py`) | `results/deep_summary/deep_summary_results.csv` | HDGM, $\Delta=0.5$, five $(d,n)$ cells, $200$ reps; Liu rows from `run_liu_published.py` |
| Fig. 3(a) (Higgs, §4.3) | `run_higgs.py` (deep), `run_higgs_extra_gpu.py` (linear, ARD, poly2, poly3) | `results/higgs/`, `results/higgs_extra_v2/` | $d=28$, $n\in\{200,500\}$, $50$ reps; both scripts standardize the 28 features over the first $10^5$ events; the figure uses the sample-split rows (`permsplit`); Liu rows from `run_liu_published.py` |
| Fig. 3(b) (per-test cost, §4.4) | `run_deployment_cost.py` | `results/deployment_cost/deployment_cost.csv` | $30$ test pairs at each $B$; MMDAgg at $200$ and at its default $2000$ permutations |
| Selection cost and $K^*$ (§4.4) | `run_selection_cost.py` | `results/deployment_cost/selection_cost.csv` | grid calibration and selection ($n_{\mathrm{cal}}=30$, $24$ bandwidths) for linear and poly3, Adam ascent ($200$ steps) for ARD, on one CPU thread; reads `deployment_cost.csv` |
| Table 3, Fig. 4 (variance collapse, App. C) | `run_variance_collapse.py`, `run_collapse_figure.py` | `results/variance_collapse/`, `figures/collapse_trajectory.pdf` | `run_collapse_figure.py` writes its PDF directly |
| Fig. 5 ($\widehat C_1$ ablation, App. F.6) | `run_c1_ablation.py` | `results/c1_ablation/c1_ablation_results.csv` | $\widehat C_1$ swept over seven orders of magnitude |
| Tables 4–5 (stability sweeps, App. F.8) | `run_c1_stability.py` | `results/c1_stability/` | not committed; run the script first |
| Table 6 (unequal sample sizes, App. F.9) | `run_unbalanced.py` | `results/unbalanced/unbalanced_results.csv` | $(m,n) \in \{(150,150),(200,100),(100,200)\}$, $100$ reps |
| Table 7 (Liu-mixture factorial, App. J) | `run_liu_mixture_ablation.py` | `results/liu_mixture_ablation/results_final.csv` | $3\times 2$ criterion-by-kernel factorial, $100$ reps; Liu rows from `run_liu_published.py` |
| App. K (real data) | `run_tep.py`, `run_cifar.py` | `results/tep/`, `results/cifar_pca500/` | Tennessee Eastman Process; CIFAR-10 vs. CIFAR-10.1 after PCA to $500$ dimensions, $n=200$, $50$ reps; Liu rows from `run_liu_published.py` |
| App. L (search-interval sensitivity) | `run_polynomial_200reps.py` with `SIGMA_MAX=10000` | `results/polynomial_200reps/` | widens the bandwidth search interval of Fig. 2(b) |

`run_liu_published.py` reruns only the Liu baseline of Tables 2 and 7, Fig. 3(a), and App. K, reusing each driver's data, seeds, training and held-out test; `python run_liu_published.py merge` writes its rows, saved under `results/liu_published/`, into the drivers' CSVs.

`run_hdgm_revised.py` and `run_aggregation_grid.py` are exploratory drivers whose results are not reported in the paper.

After the CSVs exist, regenerate the figures with `python figures/make_figures.py`.

## Citation

```bibtex
@inproceedings{ni2026cpmmd,
  title     = {Kernel Selection is Model Selection: A Unified Complexity-Penalized Approach for MMD Two-Sample Tests},
  author    = {Ni, Yijin and Huo, Xiaoming},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, as for the rest of the repository (see `../LICENSE`).
