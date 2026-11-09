# CP-MMD: code and data

Source code, experiment scripts, and results for *A Complexity-Penalization
Framework for Data-Dependent Kernel Selection in MMD Tests* (Ni and Huo, JMLR
submission). The journal version activates the trajectory complexity proxy
$\widetilde G_T(h_\sigma)=L(h_\sigma)\,\lVert\mathbf D\rVert_F/N$ inside the
$\sigma$-selection objective and compares the **shared-denominator** and
**per-permutation** calibration procedures.

**NeurIPS 2026 paper.** The code and results for the conference paper *Kernel
Selection is Model Selection: A Unified Complexity-Penalized Approach for MMD
Two-Sample Tests* (Ni and Huo, NeurIPS 2026) are in [`neurips2026/`](neurips2026/),
which has its own README.

This repository is self-contained: it carries both the journal version's new
diagnostics and the benchmark experiments reproduced from the conference
companion (Ni and Huo, NeurIPS 2026, arXiv:2605.06883), so every empirical
claim in the paper can be traced to a script here.

## Setup

```bash
pip install -r requirements.txt
# Deep-kernel runs additionally need a CUDA build of PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Tested with Python 3.11, PyTorch 2.4 (CUDA 12.1) on an NVIDIA A100. The linear
ablation and the oracle diagnostic run on CPU; the polynomial and deep-kernel
runs are GPU-accelerated. Calibration uses seed `42`; repetition `i` uses seed
`7919*i`.

## Reproduction map

| Paper item | Script(s) | Committed data | Status |
|---|---|---|---|
| Figure 1 — squared-MMD oracle diagnostic (§5.1) | `run_oracle_gap.py` | `results/oracle_gap/srm_selection.csv` | **reproducible from committed data** |
| Table 7, Linear row — calibration ablation (§5.2) | `run_ablation_linear.py` | `results/ablation/linear.csv` | **reproducible from committed data** |
| Table 7, Polynomial row | `run_ablation_polynomial.py` | `results/ablation/polynomial.csv` | **reproducible from committed data** |
| Table 7, HDGM rows ($d{=}20$, $d{=}2$) | `run_ablation_hdgm.py`, `run_perperm_c1.py` | — | run to reproduce (GPU); per-perm and shared $\widehat C_1$ computed locally |
| Table 7, Higgs row | `run_higgs.py`, `run_perperm_c1.py` | — | run to reproduce (GPU); per-perm $\widehat C_1$ from the conference run logs |
| Table 7, TEP row | `run_tep.py`, `run_perperm_c1.py` | — | run to reproduce (GPU); per-perm $\widehat C_1$ from the conference run logs |
| Table 7, CIFAR row | `run_cifar.py` | — | run to reproduce (GPU); not run locally, per-perm $\widehat C_1$ unavailable |
| Table 5 — head-to-head benchmark (§5) | `run_cross_comparison.py`, `run_higgs.py`, `run_tep.py`, `run_hdgm_revised.py` | `results/cross_comparison_results.csv`, `results/higgs/higgs_results.csv`, `results/tep/tep_results.csv`, `results/hdgm/hdgm_revised_results.csv` | benchmark runs provided; the paper quotes the conference values |
| Per-test deployment cost (cited Fig. 3b) | `run_timing.py` | — | run to reproduce |

The remaining `run_*_pillar.py`, `run_collapse_*.py`, `run_blob_sweep.py`,
`run_killer.py`, `run_kurtosis_pillar.py`, `run_high_dn.py`,
`run_empirical_rademacher.py`, `run_pacbayes.py`, and `make_*.py` scripts
reproduce the conference companion's additional figures and diagnostics.

### What is and is not committed

- **Committed data (no rerun):** the parametric calibration ablation (linear,
  polynomial), the oracle diagnostic, and the head-to-head benchmark CSVs.
- **Run to reproduce:** the five deep-kernel ablation rows of Table 7
  (HDGM $d{=}20$/$d{=}2$, Higgs, TEP, CIFAR) are produced on GPU and are not
  stored as CSVs here; run the listed scripts to regenerate them. As stated in
  §5.2, per-permutation $\widehat C_1$ for Higgs and TEP is taken from the
  conference run logs, and CIFAR per-permutation $\widehat C_1$ is unavailable.

## Data

- **Higgs** — UCI Machine Learning Repository (auto-downloaded by `run_higgs.py`).
- **TEP** — Tennessee Eastman Process; generator in `data_generator.py`.
- **CIFAR-10 / CIFAR-10.1 v6** — publicly available (Recht et al., 2019), PCA to $d=50$.
- **Linear / polynomial / HDGM** — synthetic; generators in `data_generator.py`.

## Repository layout

```
simulations/
  mmd_core.py                 # MMD two-sample statistic and deep discriminator
  calibration_engine.py       # C_1 calibration (shared G_ref and per-permutation)
  fast_ablation.py            # GPU-aware J_CP-penalized optimizer + batched permutation test
  data_generator.py           # synthetic data generators (linear, polynomial, HDGM, TEP)
  run_oracle_gap.py           # Figure 1: squared-MMD concentration diagnostic
  run_ablation_linear.py      # Table 7: linear bandwidth ablation
  run_ablation_polynomial.py  # Table 7: polynomial (fixed p=4) ablation
  run_ablation_hdgm.py        # Table 7: HDGM deep-kernel ablation
  run_higgs.py / run_tep.py / run_cifar.py   # deep-kernel benchmarks
  run_perperm_c1.py           # per-permutation C_1 (calibration ablation)
  run_cross_comparison.py     # head-to-head method comparison
  run_timing.py               # per-test deployment cost
  ...                         # conference-companion figures/diagnostics
results/
  ablation/                   # linear.csv, polynomial.csv (calibration ablation)
  oracle_gap/                 # Figure 1 data
  hdgm/, higgs/, tep/         # head-to-head benchmark CSVs
requirements.txt
```

## Citation

```bibtex
@article{ni2026cpmmd,
  title   = {A Complexity-Penalization Framework for Data-Dependent Kernel Selection in {MMD} Tests},
  author  = {Ni, Yijin and Huo, Xiaoming},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  note    = {Submitted.}
}

@inproceedings{ni2026cpmmd-conf,
  title         = {Kernel Selection is Model Selection: A Unified Complexity-Penalized Approach for {MMD} Two-Sample Tests},
  author        = {Ni, Yijin and Huo, Xiaoming},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2605.06883},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```

## License

MIT. See `LICENSE`.
