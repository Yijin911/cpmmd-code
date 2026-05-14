# CP-MMD: code release

Source code and experiment scripts for *A Complexity-Penalization Framework for Data-Dependent Kernel Selection in MMD Tests* (Ni and Huo, JMLR submission).

This repository accompanies the journal version. The experimental setup follows the conference companion of Ni and Huo (NeurIPS 2026, arXiv:2605.06883); the journal version activates the trajectory complexity proxy $\widetilde G_T(h_\sigma) = L(h_\sigma)\,\|\mathbf D\|_F/N$ inside the $\sigma$-selection objective and compares the shared-denominator and per-permutation calibration procedures.

## Setup

```bash
pip install -r requirements.txt
# GPU (optional; required for the deep-kernel runs):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Tested with Python 3.11, PyTorch 2.4 (CUDA 12.1) on an NVIDIA A100. CPU runs work for the linear ablation; the polynomial and deep-kernel runs are GPU-accelerated.

## Reproducing the §5 results

| Paper artefact | Script | Output |
|---|---|---|
| §5.1 squared-MMD diagnostic (Fig. 1) | `simulations/run_oracle_gap.py` | `figures/oracle_gap.pdf` |
| §5.2 linear bandwidth ablation | `simulations/run_ablation_linear.py` | `results/ablation/linear.csv` |
| §5.2 polynomial fixed-$p$ ablation | `simulations/run_ablation_polynomial.py` | `results/ablation/polynomial.csv` |

The deep-kernel rows of Table 4 (HDGM, Higgs, TEP, CIFAR) follow the conference companion's experimental scripts and run logs; see `simulations/calibration_engine.py` for the shared-denominator $\widehat C_1$ calibration used in the journal version.

### Data

- **Higgs** — UCI Machine Learning Repository (auto-downloaded by `run_higgs.py` in the conference companion).
- **TEP** — synthetic; standard Tennessee Eastman Process model.
- **CIFAR-10/10.1 v6** — publicly available from Recht et al. (2019).
- **Linear / polynomial / HDGM** — synthetic; generators included in the experiment scripts.

## Repository layout

```
simulations/
  fast_ablation.py            # GPU-aware J_CP-penalized bandwidth optimizer
                              # and batched permutation test
  mmd_core.py                 # MMD U-statistic and deep discriminator
  calibration_engine.py       # Ĉ_1 calibration with shared G_ref
  run_oracle_gap.py           # §5.1 squared-MMD concentration check
  run_ablation_linear.py      # §5.2 linear bandwidth ablation
  run_ablation_polynomial.py  # §5.2 polynomial (fixed p=4) ablation
results/
  ablation/                   # CSV outputs (linear, polynomial)
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

@inproceedings{ni2026cpmmd-neurips,
  title     = {Kernel Selection is Model Selection: A Unified Complexity-Penalised Approach for {MMD} Two-Sample Tests},
  author    = {Ni, Yijin and Huo, Xiaoming},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
