# Operator-Fused ADMM Unfolding for Local AF Optimization

Implementation of **Accelerating Two-Dimensional Ambiguity Function Optimization via Operator-Fused ADMM Unfolding**, including the proposed solver and six local-AF comparison methods: ADMM, AISO, QGD, Consensus-ADMM, WaveNet, and MAL-Net.

[中文](README_CN.md) · [Experiment protocol](docs/EXPERIMENTS.md) · [Code map](docs/CODE_MAP.md)

## Installation

Python 3.10–3.12 and PyTorch 2.5.1 are required. Install the PyTorch build appropriate for the device, then install from the repository root:

```bash
python -m pip install -e ".[gpu,figures,test]"
```

For CPU use, omit the `gpu` extra. Verified GPU environment: Windows, Python 3.12.7, PyTorch 2.5.1+cu121, Triton-Windows 3.1.0.post17, NVIDIA RTX 3070. Linux has not been tested.

## Run

```bash
python main.py
```

The default run executes all seven methods at full iteration budgets using seed `1860028`. It automatically selects CUDA when available, independently evaluates each waveform, and exports tables and figures to a new directory under `outputs/`.

```bash
# Paper seed sets and stopping rules
python main.py --protocol paper

# Five timing repetitions per initialization
python main.py --protocol paper --repeats 5

# Selected methods with a common initialization
python main.py --methods admm consensus ours --seed 1860028

# CPU reference pipeline
python main.py --device cpu

# Short installation/pipeline check
python main.py --quick

# Explicit output directory
python main.py --output outputs/comparison
```

`--quick` uses reduced budgets and is not a convergence experiment. `--protocol paper` uses the recorded method-specific seed sets; the default example uses a common seed. Both execute the solvers and report newly measured values.

## Outputs

| File | Contents |
|---|---|
| `results.csv`, `results.json` | Per-method averages and detailed measurements |
| `table.md`, `table.tex` | Comparison table |
| `psl_time.png`, `psl_time.pdf` | PSL–time scatter and runtime bars |
| `af_before_after.png`, `af_before_after.pdf` | Expanded AF maps with the optimization region outlined |
| `<method>_<seed>_r<repeat>.npz` | Initial waveform, optimized waveform, and PSL history |
| `<method>_<seed>.json` | Configuration, measurements, stopping condition, and environment |
| `run.json` | Run configuration and completion status |


## Methods

| Method | Implementation | Default budget |
|---|---|---|
| ADMM | Fixed-schedule complex ADMM | 448 × 16 updates |
| AISO | Phase-block update with two-point acceleration | 448 × 16 updates |
| QGD | Complex-circle gradient update | 448 × 16 updates |
| Consensus-ADMM | Nine paired Doppler-slice subproblems | 448 × 16 updates |
| WaveNet | Six-layer residual network, per-instance Adam | Plateau stopping; ≤30,000 steps |
| MAL-Net | 56-stage phase update with preset steps | Plateau stopping; ≤100 passes |
| Ours | Four fixed controls, Triton K1/K2/K3, CUDA Graph | 448 × 16 updates |


## Repository

```text
main.py                         Unified experiment entry point
src/af_unfolding/
  api.py, core.py                Proposed solver and AF objective
  triton_kernels.py              Fused GPU operators
  reference.py                  Independent complex128 AF evaluator
  pipeline.py, plotting.py       Experiment orchestration and figures
  baselines/                    Six comparison implementations
  data/                         Exact solver settings
scripts/                        Focused reproduction and memory utilities
tests/                          Numerical and execution checks
data/                           Selected paper reference data
paper_figures/                  Final figures and K3 source
docs/                           Protocol, source provenance, and validation
```

## Verification

```bash
python -m pytest -q
python scripts/reproduce.py --all-seeds --repeats 5
python scripts/measure_memory.py
```

`scripts/plot_paper.py` renders saved paper data. Figures from `main.py` use the current run's outputs. Numerical verification is recorded in [VERIFICATION_CN.md](docs/VERIFICATION_CN.md).
