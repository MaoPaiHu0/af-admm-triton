# Final manuscript to code map

The source receipt is `source_manifest.json`. It pins the current final manuscript and the implementation actually used by its selected results. Older algorithms and experiment bundles are not included.

| Manuscript component | Released code | What is checked |
|---|---|---|
| Aperiodic AF, valid-overlap sum | `LocalAmbiguityObjective.ambiguity_waveform`, `reference.ambiguity_numpy` | Positive/negative delay, boundary overlap, Doppler phase sign, independent complex128 sum |
| Normalized power and local PSL | `waveform_terms`, `reference.psl_numpy` | Division by N², origin exclusion, 10log10(power) |
| Log-domain high-order surrogate | `dense_waveform_terms_and_gradient` | Independent direct-sum autograd and finite differences |
| Inexact ADMM inner recurrence | `OperatorAwarePureLocalAFADMMNet.forward` | Independent recurrence; fixed x/u inside inner loop; momentum reset outside it |
| K1 | `_local_af_correlation_kernel` | Correlations compared to independent CPU complex128 sums |
| K2 | `_local_af_softmax_gradient_kernel` | Its coefficients feed K3 checked against independent objective derivatives |
| K3 | `_local_af_gradient_momentum_update_kernel` | Gradient, coupling, candidate, momentum and update; distinct input/output buffers |
| Projection and dual update | `_local_af_projection_dual_kernel`, `_repair_tiny_projection` | Unit modulus and scaled dual recursion; zero and tiny-value boundary |
| Fixed replay | `CapturedPureLocalAFADMM` | Eager/replay equality; A-B-A inputs restore state; repeated outputs reuse storage |
| Exact selected controls | `api.build_solver`, `data/paper_final.json` | Full-precision scalar encoding; 32 tied storage slots representing four independent controls |
| Mean PSL and improvement | `scripts/reproduce.py`, `data/paper_reference_results.csv` | All32 final selected initializations; independent output scoring |
| Memory | `scripts/measure_memory.py` | Fresh process, allocated vs reserved separation |
| Current figures | `scripts/plot_paper.py`, `paper_figures/` | Original values preserved; interpolation is display-only; text bounds inspected |
| Seven-method pipeline | `main.py`, `pipeline.py`, `plotting.py` | Fresh-process solves, independent scores, tables and figures from current outputs |
| ADMM / AISO / QGD | `baselines/classical.py` | Selected schedules; independent smooth-PSL gradient; short and full-run regression |
| Consensus-ADMM | `baselines/classical.py`, `baselines/consensus.py` | Paired Doppler slices; separate normalization; independent subproblem gradients |
| WaveNet | `baselines/networks.py`, `baselines/online.py` | Six-layer architecture; online Adam; graph reset and recorded stopping |
| MAL-Net | `baselines/networks.py`, `baselines/online.py` | Preset 56-stage steps; repeated inference; recorded selection and stopping |

`core.py` retains six selected classes from the source module. Type references to excluded research classes were adapted; the default configuration is imposed explicitly in `api.py`. The frozen controls overwrite all stage slots, so a previous trained checkpoint is unnecessary.

`triton_kernels.py` retains the currently used source module and its numerical helper interfaces. The public API selects fused momentum, projection mode3, and no diagonal preconditioner. No alternative experiment configurations are provided.

The numerical boundary repair is the only intentional numerical implementation change from the selected source. A separate masked GPU guard is used because modifying the main projection expression changed compiler arithmetic and accumulated along the long trajectory. Normal-range values are never written by the guard. All32 selected outputs are rechecked after this final change.
