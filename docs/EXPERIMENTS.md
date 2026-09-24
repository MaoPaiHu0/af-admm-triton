# Experiment protocol

## Problem and metric

One length-256 unimodular waveform is optimized over integer delays −10…10 and Doppler bins −4…4. The origin is excluded from the objective and PSL evaluation. The AF is a valid-overlap aperiodic sum, power is normalized by N², and PSL is `10 log10(max power)`. Output waveforms are independently evaluated by CPU complex128 direct sums in `reference.py`.

The AF figures cover delays −20…20 and Dopplers −8…8. The outlined region includes the origin geometrically, but the origin is excluded from optimization and scoring. No interpolation is applied to AF measurements.

## Execution protocols

`example` uses seed 1860028 for all seven methods. `--seed` changes that common seed. `paper` uses the seed sets associated with the selected paper records:

| Method | Seeds | Iteration / stopping configuration |
|---|---|---|
| ADMM, AISO, QGD | 95000…95007 | 448 outer iterations × 16 updates |
| Consensus-ADMM | 1860028 | 448 × 16; one local copy per Doppler slice |
| WaveNet | 95000…95007 | Adam; ≥2000, ≤30000 steps; ten consecutive 100-step blocks with PSL change <0.005 dB |
| MAL-Net | 95000 | 56 stages per pass; ≤100 passes; stop after 20 passes without >0.01 dB improvement |
| Ours | 1860000…1860031 | 448 × 16; exact full-precision controls and stored q schedule |

These recorded seed sets are not a shared-seed statistical comparison. Use `example` for a common initialization. The published table and a new single-initialization run are different summaries. Runtime is measured on the current machine; reference runtime values are never inserted into new measurements.

The phase generator uses a CPU float32 Torch generator. WaveNet converts phase to normalized phase and back, preserving the original experiment's rounding convention. Classical continuation traverses q=8…128 over the first 56 outer iterations and holds the final value thereafter. `--quick` executes the first two outer iterations without compressing the schedule; WaveNet uses 100 steps and MAL-Net uses two complete 56-stage passes.

## Comparison implementations

These are the local-AF adaptations used in this work, not original-author source distributions or exact reproductions of each reference's original experiment. All use the local AF operator defined above. Original methods can use different losses, constraints, or training protocols.

| Label | Released implementation and adaptation |
|---|---|
| ADMM | Fixed-schedule inexact complex ADMM with analytic smooth-PSL gradient, unit-circle projection, and scaled dual update |
| AISO | Full phase-block first-order update for a single waveform, with two-point acceleration 0.35 and the local smooth-PSL surrogate |
| QGD | Complex-circle phase gradient and retraction using the local smooth-PSL surrogate, replacing the original quartic WISL cost |
| Consensus-ADMM | Nine local states paired with nine Doppler slices; independent slice normalization and shared unit-circle consensus; compact batched gradient |
| WaveNet | Six 256×256 fully connected layers with two residual blocks; sigmoid output interpreted as phase/(2π); Adam lr=0.001 optimizes exact local PSL online |
| MAL-Net | 56 phase updates; preset steps 0.04…0.008 and orders 8…128; no pretrained weights or offline step training in this comparison |

MAL-Net retains the recorded factor-two phase update. With the real-Euclidean complex gradient convention used by the objective, this factor is part of the effective step-size setting. It is not a different identity for the phase derivative.

ADMM, AISO, QGD, Consensus-ADMM, WaveNet, and Ours retain the best feasible waveform including initialization. MAL-Net retains the best checkpoint satisfying its >0.01 dB update threshold. Smooth objective values are not substituted for final PSL. Saved update counts exclude metric-only evaluations.

WaveNet initialization uses seed `phase_seed + 100000`. Its model, optimizer, and recurrent inputs reset before every measured repetition. Ours resets its captured state on every call. CUDA Graph is part of the released GPU execution path for Ours and WaveNet.

## Timing and memory

Each method/seed runs in a fresh process. Setup and warmup precede measurement. Wall time covers a synchronized device-resident solve, including iteration control and stopping checks. CUDA-event time records the corresponding device timeline. Setup, compilation, capture, state reset, host transfers, independent scoring and file output are excluded.

Per-run JSON records process peak allocated tensor memory after setup/warmup and a peak-statistics reset. This includes resident solver state, graph pools, and result histories; it is not an asymptotic storage estimate. `scripts/measure_memory.py` reproduces the proposed solver's dedicated memory measurement. CPU/GPU paths can diverge after many nonlinear iterations; exact trajectory equality is only expected for a pinned arithmetic environment.

## References

Bibliographic identifiers are copied from the final manuscript. They identify the methods behind the adaptations; original-author code is not bundled.

| Method | Reference |
|---|---|
| ADMM | Liang et al., “Unimodular Sequence Design Based on Alternating Direction Method of Multipliers,” IEEE TSP, 2016. DOI: 10.1109/TSP.2016.2597123 |
| AISO | Cui et al., “Local Ambiguity Function Shaping via Unimodular Sequence Design,” IEEE SPL, 2017. DOI: 10.1109/LSP.2017.2700396 |
| QGD | Alhujaili et al., “Quartic Gradient Descent for Tractable Radar Slow-Time Ambiguity Function Shaping,” IEEE TAES, 2020. DOI: 10.1109/TAES.2019.2934336 |
| Consensus-ADMM | Wang and Wang, “Designing Unimodular Sequences With Optimized Auto/Cross-Correlation Properties via Consensus-ADMM/PDMM Approaches,” IEEE TSP, 2021. DOI: 10.1109/TSP.2021.3079819 |
| WaveNet | Shi et al., “An Optimized Neural Network Framework for Designing Spectrally Compatible Radar Waveforms,” IEEE TCCN, 2025. DOI: 10.1109/TCCN.2024.3488815 |
| MAL-Net | Wang et al., “MAL-Net: Model-Adaptive Learned Network for Slow-Time Ambiguity Function Shaping,” Remote Sensing, 2025. DOI: 10.3390/rs17010173 |
