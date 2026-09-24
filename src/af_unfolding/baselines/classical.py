"""Selected local-AF adaptations: ADMM, AISO, QGD and paired Consensus-ADMM.

Recurrences and numerical operation ordering follow the paper experiments.
execution_layers runs a prefix without rescaling the continuation schedule.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from ..core import LocalAmbiguityObjective
from .consensus import PackedDopplerGradient
Tensor = torch.Tensor

@dataclass(frozen=True)
class AFBaselineConfig:
    """Common budget and continuation schedule for all baselines."""
    outer_layers: int = 56
    inner_updates: int = 16
    initial_peak_order: float = 8.0
    final_peak_order: float = 128.0
    initial_rho: float = 0.2
    final_rho: float = 0.4
    initial_step: float = 0.08
    final_step: float = 0.03
    dual_scale: float = 1.0
    select_best: bool = True
    schedule_layers: int | None = None

    @property
    def physical_updates(self) -> int:
        return self.outer_layers * self.inner_updates

@dataclass(frozen=True)
class AISOConfig(AFBaselineConfig):
    """AF-PSL adaptation of accelerated iterative sequential optimization."""
    block_size: int = 256
    acceleration: float = 0.35
    step_multiplier: float = 1.0

@dataclass(frozen=True)
class QGDConfig(AFBaselineConfig):
    """AF-PSL adaptation of quartic/manifold gradient descent."""
    initial_step: float = 0.2
    final_step: float = 0.04
    momentum: float = 0.0
    gradient_normalization: bool = False

@dataclass(frozen=True)
class PlainADMMConfig(AFBaselineConfig):
    """Fixed-schedule, matrix-free plain ADMM."""
    pass

@dataclass(frozen=True)
class ConsensusADMMConfig(AFBaselineConfig):
    """Consensus-ADMM adaptation with one local copy per Doppler slice."""
    consensus_blocks: int = 9
    local_rho_multiplier: float = 1.0
    local_step_multiplier: float = 0.75

@dataclass
class AFBaselineResult:
    """Common result object returned by every baseline."""
    method: str
    phase: Tensor
    waveform: Tensor
    initial_local_af_peak_power: Tensor
    local_af_peak_power: Tensor
    wisl_power: Tensor
    objective_history: Tensor
    smooth_history: Tensor
    runtime_seconds: float = 0.0
    operator_evaluations: int = 0
    notes: str = ''

    @property
    def local_af_psl_db(self) -> Tensor:
        return 10.0 * torch.log10(self.local_af_peak_power.clamp_min(1e-30))

    @property
    def initial_local_af_psl_db(self) -> Tensor:
        return 10.0 * torch.log10(self.initial_local_af_peak_power.clamp_min(1e-30))

    @property
    def wisl_db(self) -> Tensor:
        return 10.0 * torch.log10(self.wisl_power.clamp_min(1e-30))

def phase_to_waveform(phase: Tensor) -> Tensor:
    """Convert ``[B,N,1]`` phase to a complex waveform."""
    if phase.ndim == 2:
        phase = phase.unsqueeze(0)
    return torch.complex(torch.cos(phase), torch.sin(phase))

def _schedule(start: float, stop: float, fraction: float) -> float:
    """Geometric interpolation for positive algorithm parameters."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    if start <= 0.0 or stop <= 0.0:
        return (1.0 - fraction) * start + fraction * stop
    return float(start * (stop / start) ** fraction)

def _peak_order(config: AFBaselineConfig, outer: int) -> float:
    horizon = config.outer_layers if config.schedule_layers is None else min(max(int(config.schedule_layers), 1), config.outer_layers)
    clipped_outer = min(int(outer), horizon - 1)
    denominator = max(horizon - 1, 1)
    return _schedule(config.initial_peak_order, config.final_peak_order, clipped_outer / denominator)

def _schedule_fraction(config: AFBaselineConfig, outer: int) -> float:
    """Continuation fraction with an optional held-final tail."""
    horizon = config.outer_layers if config.schedule_layers is None else min(max(int(config.schedule_layers), 1), config.outer_layers)
    return min(int(outer), horizon - 1) / max(horizon - 1, 1)

def _dense_af_power(objective: LocalAmbiguityObjective, sequence: Tensor) -> tuple[Tensor, Tensor]:
    """Return dense AF and normalized power for a complex sequence."""
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)
    reference = sequence[:, objective.dense_reference_index, :]
    shifted = sequence[:, objective.dense_shifted_index, :]
    overlap = objective.dense_overlap_mask[None, :, :, None]
    product = shifted * reference.conj() * overlap
    steering = objective._dense_steering(sequence)
    ambiguity = torch.einsum('lkn,bknm->blkm', steering, product)
    power = ambiguity.abs().square() / float(objective.length * objective.length)
    return (ambiguity, power)

def weighted_af_terms_and_gradient(objective: LocalAmbiguityObjective, sequence: Tensor, peak_order: float | Tensor, *, cell_mask: Tensor | None=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate a high-order local-PSL surrogate and its complex gradient.

    ``cell_mask`` is an optional ``[L,K]`` mask used by consensus ADMM.  The
    derivative is the analytic derivative of

    ``(1/q) log sum_(l,k in Omega) p_lk**q``.

    The returned tuple is ``(smooth_log_peak, exact_peak, gradient, power)``.
    The gradient follows PyTorch's complex-real convention and can therefore
    be used directly in a complex gradient step ``z -= step * gradient``.
    """
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)
    ambiguity, power = _dense_af_power(objective, sequence)
    tiny = torch.finfo(power.dtype).tiny
    valid = objective.keep_mask.to(device=sequence.device)
    if cell_mask is not None:
        valid = valid & cell_mask.to(device=sequence.device, dtype=torch.bool)
    valid_b = valid[None, :, :, None]
    logits = float(peak_order) * power.clamp_min(tiny).log()
    logits = logits.masked_fill(~valid_b, -torch.inf)
    smooth = torch.logsumexp(logits.flatten(1), dim=1) / float(peak_order)
    exact = power.masked_fill(~valid_b, -torch.inf).flatten(1).amax(dim=1)
    weights = torch.softmax(logits.flatten(1), dim=1).reshape_as(power)
    ambiguity_gradient = 2.0 * weights * ambiguity / (float(objective.length * objective.length) * power.clamp_min(tiny))
    shifted_steering, reference_steering = objective._gradient_steering(sequence)
    reference_for_shifted = sequence[:, objective.gradient_reference_for_shifted_index, :] * objective.gradient_shifted_valid[None, :, :, None]
    shifted_for_reference = sequence[:, objective.gradient_shifted_for_reference_index, :] * objective.gradient_reference_valid[None, :, :, None]
    expanded = ambiguity_gradient[:, :, :, None, :]
    shifted_contribution = expanded * reference_for_shifted[:, None, :, :, :] * shifted_steering[None, :, :, :, None].conj()
    reference_contribution = expanded.conj() * shifted_for_reference[:, None, :, :, :] * reference_steering[None, :, :, :, None]
    gradient = (shifted_contribution + reference_contribution).sum(dim=(1, 2))
    return (smooth, exact, gradient, power)

@torch.no_grad()
def exact_local_metrics(objective: LocalAmbiguityObjective, waveform: Tensor) -> tuple[Tensor, Tensor]:
    """Return exact normalized PSL power and WISL power."""
    power = objective.normalized_power_samples_waveform(waveform)
    return (power.amax(dim=1), power.mean(dim=1))

def _make_result(method: str, objective: LocalAmbiguityObjective, initial_waveform: Tensor, final_waveform: Tensor, objective_history: Iterable[Tensor], smooth_history: Iterable[Tensor], *, runtime_seconds: float=0.0, operator_evaluations: int=0, notes: str='') -> AFBaselineResult:
    with torch.no_grad():
        initial_peak, _ = exact_local_metrics(objective, initial_waveform)
        final_peak, wisl = exact_local_metrics(objective, final_waveform)
        phase = torch.remainder(torch.angle(final_waveform), 2.0 * torch.pi)
    return AFBaselineResult(method=method, phase=phase, waveform=final_waveform, initial_local_af_peak_power=initial_peak, local_af_peak_power=final_peak, wisl_power=wisl, objective_history=torch.stack(tuple(objective_history), dim=1), smooth_history=torch.stack(tuple(smooth_history), dim=1), runtime_seconds=runtime_seconds, operator_evaluations=operator_evaluations, notes=notes)

def _initial_state(initial_phase: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if initial_phase.ndim == 2:
        initial_phase = initial_phase.unsqueeze(0)
    waveform = phase_to_waveform(initial_phase)
    return (initial_phase.detach().clone(), waveform.detach().clone(), waveform.detach().clone())

def solve_af_plain_admm(objective: LocalAmbiguityObjective, initial_phase: Tensor, config: PlainADMMConfig | None=None, *, execution_layers=None) -> AFBaselineResult:
    """Solve the AF problem with fixed, matrix-free plain ADMM updates."""
    config = config or PlainADMMConfig()
    _, initial_waveform, z = _initial_state(initial_phase)
    x = z.clone()
    dual = torch.zeros_like(z)
    best_peak, _ = exact_local_metrics(objective, x)
    best_x = x.clone()
    history: list[Tensor] = [best_peak.detach()]
    smooth_history: list[Tensor] = []
    evaluations = 0
    with torch.no_grad():
        for outer in range(config.outer_layers if execution_layers is None else execution_layers):
            fraction = _schedule_fraction(config, outer)
            rho = _schedule(config.initial_rho, config.final_rho, fraction)
            step = _schedule(config.initial_step, config.final_step, fraction)
            q = _peak_order(config, outer)
            for _ in range(config.inner_updates):
                smooth, _, gradient, _ = weighted_af_terms_and_gradient(objective, z, q)
                evaluations += 1
                z = z - step * (gradient + rho * (z - x + dual))
                smooth_history.append(smooth.detach())
            projected = z + dual
            x = projected / projected.abs().clamp_min(1e-12)
            dual = dual + config.dual_scale * (z - x)
            peak, _ = exact_local_metrics(objective, x)
            evaluations += 1
            history.append(peak.detach())
            if config.select_best:
                improve = peak < best_peak
                best_peak = torch.where(improve, peak, best_peak)
                best_x = torch.where(improve[:, None, None], x, best_x)
    output = best_x if config.select_best else x
    return _make_result('AF-Plain-ADMM (PSL-adapted, analytic Torch)', objective, initial_waveform, output, history, smooth_history, operator_evaluations=evaluations, notes='Fixed-schedule matrix-free ADMM; high-order smooth-PSL surrogate and exact PSL scoring.')

def solve_af_qgd(objective: LocalAmbiguityObjective, initial_phase: Tensor, config: QGDConfig | None=None, *, execution_layers=None) -> AFBaselineResult:
    """Direct complex-circle (phase-manifold) gradient descent.

    QGD in the cited TAES paper uses the exact gradient of a quartic WISL
    cost.  Here the same retraction is applied to the high-order PSL
    continuation, so the final comparison is aligned with the requested PSL
    target.
    """
    config = config or QGDConfig()
    _, initial_waveform, waveform = _initial_state(initial_phase)
    phase = torch.remainder(torch.angle(waveform), 2.0 * torch.pi)
    velocity = torch.zeros_like(phase)
    best_peak, _ = exact_local_metrics(objective, waveform)
    best_waveform = waveform.clone()
    history: list[Tensor] = [best_peak.detach()]
    smooth_history: list[Tensor] = []
    evaluations = 0
    with torch.no_grad():
        for outer in range(config.outer_layers if execution_layers is None else execution_layers):
            fraction = _schedule_fraction(config, outer)
            step = _schedule(config.initial_step, config.final_step, fraction)
            q = _peak_order(config, outer)
            for _ in range(config.inner_updates):
                sequence = phase_to_waveform(phase)
                smooth, _, gradient, _ = weighted_af_terms_and_gradient(objective, sequence, q)
                evaluations += 1
                phase_gradient = (sequence.conj() * gradient).imag
                if config.gradient_normalization:
                    scale = phase_gradient.square().mean(dim=(1, 2), keepdim=True).sqrt()
                    phase_gradient = phase_gradient / scale.clamp_min(1e-08)
                velocity = config.momentum * velocity + phase_gradient
                phase = torch.remainder(phase - step * velocity, 2.0 * torch.pi)
                smooth_history.append(smooth.detach())
            waveform = phase_to_waveform(phase)
            peak, _ = exact_local_metrics(objective, waveform)
            evaluations += 1
            history.append(peak.detach())
            improve = peak < best_peak
            best_peak = torch.where(improve, peak, best_peak)
            best_waveform = torch.where(improve[:, None, None], waveform, best_waveform)
    return _make_result('AF-QGD (PSL-adapted complex-circle GD)', objective, initial_waveform, best_waveform, history, smooth_history, operator_evaluations=evaluations, notes='QGD-style phase-manifold retraction with the exact gradient of a high-order PSL continuation; the original quartic cost is not used as the reported metric.')

def solve_af_aiso(objective: LocalAmbiguityObjective, initial_phase: Tensor, config: AISOConfig | None=None, *, execution_layers=None) -> AFBaselineResult:
    """Sequential MM/IRLS-style AF-PSL optimization.

    AISO's published update is a sequential MM step with a two-point
    acceleration policy.  The local AF operator in this repository supports a
    general Doppler-delay mask, so we implement its scalable block version:
    each update solves a phase-projected MM surrogate on one contiguous block,
    while a secant term carries the two-point acceleration between visits.
    """
    config = config or AISOConfig()
    _, initial_waveform, waveform = _initial_state(initial_phase)
    phase = torch.remainder(torch.angle(waveform), 2.0 * torch.pi)
    n = phase.shape[1]
    block_size = max(1, min(int(config.block_size), n))
    previous_delta = torch.zeros_like(phase)
    best_peak, _ = exact_local_metrics(objective, waveform)
    best_waveform = waveform.clone()
    history: list[Tensor] = [best_peak.detach()]
    smooth_history: list[Tensor] = []
    evaluations = 0
    update_index = 0
    with torch.no_grad():
        for outer in range(config.outer_layers if execution_layers is None else execution_layers):
            fraction = _schedule_fraction(config, outer)
            base_step = _schedule(config.initial_step, config.final_step, fraction)
            base_step *= config.step_multiplier
            q = _peak_order(config, outer)
            for _ in range(config.inner_updates):
                waveform = phase_to_waveform(phase)
                smooth, _, gradient, _ = weighted_af_terms_and_gradient(objective, waveform, q)
                evaluations += 1
                phase_gradient = (waveform.conj() * gradient).imag
                start = update_index * block_size % n
                indices = torch.arange(start, start + block_size, device=phase.device) % n
                delta = torch.zeros_like(phase)
                delta[:, indices, :] = -base_step * phase_gradient[:, indices, :]
                delta[:, indices, :] += config.acceleration * (delta[:, indices, :] - previous_delta[:, indices, :])
                phase = torch.remainder(phase + delta, 2.0 * torch.pi)
                previous_delta = delta
                smooth_history.append(smooth.detach())
                update_index += 1
            waveform = phase_to_waveform(phase)
            peak, _ = exact_local_metrics(objective, waveform)
            evaluations += 1
            history.append(peak.detach())
            improve = peak < best_peak
            best_peak = torch.where(improve, peak, best_peak)
            best_waveform = torch.where(improve[:, None, None], waveform, best_waveform)
    return _make_result('AF-AISO (PSL-adapted sequential MM/IRLS)', objective, initial_waveform, best_waveform, history, smooth_history, operator_evaluations=evaluations, notes='Sequential block MM/IRLS update with AISO-style two-point secant acceleration; exact local PSL is the primary score.')

def _doppler_slice_masks(objective: LocalAmbiguityObjective, blocks: int) -> Tensor:
    """Construct approximately equal Doppler-slice masks."""
    count = objective.keep_mask.shape[0]
    blocks = max(1, min(int(blocks), count))
    masks = torch.zeros((blocks, count, objective.keep_mask.shape[1]), dtype=torch.bool, device=objective.keep_mask.device)
    for block, indices in enumerate(torch.tensor_split(torch.arange(count), blocks)):
        masks[block, indices.to(masks.device)] = objective.keep_mask[indices.to(masks.device)]
    valid_counts = masks.flatten(1).sum(dim=1)
    for block in range(blocks):
        if int(valid_counts[block].item()) == 0:
            masks[block] = objective.keep_mask
    return masks

def solve_af_consensus_admm(objective: LocalAmbiguityObjective, initial_phase: Tensor, config: ConsensusADMMConfig | None=None, *, execution_layers=None, gradient_operator=None) -> AFBaselineResult:
    """Consensus-ADMM with one local AF copy per Doppler slice."""
    config = config or ConsensusADMMConfig()
    _, initial_waveform, common = _initial_state(initial_phase)
    phase_shape = common.shape
    masks = _doppler_slice_masks(objective, config.consensus_blocks)
    blocks = masks.shape[0]
    local = common.expand(blocks, -1, -1).clone()
    if gradient_operator is None:
        gradient_operator = PackedDopplerGradient(objective, local)
    local_dual = torch.zeros_like(local)
    consensus = common.clone()
    best_peak, _ = exact_local_metrics(objective, consensus)
    best_waveform = consensus.clone()
    history: list[Tensor] = [best_peak.detach()]
    smooth_history: list[Tensor] = []
    evaluations = 0
    with torch.no_grad():
        for outer in range(config.outer_layers if execution_layers is None else execution_layers):
            fraction = _schedule_fraction(config, outer)
            rho = _schedule(config.initial_rho, config.final_rho, fraction)
            rho *= config.local_rho_multiplier
            step = _schedule(config.initial_step, config.final_step, fraction)
            step *= config.local_step_multiplier
            q = _peak_order(config, outer)
            for _ in range(config.inner_updates):
                gradients = gradient_operator(local, q)
                evaluations += 1
                local = local - step * (gradients + rho * (local - consensus + local_dual))
                consensus_argument = (local + local_dual).mean(dim=0, keepdim=True)
                consensus = consensus_argument / consensus_argument.abs().clamp_min(1e-12)
                local_dual = local_dual + (local - consensus)
                smooth_history.append(torch.zeros(1, device=consensus.device))
            peak, _ = exact_local_metrics(objective, consensus)
            evaluations += 1
            history.append(peak.detach())
            improve = peak < best_peak
            best_peak = torch.where(improve, peak, best_peak)
            best_waveform = torch.where(improve[:, None, None], consensus, best_waveform)
    return _make_result('AF-Consensus-ADMM (PSL-adapted Doppler consensus)', objective, initial_waveform, best_waveform, history, smooth_history, operator_evaluations=evaluations, notes=f'{blocks} local AF copies partitioned by Doppler slices; matrix-free consensus and exact PSL scoring.')
