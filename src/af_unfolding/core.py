"""Current local AF objective, fixed-control solver, and graph replay.

Numerical class bodies are extracted from the final experiment implementation.
Unrelated classes/imports and type references were removed. Unit-circle
projection now defines the zero input as 1+0j and normalizes tiny nonzero inputs.
Use build_solver() to obtain the exact final-paper configuration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from .triton_kernels import TritonLocalAFUpdate

@dataclass(frozen=True)
class LocalAFWaveformDesign:
    """Minimal design descriptor that permits one or more AF waveforms."""

    length: int = 256
    num_sequences: int = 1

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("length must be at least two")
        if self.num_sequences < 1:
            raise ValueError("num_sequences must be positive")


@dataclass(frozen=True)
class LocalAFRegion:
    """A Cartesian Doppler-delay region on the paper's integer grid."""

    doppler_bins: tuple[float, ...]
    delay_bins: tuple[int, ...]
    exclude_origin: bool = True

    def __post_init__(self) -> None:
        if not self.doppler_bins or not self.delay_bins:
            raise ValueError("the local-AF region must be nonempty")
        if len(set(self.doppler_bins)) != len(self.doppler_bins):
            raise ValueError("doppler bins must be unique")
        if len(set(self.delay_bins)) != len(self.delay_bins):
            raise ValueError("delay bins must be unique")

    @classmethod
    def paper_exp2(cls) -> "LocalAFRegion":
        return cls(
            tuple(float(value) for value in range(-4, 5)),
            tuple(range(-10, 11)),
            exclude_origin=True,
        )


class LocalAmbiguityObjective(nn.Module):
    """Batchable, autograd-compatible aperiodic ambiguity operator."""

    def __init__(
        self,
        length: int,
        region: LocalAFRegion,
        *,
        peak_order: float = 16.0,
        implementation: str = "loop",
    ) -> None:
        super().__init__()
        if length < 2:
            raise ValueError("length must be at least two")
        if peak_order <= 1.0:
            raise ValueError("peak_order must exceed one")
        if any(abs(delay) >= length for delay in region.delay_bins):
            raise ValueError("every delay must satisfy |k| < length")
        self.length = int(length)
        self.region = region
        self.peak_order = float(peak_order)
        if implementation not in {"loop", "dense"}:
            raise ValueError("implementation must be 'loop' or 'dense'")
        self.implementation = implementation
        self.register_buffer(
            "doppler_bins",
            torch.tensor(region.doppler_bins, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "delay_bins",
            torch.tensor(region.delay_bins, dtype=torch.long),
            persistent=False,
        )
        keep = torch.ones(
            (len(region.doppler_bins), len(region.delay_bins)),
            dtype=torch.bool,
        )
        if region.exclude_origin:
            doppler_zero = torch.tensor(region.doppler_bins) == 0
            delay_zero = torch.tensor(region.delay_bins) == 0
            keep &= ~(doppler_zero[:, None] & delay_zero[None, :])
        if not bool(keep.any()):
            raise ValueError("excluding the origin leaves an empty region")
        self.register_buffer("keep_mask", keep, persistent=False)
        reference = torch.zeros(
            (len(region.delay_bins), length), dtype=torch.long
        )
        shifted = torch.zeros_like(reference)
        overlap = torch.zeros_like(reference, dtype=torch.bool)
        for delay_index, delay in enumerate(region.delay_bins):
            start = max(0, -delay)
            stop = min(length, length - delay)
            indices = torch.arange(start, stop)
            count = indices.numel()
            reference[delay_index, :count] = indices
            shifted[delay_index, :count] = indices + delay
            overlap[delay_index, :count] = True
        self.register_buffer("dense_reference_index", reference, persistent=False)
        self.register_buffer("dense_shifted_index", shifted, persistent=False)
        self.register_buffer("dense_overlap_mask", overlap, persistent=False)
        reference_positions = reference.to(dtype=torch.float64)
        angle = (
            -2.0
            * torch.pi
            * self.doppler_bins[:, None, None]
            * reference_positions[None, :, :]
            / float(self.length)
        )
        steering = torch.complex(torch.cos(angle), torch.sin(angle))
        steering = steering * overlap[None, :, :]
        self.register_buffer(
            "dense_steering_real", steering.real, persistent=False
        )
        self.register_buffer(
            "dense_steering_imag", steering.imag, persistent=False
        )
        output_position = torch.arange(length)[None, :]
        delay_column = self.delay_bins[:, None]
        reference_for_shifted = output_position - delay_column
        shifted_for_reference = output_position + delay_column
        shifted_valid = (reference_for_shifted >= 0) & (
            reference_for_shifted < length
        )
        reference_valid = (shifted_for_reference >= 0) & (
            shifted_for_reference < length
        )
        self.register_buffer(
            "gradient_reference_for_shifted_index",
            reference_for_shifted.clamp(0, length - 1),
            persistent=False,
        )
        self.register_buffer(
            "gradient_shifted_for_reference_index",
            shifted_for_reference.clamp(0, length - 1),
            persistent=False,
        )
        self.register_buffer(
            "gradient_shifted_valid", shifted_valid, persistent=False
        )
        self.register_buffer(
            "gradient_reference_valid", reference_valid, persistent=False
        )
        shifted_angle = (
            -2.0
            * torch.pi
            * self.doppler_bins[:, None, None]
            * reference_for_shifted.clamp(0, length - 1).to(torch.float64)[
                None, :, :
            ]
            / float(length)
        )
        reference_angle = (
            -2.0
            * torch.pi
            * self.doppler_bins[:, None, None]
            * output_position.to(torch.float64)[None, :, :]
            / float(length)
        )
        shifted_steering = torch.complex(
            torch.cos(shifted_angle), torch.sin(shifted_angle)
        ) * shifted_valid[None, :, :]
        reference_steering = torch.complex(
            torch.cos(reference_angle), torch.sin(reference_angle)
        ) * reference_valid[None, :, :]
        self.register_buffer(
            "gradient_shifted_steering_real",
            shifted_steering.real,
            persistent=False,
        )
        self.register_buffer(
            "gradient_shifted_steering_imag",
            shifted_steering.imag,
            persistent=False,
        )
        self.register_buffer(
            "gradient_reference_steering_real",
            reference_steering.real,
            persistent=False,
        )
        self.register_buffer(
            "gradient_reference_steering_imag",
            reference_steering.imag,
            persistent=False,
        )

    def _dense_steering(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.dtype not in {torch.complex64, torch.complex128}:
            raise TypeError("waveform must use complex64 or complex128")
        real_dtype = sequence.real.dtype
        return torch.complex(
            self.dense_steering_real.to(dtype=real_dtype),
            self.dense_steering_imag.to(dtype=real_dtype),
        )

    def _gradient_steering(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_dtype = sequence.real.dtype
        shifted = torch.complex(
            self.gradient_shifted_steering_real.to(dtype=real_dtype),
            self.gradient_shifted_steering_imag.to(dtype=real_dtype),
        )
        reference = torch.complex(
            self.gradient_reference_steering_real.to(dtype=real_dtype),
            self.gradient_reference_steering_imag.to(dtype=real_dtype),
        )
        return shifted, reference

    def _validate(self, phase: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeezed = phase.ndim == 2
        if squeezed:
            phase = phase.unsqueeze(0)
        if phase.ndim != 3 or phase.shape[1] != self.length:
            raise ValueError("phase must have shape [N,M] or [B,N,M]")
        if not torch.is_floating_point(phase):
            raise TypeError("phase must be floating point")
        return phase, squeezed

    def ambiguity_waveform(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return ``r[l,k]`` for a possibly non-unimodular complex waveform."""

        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3 or sequence.shape[1] != self.length:
            raise ValueError("waveform must have shape [N,M] or [B,N,M]")
        if not torch.is_complex(sequence):
            raise TypeError("waveform must be complex")
        if self.implementation == "dense":
            reference = sequence[:, self.dense_reference_index, :]
            shifted = sequence[:, self.dense_shifted_index, :]
            product = shifted * reference.conj()
            product = product * self.dense_overlap_mask[None, :, :, None]
            steering = self._dense_steering(sequence)
            return torch.einsum("lkn,bknm->blkm", steering, product)
        real_dtype = sequence.real.dtype
        doppler = self.doppler_bins.to(device=sequence.device, dtype=real_dtype)
        values = []
        for delay_value in self.delay_bins.tolist():
            delay = int(delay_value)
            start = max(0, -delay)
            stop = min(self.length, self.length - delay)
            reference_index = torch.arange(start, stop, device=sequence.device)
            shifted_index = reference_index + delay
            product = (
                sequence[:, shifted_index, :]
                * sequence[:, reference_index, :].conj()
            )
            angle = (
                -2.0
                * torch.pi
                * doppler[:, None]
                * reference_index.to(dtype=real_dtype)[None, :]
                / float(self.length)
            )
            steering = torch.complex(torch.cos(angle), torch.sin(angle))
            values.append(torch.einsum("dl,blm->bdm", steering, product))
        return torch.stack(values, dim=2)

    def ambiguity(self, phase: torch.Tensor) -> torch.Tensor:
        """Return ``r[l,k]`` with shape ``[B,L,K,M]``."""

        phase, _ = self._validate(phase)
        sequence = torch.complex(torch.cos(phase), torch.sin(phase))
        return self.ambiguity_waveform(sequence)

    def normalized_power_samples_waveform(
        self, sequence: torch.Tensor
    ) -> torch.Tensor:
        ambiguity = self.ambiguity_waveform(sequence)
        power = ambiguity.abs().square() / float(self.length * self.length)
        batch, _, _, sequences = power.shape
        selected = power.reshape(batch, -1, sequences)[
            :, self.keep_mask.flatten(), :
        ]
        return selected.flatten(1).clamp_min(torch.finfo(power.dtype).tiny)

    def normalized_power_samples(self, phase: torch.Tensor) -> torch.Tensor:
        phase, _ = self._validate(phase)
        sequence = torch.complex(torch.cos(phase), torch.sin(phase))
        return self.normalized_power_samples_waveform(sequence)

    def waveform_terms(
        self,
        sequence: torch.Tensor,
        peak_order: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.implementation == "dense":
            ambiguity = self.ambiguity_waveform(sequence)
            power = ambiguity.abs().square() / float(self.length * self.length)
            tiny = torch.finfo(power.dtype).tiny
            order = self.peak_order if peak_order is None else peak_order
            logits = order * power.clamp_min(tiny).log()
            logits = logits.masked_fill(
                ~self.keep_mask[None, :, :, None], -torch.inf
            )
            smooth_log_peak = (
                torch.logsumexp(logits.flatten(1), dim=1) / order
            )
            exact_peak = power.masked_fill(
                ~self.keep_mask[None, :, :, None], -torch.inf
            ).flatten(1).amax(dim=1)
            return smooth_log_peak, exact_peak
        power = self.normalized_power_samples_waveform(sequence)
        order = self.peak_order if peak_order is None else peak_order
        smooth_log_peak = (
            torch.logsumexp(order * power.log(), dim=1)
            / order
        )
        return smooth_log_peak, power.amax(dim=1)

    def dense_waveform_terms_and_gradient(
        self,
        sequence: torch.Tensor,
        peak_order: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local-AF terms and their analytic complex gradient.

        The gradient follows PyTorch's convention for real-valued losses of
        complex inputs.  It avoids rebuilding an autograd graph in every
        unfolded inner update while preserving the same smooth objective.
        """

        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3 or sequence.shape[1] != self.length:
            raise ValueError("waveform must have shape [N,M] or [B,N,M]")
        if not torch.is_complex(sequence):
            raise TypeError("waveform must be complex")

        reference = sequence[:, self.dense_reference_index, :]
        shifted = sequence[:, self.dense_shifted_index, :]
        overlap = self.dense_overlap_mask[None, :, :, None]
        product = shifted * reference.conj() * overlap
        steering = self._dense_steering(sequence)
        ambiguity = torch.einsum("lkn,bknm->blkm", steering, product)
        power_full = ambiguity.abs().square() / float(self.length * self.length)
        batch, _, _, _ = power_full.shape
        tiny = torch.finfo(power_full.dtype).tiny
        order = self.peak_order if peak_order is None else peak_order
        full_logits = order * power_full.clamp_min(tiny).log()
        full_logits = full_logits.masked_fill(
            ~self.keep_mask[None, :, :, None], -torch.inf
        )
        smooth_log_peak = torch.logsumexp(full_logits.flatten(1), dim=1) / order
        exact_peak = power_full.masked_fill(
            ~self.keep_mask[None, :, :, None], -torch.inf
        ).flatten(1).amax(dim=1)
        weights = torch.softmax(full_logits.flatten(1), dim=1).reshape_as(
            power_full
        )
        ambiguity_gradient = (
            2.0
            * weights
            * ambiguity
            / (
                float(self.length * self.length)
                * power_full.clamp_min(tiny)
            )
        )

        shifted_steering, reference_steering = self._gradient_steering(sequence)
        reference_for_shifted = sequence[
            :, self.gradient_reference_for_shifted_index, :
        ] * self.gradient_shifted_valid[None, :, :, None]
        shifted_for_reference = sequence[
            :, self.gradient_shifted_for_reference_index, :
        ] * self.gradient_reference_valid[None, :, :, None]
        expanded_gradient = ambiguity_gradient[:, :, :, None, :]
        shifted_contribution = (
            expanded_gradient
            * reference_for_shifted[:, None, :, :, :]
            * shifted_steering[None, :, :, :, None].conj()
        )
        reference_contribution = (
            expanded_gradient.conj()
            * shifted_for_reference[:, None, :, :, :]
            * reference_steering[None, :, :, :, None]
        )
        gradient = (shifted_contribution + reference_contribution).sum(dim=(1, 2))
        return smooth_log_peak, exact_peak, gradient

    def terms(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smooth log-peak and the exact normalized peak power."""

        power = self.normalized_power_samples(phase)
        smooth_log_peak = (
            torch.logsumexp(self.peak_order * power.log(), dim=1)
            / self.peak_order
        )
        return smooth_log_peak, power.amax(dim=1)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return self.terms(phase)[0].mean()


@dataclass(frozen=True)
class PureLocalAFADMMResult:
    phase: torch.Tensor
    waveform: torch.Tensor
    local_af_peak_power: torch.Tensor
    initial_local_af_peak_power: torch.Tensor
    objective_history: torch.Tensor

    @property
    def local_af_psl_db(self) -> torch.Tensor:
        return 10.0 * torch.log10(self.local_af_peak_power.clamp_min(1.0e-30))


class OperatorAwarePureLocalAFADMMNet(nn.Module):
    """Operator-friendly local-AF ADMM with a tiny learned scalar schedule.

    The expensive local-AF direction remains analytic.  Only four scalars are
    exposed per parameter stage: the AF step, consensus step, inertial
    coefficient, and dual relaxation.  Several outer layers may share a stage,
    which keeps the schedule small and makes every inference update a fixed,
    elementwise-fuseable expression.

    Setting ``parameter_stages == outer_layers`` and both momentum endpoints to
    zero exactly reproduces the update coefficients of ``PureLocalAFADMMNet``
    when ``consensus_step = af_step * rho``.
    """

    def __init__(
        self,
        design: LocalAFWaveformDesign,
        region: LocalAFRegion,
        *,
        outer_layers: int = 56,
        inner_updates: int = 16,
        execution_layers: int | None = None,
        parameter_stages: int = 8,
        initial_rho: float = 0.2,
        final_rho: float | None = None,
        initial_af_step: float = 8.0e-2,
        final_af_step: float | None = None,
        initial_momentum: float = 0.0,
        final_momentum: float | None = None,
        max_abs_momentum: float = 0.95,
        initial_dual_scale: float = 1.0,
        final_dual_scale: float | None = None,
        peak_order: float = 8.0,
        final_peak_order: float | None = 128.0,
        af_implementation: str = "dense",
        gradient_implementation: str = "analytic",
        triton_batch: int = 1,
        outer_implementation: str = "auto",
        triton_projection_mode: int = 3,
        fuse_momentum: bool = False,
        diagonal_preconditioner: str = "none",
        select_best: bool = True,
    ) -> None:
        super().__init__()
        if outer_layers < 1 or inner_updates < 1:
            raise ValueError("outer_layers and inner_updates must be positive")
        if not 1 <= parameter_stages <= outer_layers:
            raise ValueError("parameter_stages must lie in [1, outer_layers]")
        if initial_rho <= 0.0 or initial_af_step <= 0.0:
            raise ValueError("rho and AF step must be positive")
        if initial_dual_scale <= 0.0:
            raise ValueError("dual scale must be positive")
        if not 0.0 < max_abs_momentum < 1.0:
            raise ValueError("max_abs_momentum must lie in (0, 1)")

        final_rho = initial_rho if final_rho is None else final_rho
        final_af_step = (
            initial_af_step if final_af_step is None else final_af_step
        )
        final_momentum = (
            initial_momentum if final_momentum is None else final_momentum
        )
        final_dual_scale = (
            initial_dual_scale
            if final_dual_scale is None
            else final_dual_scale
        )
        if final_rho <= 0.0 or final_af_step <= 0.0:
            raise ValueError("final rho and AF step must be positive")
        if final_dual_scale <= 0.0:
            raise ValueError("final dual scale must be positive")
        if max(abs(initial_momentum), abs(final_momentum)) >= max_abs_momentum:
            raise ValueError("momentum endpoints must be below max_abs_momentum")
        if peak_order <= 1.0:
            raise ValueError("peak_order must exceed one")
        if final_peak_order is not None and final_peak_order <= 1.0:
            raise ValueError("final peak order must exceed one")
        if gradient_implementation not in {
            "autograd",
            "analytic",
            "compiled",
            "triton",
        }:
            raise ValueError(
                "gradient_implementation must be 'autograd', 'analytic', or "
                "'compiled', or 'triton'"
            )
        if gradient_implementation in {"analytic", "compiled", "triton"} and (
            af_implementation != "dense"
        ):
            raise ValueError(
                "analytic, compiled, and Triton gradients require dense AF "
                "implementation"
            )
        if triton_batch < 1:
            raise ValueError("triton_batch must be positive")
        if gradient_implementation == "triton" and design.num_sequences != 1:
            raise ValueError("Triton local-AF update currently supports one waveform")
        if outer_implementation not in {"auto", "torch", "triton"}:
            raise ValueError(
                "outer_implementation must be 'auto', 'torch', or 'triton'"
            )
        if (
            outer_implementation == "triton"
            and gradient_implementation != "triton"
        ):
            raise ValueError(
                "Triton outer operators require gradient_implementation='triton'"
            )
        if triton_projection_mode not in {0, 1, 2, 3, 4}:
            raise ValueError(
                "triton_projection_mode must be one of 0, 1, 2, 3, or 4"
            )
        if fuse_momentum and gradient_implementation != "triton":
            raise ValueError(
                "fuse_momentum requires gradient_implementation='triton'"
            )
        if diagonal_preconditioner not in {"none", "coverage"}:
            raise ValueError(
                "diagonal_preconditioner must be 'none' or 'coverage'"
            )

        self.design = design
        self.region = region
        self.outer_layers = int(outer_layers)
        self.inner_updates = int(inner_updates)
        self.execution_layers = int(
            outer_layers if execution_layers is None else execution_layers
        )
        if not 1 <= self.execution_layers <= self.outer_layers:
            raise ValueError("execution_layers must lie in [1, outer_layers]")
        self.parameter_stages = int(parameter_stages)
        self.physical_updates = self.execution_layers * self.inner_updates
        self.select_best = bool(select_best)
        self.max_abs_momentum = float(max_abs_momentum)
        self.gradient_implementation = gradient_implementation
        self.outer_implementation = (
            "triton"
            if outer_implementation == "auto"
            and gradient_implementation == "triton"
            else (
                "torch" if outer_implementation == "auto" else outer_implementation
            )
        )
        self.triton_projection_mode = int(triton_projection_mode)
        # Optional inference-only fusion of the AF/ADMM direction and the
        # inertial recurrence.  Keep the default false so existing checkpoints
        # and backend comparisons retain their historical execution path.
        self.fuse_momentum = bool(fuse_momentum)
        self.diagonal_preconditioner = diagonal_preconditioner
        self.initial_peak_order = float(peak_order)
        self.final_peak_order = float(
            peak_order if final_peak_order is None else final_peak_order
        )

        # Contiguous outer-layer groups share one scalar tuple.  The mapping is
        # a compile-time constant during inference and introduces no branches
        # that depend on waveform data.
        stage_index = torch.div(
            torch.arange(outer_layers) * parameter_stages,
            outer_layers,
            rounding_mode="floor",
        ).clamp_max(parameter_stages - 1)
        self._layer_stage_indices = tuple(int(value) for value in stage_index)
        self.register_buffer("layer_stage_index", stage_index, persistent=False)
        self.register_buffer(
            "layer_peak_orders",
            torch.logspace(
                float(torch.log10(torch.tensor(self.initial_peak_order))),
                float(torch.log10(torch.tensor(self.final_peak_order))),
                outer_layers,
            ),
            persistent=False,
        )

        self.objective = LocalAmbiguityObjective(
            design.length,
            region,
            peak_order=peak_order,
            implementation=af_implementation,
        )
        self._compiled_terms_and_gradient = None
        self.triton_update: TritonLocalAFUpdate | None = None
        if gradient_implementation == "compiled":
            self._compiled_terms_and_gradient = torch.compile(
                self.objective.dense_waveform_terms_and_gradient,
                fullgraph=True,
                mode="reduce-overhead",
            )
        elif gradient_implementation == "triton":
            self.triton_update = TritonLocalAFUpdate(
                design.length,
                region.doppler_bins,
                region.delay_bins,
                batch=triton_batch,
                diagonal_preconditioner=diagonal_preconditioner,
            )

        def log_schedule(start: float, stop: float) -> torch.Tensor:
            return torch.linspace(
                float(torch.log(torch.tensor(start))),
                float(torch.log(torch.tensor(stop))),
                parameter_stages,
            )

        self.log_af_steps = nn.Parameter(
            log_schedule(initial_af_step, final_af_step)
        )
        # This is beta in alpha * grad(AF) + beta * (z - x + u).  Initializing
        # beta=alpha*rho makes a zero-momentum full-stage model equivalent to
        # the original ADMM gradient update while allowing the two scales to be
        # tuned independently afterwards.
        self.log_consensus_steps = nn.Parameter(
            log_schedule(
                initial_af_step * initial_rho,
                final_af_step * final_rho,
            )
        )
        momentum = torch.linspace(
            float(initial_momentum),
            float(final_momentum),
            parameter_stages,
        )
        normalized_momentum = momentum / self.max_abs_momentum
        self.raw_momenta = nn.Parameter(torch.atanh(normalized_momentum))
        self.log_dual_scales = nn.Parameter(
            log_schedule(initial_dual_scale, final_dual_scale)
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def scalar_schedules(self) -> dict[str, torch.Tensor]:
        """Return detached per-stage coefficients for logging and deployment."""

        af_steps = self.log_af_steps.exp().clamp(1.0e-7, 0.25)
        consensus_steps = self.log_consensus_steps.exp().clamp(1.0e-7, 0.25)
        momenta = self.max_abs_momentum * torch.tanh(self.raw_momenta)
        dual_scales = self.log_dual_scales.exp().clamp(0.05, 2.0)
        return {
            "af_steps": af_steps.detach(),
            "consensus_steps": consensus_steps.detach(),
            "effective_rhos": (consensus_steps / af_steps).detach(),
            "momenta": momenta.detach(),
            "dual_scales": dual_scales.detach(),
        }

    def _local_gradient(
        self,
        z: torch.Tensor,
        peak_order: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_implementation in {"analytic", "compiled"}:
            if self.gradient_implementation == "compiled":
                assert self._compiled_terms_and_gradient is not None
                _, peak, gradient = self._compiled_terms_and_gradient(z, peak_order)
            else:
                _, peak, gradient = self.objective.dense_waveform_terms_and_gradient(
                    z, peak_order
                )
            return peak, gradient

        if not z.requires_grad:
            z = z.requires_grad_(True)
        smooth_local, peak = self.objective.waveform_terms(z, peak_order)
        gradient = torch.autograd.grad(
            smooth_local.sum(), z, create_graph=self.training
        )[0]
        return peak, gradient

    def forward(self, initial_phase: torch.Tensor) -> PureLocalAFADMMResult:
        if initial_phase.ndim == 2:
            initial_phase = initial_phase.unsqueeze(0)
        if self.training and self.gradient_implementation in {"compiled", "triton"}:
            raise RuntimeError(
                "compiled and Triton AF gradients are inference-only backends"
            )
        expected = (self.design.length, self.design.num_sequences)
        if initial_phase.ndim != 3 or tuple(initial_phase.shape[-2:]) != expected:
            raise ValueError("initial_phase must have shape [N,M] or [B,N,M]")

        x = torch.complex(torch.cos(initial_phase), torch.sin(initial_phase))
        z = x.clone()
        dual = torch.zeros_like(z)
        with torch.no_grad():
            if self.outer_implementation == "triton":
                assert self.triton_update is not None
                _, initial_peak = self.triton_update.evaluate_peak(
                    x, self.layer_peak_orders[0]
                )
            else:
                _, initial_peak = self.objective.waveform_terms(x)
            best_peak = initial_peak.clone()
            best_x = x.detach().clone()
        history = [initial_peak]

        af_steps = self.log_af_steps.exp().clamp(1.0e-7, 0.25)
        consensus_steps = self.log_consensus_steps.exp().clamp(1.0e-7, 0.25)
        momenta = self.max_abs_momentum * torch.tanh(self.raw_momenta)
        dual_scales = self.log_dual_scales.exp().clamp(0.05, 2.0)

        for layer in range(self.execution_layers):
            # Use the construction-time Python constant during execution.
            # Converting a CUDA scalar to ``int`` would introduce a host sync
            # and is illegal while a CUDA Graph is being captured.
            stage = self._layer_stage_indices[layer]
            af_step = af_steps[stage]
            consensus_step = consensus_steps[stage]
            momentum = momenta[stage]
            dual_scale = dual_scales[stage]
            peak_order = self.layer_peak_orders[layer]
            # Momentum belongs to the z-subproblem and is reset after every
            # unit-circle projection, avoiding stale directions across ADMM
            # macro layers while retaining a fixed 16-update inner loop.
            velocity = torch.zeros_like(z)
            for _ in range(self.inner_updates):
                if self.gradient_implementation == "triton":
                    assert self.triton_update is not None
                    z, velocity = self.triton_update.operator_aware_update(
                        z,
                        x,
                        dual,
                        velocity,
                        af_step,
                        consensus_step,
                        momentum,
                        peak_order,
                        fuse_momentum=self.fuse_momentum,
                    )
                else:
                    _, local_gradient = self._local_gradient(z, peak_order)
                    raw_update = (
                        af_step * local_gradient
                        + consensus_step * (z - x + dual)
                    )
                    velocity = momentum * velocity + raw_update
                    z = z - velocity
                if not self.training:
                    z = z.detach()

            if self.outer_implementation == "triton":
                assert self.triton_update is not None
                x, dual = self.triton_update.project_and_update_dual(
                    z,
                    dual,
                    dual_scale,
                    projection_mode=self.triton_projection_mode,
                )
            else:
                projected = z + dual
                magnitude = projected.abs()
                denominator = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
                x = torch.where(magnitude > 0, projected / denominator, torch.ones_like(projected))
                dual = dual + dual_scale * (z - x)
            if not self.training:
                x = x.detach()
                dual = dual.detach()
            if self.outer_implementation == "triton":
                assert self.triton_update is not None
                _, peak = self.triton_update.evaluate_peak(
                    x, self.layer_peak_orders[0]
                )
            else:
                _, peak = self.objective.waveform_terms(x)
            history.append(peak.detach())
            if self.select_best:
                with torch.no_grad():
                    improve = peak < best_peak
                    best_peak = torch.where(improve, peak, best_peak)
                    best_x = torch.where(improve[:, None, None], x, best_x)

        output = best_x if (self.select_best and not self.training) else x
        if self.outer_implementation == "triton":
            assert self.triton_update is not None
            _, output_peak = self.triton_update.evaluate_peak(
                output, self.layer_peak_orders[0]
            )
        else:
            _, output_peak = self.objective.waveform_terms(output)
        phase = torch.remainder(torch.angle(output), 2.0 * torch.pi)
        return PureLocalAFADMMResult(
            phase=phase,
            waveform=output,
            local_af_peak_power=output_peak,
            initial_local_af_peak_power=initial_peak,
            objective_history=torch.stack(history, dim=1),
        )

    def unsupervised_loss(self, initial_phase: torch.Tensor) -> torch.Tensor:
        result = self(initial_phase)
        return self.objective(result.phase)


class CapturedPureLocalAFADMM(nn.Module):
    """Replay a fixed-shape local-AF network as one CUDA Graph.

    The captured model may use either the analytic PyTorch outer path or the
    Triton inner/outer inference path.
    """

    def __init__(
        self,
        model: OperatorAwarePureLocalAFADMMNet,
        example_phase: torch.Tensor,
        *,
        warmup: int = 1,
    ) -> None:
        super().__init__()
        if not example_phase.is_cuda or example_phase.dtype != torch.float32:
            raise TypeError("CUDA Graph backend requires CUDA float32 phase")
        if model.training:
            raise ValueError("model must be in eval mode before CUDA capture")
        if model.gradient_implementation not in {"analytic", "compiled", "triton"}:
            raise ValueError(
                "CUDA Graph backend requires an analytic, compiled, or Triton "
                "AF update"
            )
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.model = model
        self.static_phase = example_phase.detach().clone()

        current_stream = torch.cuda.current_stream(example_phase.device)
        warmup_stream = torch.cuda.Stream(device=example_phase.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(warmup):
                model(self.static_phase)
        current_stream.wait_stream(warmup_stream)
        current_stream.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_result = model(self.static_phase)

    @torch.inference_mode()
    def forward(self, initial_phase: torch.Tensor) -> PureLocalAFADMMResult:
        if initial_phase.shape != self.static_phase.shape:
            raise ValueError("captured input shape cannot change")
        if initial_phase.dtype != self.static_phase.dtype:
            raise TypeError("captured input dtype cannot change")
        self.static_phase.copy_(initial_phase)
        self.graph.replay()
        return self.static_result

