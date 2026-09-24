"""Fused Triton inner update for the single-waveform local-AF ADMM net."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _local_af_correlation_kernel(
        waveform_real_ptr,
        waveform_imag_ptr,
        delay_ptr,
        steering_real_ptr,
        steering_imag_ptr,
        residual_real_ptr,
        residual_imag_ptr,
        LENGTH: tl.constexpr,
        DOPPLERS: tl.constexpr,
        DELAYS: tl.constexpr,
        CELLS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        worker = tl.program_id(0)
        batch = worker // CELLS
        cell = worker % CELLS
        delay_index = cell % DELAYS
        delay = tl.load(delay_ptr + delay_index).to(tl.int32)
        position = tl.arange(0, BLOCK_N)
        shifted = position + delay
        valid = (position < LENGTH) & (shifted >= 0) & (shifted < LENGTH)
        base = batch * LENGTH
        reference_offset = 2 * (base + position)
        shifted_offset = 2 * (base + shifted)
        reference_real = tl.load(
            waveform_real_ptr + reference_offset, mask=valid, other=0.0
        )
        reference_imag = tl.load(
            waveform_imag_ptr + reference_offset, mask=valid, other=0.0
        )
        shifted_real = tl.load(
            waveform_real_ptr + shifted_offset, mask=valid, other=0.0
        )
        shifted_imag = tl.load(
            waveform_imag_ptr + shifted_offset, mask=valid, other=0.0
        )
        product_real = (
            shifted_real * reference_real + shifted_imag * reference_imag
        )
        product_imag = (
            shifted_imag * reference_real - shifted_real * reference_imag
        )
        steering_offset = cell * LENGTH + position
        steering_real = tl.load(
            steering_real_ptr + steering_offset, mask=valid, other=0.0
        )
        steering_imag = tl.load(
            steering_imag_ptr + steering_offset, mask=valid, other=0.0
        )
        value_real = tl.sum(
            product_real * steering_real - product_imag * steering_imag,
            axis=0,
        )
        value_imag = tl.sum(
            product_real * steering_imag + product_imag * steering_real,
            axis=0,
        )
        output = batch * CELLS + cell
        tl.store(residual_real_ptr + output, value_real)
        tl.store(residual_imag_ptr + output, value_imag)


    @triton.jit
    def _local_af_softmax_gradient_kernel(
        residual_real_ptr,
        residual_imag_ptr,
        ambiguity_gradient_real_ptr,
        ambiguity_gradient_imag_ptr,
        peak_order_ptr,
        LENGTH_SQUARED: tl.constexpr,
        CELLS: tl.constexpr,
        ORIGIN_CELL: tl.constexpr,
        BLOCK_CELLS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cell = tl.arange(0, BLOCK_CELLS)
        valid = (cell < CELLS) & (cell != ORIGIN_CELL)
        base = batch * CELLS
        real = tl.load(residual_real_ptr + base + cell, mask=valid, other=0.0)
        imag = tl.load(residual_imag_ptr + base + cell, mask=valid, other=0.0)
        squared = real * real + imag * imag
        order = tl.load(peak_order_ptr)
        log_power = tl.log(tl.maximum(squared / LENGTH_SQUARED, 1.0e-30))
        logits = tl.where(valid, order * log_power, -float("inf"))
        maximum = tl.max(logits, axis=0)
        exponential = tl.where(valid, tl.exp(logits - maximum), 0.0)
        normalizer = tl.sum(exponential, axis=0)
        weight = exponential / tl.maximum(normalizer, 1.0e-30)
        scale = 2.0 * weight / tl.maximum(squared, 1.0e-30)
        tl.store(
            ambiguity_gradient_real_ptr + base + cell,
            scale * real,
            mask=cell < CELLS,
        )
        tl.store(
            ambiguity_gradient_imag_ptr + base + cell,
            scale * imag,
            mask=cell < CELLS,
        )


    @triton.jit
    def _local_af_gradient_update_kernel(
        waveform_real_ptr,
        waveform_imag_ptr,
        consensus_real_ptr,
        consensus_imag_ptr,
        dual_real_ptr,
        dual_imag_ptr,
        delay_ptr,
        steering_real_ptr,
        steering_imag_ptr,
        ambiguity_gradient_real_ptr,
        ambiguity_gradient_imag_ptr,
        preconditioner_ptr,
        rho_ptr,
        step_ptr,
        output_real_ptr,
        output_imag_ptr,
        LENGTH: tl.constexpr,
        DELAYS: tl.constexpr,
        CELLS: tl.constexpr,
        BLOCK_CELLS: tl.constexpr,
    ):
        worker = tl.program_id(0)
        batch = worker // LENGTH
        position = worker % LENGTH
        cell = tl.arange(0, BLOCK_CELLS)
        valid_cell = cell < CELLS
        delay_index = cell % DELAYS
        delay = tl.load(delay_ptr + delay_index, mask=valid_cell, other=0).to(
            tl.int32
        )
        residual_offset = batch * CELLS + cell
        gradient_real = tl.load(
            ambiguity_gradient_real_ptr + residual_offset,
            mask=valid_cell,
            other=0.0,
        )
        gradient_imag = tl.load(
            ambiguity_gradient_imag_ptr + residual_offset,
            mask=valid_cell,
            other=0.0,
        )

        reference = position - delay
        shifted_valid = valid_cell & (reference >= 0) & (reference < LENGTH)
        reference_offset = 2 * (batch * LENGTH + reference)
        reference_real = tl.load(
            waveform_real_ptr + reference_offset,
            mask=shifted_valid,
            other=0.0,
        )
        reference_imag = tl.load(
            waveform_imag_ptr + reference_offset,
            mask=shifted_valid,
            other=0.0,
        )
        steering_offset = cell * LENGTH + reference
        steering_real = tl.load(
            steering_real_ptr + steering_offset,
            mask=shifted_valid,
            other=0.0,
        )
        steering_imag = tl.load(
            steering_imag_ptr + steering_offset,
            mask=shifted_valid,
            other=0.0,
        )
        product_real = (
            gradient_real * reference_real - gradient_imag * reference_imag
        )
        product_imag = (
            gradient_real * reference_imag + gradient_imag * reference_real
        )
        shifted_gradient_real = (
            product_real * steering_real + product_imag * steering_imag
        )
        shifted_gradient_imag = (
            product_imag * steering_real - product_real * steering_imag
        )

        shifted = position + delay
        reference_valid = valid_cell & (shifted >= 0) & (shifted < LENGTH)
        shifted_offset = 2 * (batch * LENGTH + shifted)
        shifted_real = tl.load(
            waveform_real_ptr + shifted_offset,
            mask=reference_valid,
            other=0.0,
        )
        shifted_imag = tl.load(
            waveform_imag_ptr + shifted_offset,
            mask=reference_valid,
            other=0.0,
        )
        reference_steering_offset = cell * LENGTH + position
        reference_steering_real = tl.load(
            steering_real_ptr + reference_steering_offset,
            mask=reference_valid,
            other=0.0,
        )
        reference_steering_imag = tl.load(
            steering_imag_ptr + reference_steering_offset,
            mask=reference_valid,
            other=0.0,
        )
        conjugate_product_real = (
            gradient_real * shifted_real + gradient_imag * shifted_imag
        )
        conjugate_product_imag = (
            gradient_real * shifted_imag - gradient_imag * shifted_real
        )
        reference_gradient_real = (
            conjugate_product_real * reference_steering_real
            - conjugate_product_imag * reference_steering_imag
        )
        reference_gradient_imag = (
            conjugate_product_real * reference_steering_imag
            + conjugate_product_imag * reference_steering_real
        )
        local_real = tl.sum(
            tl.where(
                shifted_valid, shifted_gradient_real, 0.0
            )
            + tl.where(reference_valid, reference_gradient_real, 0.0),
            axis=0,
        )
        local_imag = tl.sum(
            tl.where(
                shifted_valid, shifted_gradient_imag, 0.0
            )
            + tl.where(reference_valid, reference_gradient_imag, 0.0),
            axis=0,
        )
        preconditioner = tl.load(preconditioner_ptr + position)
        local_real /= preconditioner
        local_imag /= preconditioner

        current_offset = 2 * (batch * LENGTH + position)
        current_real = tl.load(waveform_real_ptr + current_offset)
        current_imag = tl.load(waveform_imag_ptr + current_offset)
        consensus_real = tl.load(consensus_real_ptr + current_offset)
        consensus_imag = tl.load(consensus_imag_ptr + current_offset)
        dual_real = tl.load(dual_real_ptr + current_offset)
        dual_imag = tl.load(dual_imag_ptr + current_offset)
        rho = tl.load(rho_ptr)
        step = tl.load(step_ptr)
        local_real += rho * (current_real - consensus_real + dual_real)
        local_imag += rho * (current_imag - consensus_imag + dual_imag)
        tl.store(output_real_ptr + current_offset, current_real - step * local_real)
        tl.store(output_imag_ptr + current_offset, current_imag - step * local_imag)


    @triton.jit
    def _local_af_gradient_momentum_update_kernel(
        waveform_real_ptr,
        waveform_imag_ptr,
        consensus_real_ptr,
        consensus_imag_ptr,
        dual_real_ptr,
        dual_imag_ptr,
        velocity_real_ptr,
        velocity_imag_ptr,
        delay_ptr,
        steering_real_ptr,
        steering_imag_ptr,
        ambiguity_gradient_real_ptr,
        ambiguity_gradient_imag_ptr,
        preconditioner_ptr,
        rho_ptr,
        step_ptr,
        momentum_ptr,
        output_real_ptr,
        output_imag_ptr,
        next_velocity_real_ptr,
        next_velocity_imag_ptr,
        LENGTH: tl.constexpr,
        DELAYS: tl.constexpr,
        CELLS: tl.constexpr,
        BLOCK_CELLS: tl.constexpr,
    ):
        """Fuse AF/ADMM gradient construction with the inertial update.

        The original gradient kernel writes a candidate waveform, after which
        a separate elementwise recurrence computes ``v <- mu*v + raw`` and
        ``z <- z-v``.  This variant keeps the same reduction over AF cells but
        consumes the current velocity and writes the next waveform and
        velocity directly, avoiding the candidate and raw-update buffers.
        """

        worker = tl.program_id(0)
        batch = worker // LENGTH
        position = worker % LENGTH
        cell = tl.arange(0, BLOCK_CELLS)
        valid_cell = cell < CELLS
        delay_index = cell % DELAYS
        delay = tl.load(delay_ptr + delay_index, mask=valid_cell, other=0).to(
            tl.int32
        )
        residual_offset = batch * CELLS + cell
        gradient_real = tl.load(
            ambiguity_gradient_real_ptr + residual_offset,
            mask=valid_cell,
            other=0.0,
        )
        gradient_imag = tl.load(
            ambiguity_gradient_imag_ptr + residual_offset,
            mask=valid_cell,
            other=0.0,
        )

        reference = position - delay
        shifted_valid = valid_cell & (reference >= 0) & (reference < LENGTH)
        reference_offset = 2 * (batch * LENGTH + reference)
        reference_real = tl.load(
            waveform_real_ptr + reference_offset,
            mask=shifted_valid,
            other=0.0,
        )
        reference_imag = tl.load(
            waveform_imag_ptr + reference_offset,
            mask=shifted_valid,
            other=0.0,
        )
        steering_offset = cell * LENGTH + reference
        steering_real = tl.load(
            steering_real_ptr + steering_offset,
            mask=shifted_valid,
            other=0.0,
        )
        steering_imag = tl.load(
            steering_imag_ptr + steering_offset,
            mask=shifted_valid,
            other=0.0,
        )
        product_real = (
            gradient_real * reference_real - gradient_imag * reference_imag
        )
        product_imag = (
            gradient_real * reference_imag + gradient_imag * reference_real
        )
        shifted_gradient_real = (
            product_real * steering_real + product_imag * steering_imag
        )
        shifted_gradient_imag = (
            product_imag * steering_real - product_real * steering_imag
        )

        shifted = position + delay
        reference_valid = valid_cell & (shifted >= 0) & (shifted < LENGTH)
        shifted_offset = 2 * (batch * LENGTH + shifted)
        shifted_real = tl.load(
            waveform_real_ptr + shifted_offset,
            mask=reference_valid,
            other=0.0,
        )
        shifted_imag = tl.load(
            waveform_imag_ptr + shifted_offset,
            mask=reference_valid,
            other=0.0,
        )
        reference_steering_offset = cell * LENGTH + position
        reference_steering_real = tl.load(
            steering_real_ptr + reference_steering_offset,
            mask=reference_valid,
            other=0.0,
        )
        reference_steering_imag = tl.load(
            steering_imag_ptr + reference_steering_offset,
            mask=reference_valid,
            other=0.0,
        )
        conjugate_product_real = (
            gradient_real * shifted_real + gradient_imag * shifted_imag
        )
        conjugate_product_imag = (
            gradient_real * shifted_imag - gradient_imag * shifted_real
        )
        reference_gradient_real = (
            conjugate_product_real * reference_steering_real
            - conjugate_product_imag * reference_steering_imag
        )
        reference_gradient_imag = (
            conjugate_product_real * reference_steering_imag
            + conjugate_product_imag * reference_steering_real
        )
        local_real = tl.sum(
            tl.where(shifted_valid, shifted_gradient_real, 0.0)
            + tl.where(reference_valid, reference_gradient_real, 0.0),
            axis=0,
        )
        local_imag = tl.sum(
            tl.where(shifted_valid, shifted_gradient_imag, 0.0)
            + tl.where(reference_valid, reference_gradient_imag, 0.0),
            axis=0,
        )
        preconditioner = tl.load(preconditioner_ptr + position)
        local_real /= preconditioner
        local_imag /= preconditioner

        current_offset = 2 * (batch * LENGTH + position)
        current_real = tl.load(waveform_real_ptr + current_offset)
        current_imag = tl.load(waveform_imag_ptr + current_offset)
        consensus_real = tl.load(consensus_real_ptr + current_offset)
        consensus_imag = tl.load(consensus_imag_ptr + current_offset)
        dual_real = tl.load(dual_real_ptr + current_offset)
        dual_imag = tl.load(dual_imag_ptr + current_offset)
        velocity_real = tl.load(velocity_real_ptr + current_offset)
        velocity_imag = tl.load(velocity_imag_ptr + current_offset)
        rho = tl.load(rho_ptr)
        step = tl.load(step_ptr)
        momentum = tl.load(momentum_ptr)

        local_real += rho * (current_real - consensus_real + dual_real)
        local_imag += rho * (current_imag - consensus_imag + dual_imag)
        # Preserve the historical K3 -> Torch-momentum arithmetic ordering:
        # K3 first forms ``candidate = current - step * local`` and the
        # fallback then recovers ``raw = current - candidate``.  Keeping this
        # subtraction in registers avoids the candidate global-memory round
        # trip while reducing long-horizon drift between the two paths.
        candidate_real = current_real - step * local_real
        candidate_imag = current_imag - step * local_imag
        raw_real = current_real - candidate_real
        raw_imag = current_imag - candidate_imag
        next_velocity_real = momentum * velocity_real + raw_real
        next_velocity_imag = momentum * velocity_imag + raw_imag
        tl.store(next_velocity_real_ptr + current_offset, next_velocity_real)
        tl.store(next_velocity_imag_ptr + current_offset, next_velocity_imag)
        tl.store(output_real_ptr + current_offset, current_real - next_velocity_real)
        tl.store(output_imag_ptr + current_offset, current_imag - next_velocity_imag)


    @triton.jit
    def _local_af_momentum_combine_kernel(
        current_real_ptr,
        current_imag_ptr,
        candidate_real_ptr,
        candidate_imag_ptr,
        velocity_real_ptr,
        velocity_imag_ptr,
        momentum_ptr,
        output_real_ptr,
        output_imag_ptr,
        next_velocity_real_ptr,
        next_velocity_imag_ptr,
        LENGTH: tl.constexpr,
    ):
        """Apply the operator-aware inertial step elementwise.

        ``_local_af_gradient_update_kernel`` returns ``current - raw_update``
        where ``raw_update = alpha*g + beta*(z-x+u)``.  Keeping this small
        combine in a separate kernel lets us reuse the already validated AF
        correlation/gradient kernels while still implementing
        ``v <- mu*v + raw_update; z <- z-v`` without a PyTorch launch.
        """

        worker = tl.program_id(0)
        batch = worker // LENGTH
        position = worker % LENGTH
        offset = 2 * (batch * LENGTH + position)

        current_real = tl.load(current_real_ptr + offset)
        current_imag = tl.load(current_imag_ptr + offset)
        candidate_real = tl.load(candidate_real_ptr + offset)
        candidate_imag = tl.load(candidate_imag_ptr + offset)
        velocity_real = tl.load(velocity_real_ptr + offset)
        velocity_imag = tl.load(velocity_imag_ptr + offset)
        momentum = tl.load(momentum_ptr)

        raw_real = current_real - candidate_real
        raw_imag = current_imag - candidate_imag
        next_real = momentum * velocity_real + raw_real
        next_imag = momentum * velocity_imag + raw_imag
        tl.store(next_velocity_real_ptr + offset, next_real)
        tl.store(next_velocity_imag_ptr + offset, next_imag)
        tl.store(output_real_ptr + offset, current_real - next_real)
        tl.store(output_imag_ptr + offset, current_imag - next_imag)


    @triton.jit
    def _local_af_projection_dual_kernel(
        waveform_real_ptr,
        waveform_imag_ptr,
        dual_real_ptr,
        dual_imag_ptr,
        dual_scale_ptr,
        output_x_real_ptr,
        output_x_imag_ptr,
        output_dual_real_ptr,
        output_dual_imag_ptr,
        LENGTH: tl.constexpr,
        PROJECTION_MODE: tl.constexpr,
    ):
        """Project onto the unit circle and update the scaled dual variable.

        One program owns one waveform sample.  Keeping the projection and dual
        update in the same program avoids materialising ``z + u`` and then
        reading the projected value again for the dual update.
        """

        worker = tl.program_id(0)
        batch = worker // LENGTH
        position = worker % LENGTH
        offset = 2 * (batch * LENGTH + position)

        current_real = tl.load(waveform_real_ptr + offset)
        current_imag = tl.load(waveform_imag_ptr + offset)
        dual_real = tl.load(dual_real_ptr + offset)
        dual_imag = tl.load(dual_imag_ptr + offset)

        if PROJECTION_MODE == 4:
            current_real = current_real.to(tl.float64)
            current_imag = current_imag.to(tl.float64)
            dual_real = dual_real.to(tl.float64)
            dual_imag = dual_imag.to(tl.float64)

        projected_real = current_real + dual_real
        projected_imag = current_imag + dual_imag
        # Mode 0 is the direct float32 expression, mode 1 is a scaled
        # float32 hypot, mode 2 uses a reciprocal after the direct sqrt, and
        # mode 3 uses a reciprocal after the scaled hypot.  Mode 4 performs
        # the projection arithmetic in float64 before storing complex64.  The
        # modes make numerical/quality ablations explicit without changing the
        # ADMM ordering.
        if PROJECTION_MODE == 0 or PROJECTION_MODE == 2:
            magnitude = tl.sqrt(
                projected_real * projected_real
                + projected_imag * projected_imag
            )
        elif PROJECTION_MODE == 4:
            projected_real64 = projected_real.to(tl.float64)
            projected_imag64 = projected_imag.to(tl.float64)
            magnitude = tl.sqrt(
                projected_real64 * projected_real64
                + projected_imag64 * projected_imag64
            )
        else:
            abs_real = tl.abs(projected_real)
            abs_imag = tl.abs(projected_imag)
            scale = tl.maximum(abs_real, abs_imag)
            safe_scale = tl.maximum(scale, 1.0e-30)
            magnitude = safe_scale * tl.sqrt(
                (projected_real / safe_scale)
                * (projected_real / safe_scale)
                + (projected_imag / safe_scale)
                * (projected_imag / safe_scale)
            )
            magnitude = tl.where(scale > 0.0, magnitude, 0.0)
        denominator = tl.maximum(magnitude, 1.0e-12)
        if PROJECTION_MODE == 2 or PROJECTION_MODE == 3:
            reciprocal = 1.0 / denominator
            consensus_real = projected_real * reciprocal
            consensus_imag = projected_imag * reciprocal
        else:
            consensus_real = projected_real / denominator
            consensus_imag = projected_imag / denominator

        dual_scale = tl.load(dual_scale_ptr)
        if PROJECTION_MODE == 4:
            dual_scale = dual_scale.to(tl.float64)
        next_dual_real = dual_real + dual_scale * (
            current_real - consensus_real
        )
        next_dual_imag = dual_imag + dual_scale * (
            current_imag - consensus_imag
        )

        tl.store(output_x_real_ptr + offset, consensus_real.to(tl.float32))
        tl.store(output_x_imag_ptr + offset, consensus_imag.to(tl.float32))
        tl.store(output_dual_real_ptr + offset, next_dual_real.to(tl.float32))
        tl.store(output_dual_imag_ptr + offset, next_dual_imag.to(tl.float32))


    @triton.jit
    def _local_af_peak_kernel(
        residual_real_ptr,
        residual_imag_ptr,
        peak_order_ptr,
        smooth_output_ptr,
        exact_output_ptr,
        LENGTH_SQUARED: tl.constexpr,
        CELLS: tl.constexpr,
        ORIGIN_CELL: tl.constexpr,
        BLOCK_CELLS: tl.constexpr,
    ):
        """Reduce local AF samples to the smooth and exact peak values."""

        batch = tl.program_id(0)
        cell = tl.arange(0, BLOCK_CELLS)
        valid = (cell < CELLS) & (cell != ORIGIN_CELL)
        base = batch * CELLS
        real = tl.load(residual_real_ptr + base + cell, mask=valid, other=0.0)
        imag = tl.load(residual_imag_ptr + base + cell, mask=valid, other=0.0)
        power = (real * real + imag * imag) / LENGTH_SQUARED

        exact_peak = tl.max(
            tl.where(valid, power, -float("inf")), axis=0
        )
        order = tl.load(peak_order_ptr)
        logits = tl.where(
            valid,
            order * tl.log(tl.maximum(power, 1.0e-30)),
            -float("inf"),
        )
        maximum = tl.max(logits, axis=0)
        exponential = tl.where(valid, tl.exp(logits - maximum), 0.0)
        normalizer = tl.sum(exponential, axis=0)
        smooth_peak = tl.log(
            tl.maximum(normalizer, 1.0e-30)
        ) / order + maximum / order

        tl.store(smooth_output_ptr + batch, smooth_peak)
        tl.store(exact_output_ptr + batch, exact_peak)



    @triton.jit
    def _repair_tiny_projection(zr, zi, ur, ui, gamma, xr, xi, vr, vi,
                                COUNT: tl.constexpr, BLOCK: tl.constexpr):
        # A separate guard preserves the original kernel's compiled arithmetic.
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = p < COUNT
        a = tl.load(zr + 2*p, valid, other=0.0)
        b = tl.load(zi + 2*p, valid, other=0.0)
        c = tl.load(ur + 2*p, valid, other=0.0)
        d = tl.load(ui + 2*p, valid, other=0.0)
        pr, pi = a+c, b+d
        scale = tl.maximum(tl.abs(pr), tl.abs(pi))
        repair = valid & (scale < 1.0e-12)
        safe = tl.where(scale > 0.0, scale, 1.0)
        nr, ni = pr/safe, pi/safe
        norm = tl.sqrt(nr*nr + ni*ni)
        denom = tl.where(norm > 0.0, norm, 1.0)
        real = tl.where(scale > 0.0, nr/denom, 1.0)
        imag = tl.where(scale > 0.0, ni/denom, 0.0)
        g = tl.load(gamma)
        tl.store(xr+2*p, real, repair)
        tl.store(xi+2*p, imag, repair)
        tl.store(vr+2*p, c+g*(a-real), repair)
        tl.store(vi+2*p, d+g*(b-imag), repair)


class TritonLocalAFUpdate(nn.Module):
    """Triton local-AF kernels used by the unfolded ADMM solver.

    The original three-kernel inner update is kept intact.  The same geometry
    and scratch buffers are also used for two inference-side outer operators:
    a fused unit-circle projection/dual update and a local AF peak reduction.
    """

    def __init__(
        self,
        length: int,
        doppler_bins: tuple[float, ...],
        delay_bins: tuple[int, ...],
        *,
        batch: int = 1,
        diagonal_preconditioner: str = "none",
    ) -> None:
        super().__init__()
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        self.length = int(length)
        self.doppler_count = len(doppler_bins)
        self.delay_count = len(delay_bins)
        self.cell_count = self.doppler_count * self.delay_count
        self.batch = int(batch)
        if diagonal_preconditioner not in {"none", "coverage"}:
            raise ValueError("diagonal_preconditioner must be 'none' or 'coverage'")
        self.diagonal_preconditioner = diagonal_preconditioner
        delays = torch.tensor(delay_bins, dtype=torch.int32)
        positions = torch.arange(length, dtype=torch.float64)
        dopplers = torch.tensor(doppler_bins, dtype=torch.float64)
        shifted = positions[None, :] + delays.to(torch.float64)[:, None]
        valid = (shifted >= 0) & (shifted < length)
        angle = (
            -2.0
            * torch.pi
            * dopplers[:, None, None]
            * positions[None, None, :]
            / float(length)
        )
        steering_real = torch.cos(angle) * valid[None, :, :]
        steering_imag = torch.sin(angle) * valid[None, :, :]
        self.register_buffer("delays", delays)
        self.register_buffer("steering_real", steering_real.float().flatten())
        self.register_buffer("steering_imag", steering_imag.float().flatten())
        # A normalized diagonal approximation to the AF-gradient curvature.
        # The count is proportional to the number of valid shifted/reference
        # terms touching each waveform sample.  Normalizing by its mean keeps
        # the learned AF step on the same scale as the unpreconditioned path.
        coverage = torch.zeros(length, dtype=torch.float32)
        for delay in delay_bins:
            coverage += float(self.doppler_count) * (
                (positions - float(delay) >= 0.0)
                & (positions - float(delay) < float(length))
            ).float()
            coverage += float(self.doppler_count) * (
                (positions + float(delay) >= 0.0)
                & (positions + float(delay) < float(length))
            ).float()
        coverage = coverage / coverage.mean().clamp_min(1.0e-6)
        if diagonal_preconditioner == "none":
            coverage.fill_(1.0)
        self.register_buffer("preconditioner", coverage)
        self.register_buffer(
            "residual_real", torch.empty(batch, self.cell_count)
        )
        self.register_buffer("residual_imag", torch.empty(batch, self.cell_count))
        self.register_buffer(
            "ambiguity_gradient_real", torch.empty(batch, self.cell_count)
        )
        self.register_buffer(
            "ambiguity_gradient_imag", torch.empty(batch, self.cell_count)
        )
        doppler_origin = doppler_bins.index(0.0)
        delay_origin = delay_bins.index(0)
        self.origin_cell = doppler_origin * self.delay_count + delay_origin

    def _validate_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if not waveform.is_cuda or waveform.dtype != torch.complex64:
            raise TypeError("Triton local-AF kernels require CUDA complex64")
        expected = (self.batch, self.length, 1)
        if tuple(waveform.shape) != expected:
            raise ValueError(f"waveform shape must be {expected}")
        if not waveform.is_contiguous():
            waveform = waveform.contiguous()
        return waveform

    def _launch_correlation(self, waveform: torch.Tensor) -> None:
        """Populate the real/imaginary local AF scratch buffers."""

        block_n = triton.next_power_of_2(self.length)
        _local_af_correlation_kernel[(self.batch * self.cell_count,)](
            waveform.real,
            waveform.imag,
            self.delays,
            self.steering_real,
            self.steering_imag,
            self.residual_real,
            self.residual_imag,
            LENGTH=self.length,
            DOPPLERS=self.doppler_count,
            DELAYS=self.delay_count,
            CELLS=self.cell_count,
            BLOCK_N=block_n,
            num_warps=4,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        consensus: torch.Tensor,
        dual: torch.Tensor,
        rho: torch.Tensor,
        step_size: torch.Tensor,
        peak_order: torch.Tensor,
    ) -> torch.Tensor:
        waveform = self._validate_waveform(waveform)
        consensus = self._validate_waveform(consensus)
        dual = self._validate_waveform(dual)
        if not consensus.is_contiguous():
            consensus = consensus.contiguous()
        if not dual.is_contiguous():
            dual = dual.contiguous()
        output = torch.empty_like(waveform)
        self._launch_correlation(waveform)
        block_cells = triton.next_power_of_2(self.cell_count)
        _local_af_softmax_gradient_kernel[(self.batch,)](
            self.residual_real,
            self.residual_imag,
            self.ambiguity_gradient_real,
            self.ambiguity_gradient_imag,
            peak_order,
            LENGTH_SQUARED=float(self.length * self.length),
            CELLS=self.cell_count,
            ORIGIN_CELL=self.origin_cell,
            BLOCK_CELLS=block_cells,
            num_warps=4,
        )
        _local_af_gradient_update_kernel[(self.batch * self.length,)](
            waveform.real,
            waveform.imag,
            consensus.real,
            consensus.imag,
            dual.real,
            dual.imag,
            self.delays,
            self.steering_real,
            self.steering_imag,
            self.ambiguity_gradient_real,
            self.ambiguity_gradient_imag,
            self.preconditioner,
            rho,
            step_size,
            output.real,
            output.imag,
            LENGTH=self.length,
            DELAYS=self.delay_count,
            CELLS=self.cell_count,
            BLOCK_CELLS=block_cells,
            num_warps=4,
        )
        return output

    @torch.no_grad()
    def operator_aware_update(
        self,
        waveform: torch.Tensor,
        consensus: torch.Tensor,
        dual: torch.Tensor,
        velocity: torch.Tensor,
        af_step: torch.Tensor,
        consensus_step: torch.Tensor,
        momentum: torch.Tensor,
        peak_order: torch.Tensor,
        *,
        fuse_momentum: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one operator-aware inner update.

        The AF and consensus directions have independent scalar coefficients
        (``af_step`` and ``consensus_step``), followed by a stage-shared
        inertial coefficient.  The existing AF kernel computes the raw
        direction by setting ``rho=consensus_step/af_step`` and
        ``step=af_step``.  When ``fuse_momentum`` is true, the AF/ADMM
        direction and the inertial recurrence are emitted by one Triton
        kernel; otherwise the historical Torch fallback is retained.
        """

        waveform = self._validate_waveform(waveform)
        consensus = self._validate_waveform(consensus)
        dual = self._validate_waveform(dual)
        velocity = self._validate_waveform(velocity)
        if not torch.is_tensor(af_step) or af_step.ndim != 0:
            raise ValueError("af_step must be a scalar tensor")
        if not torch.is_tensor(consensus_step) or consensus_step.ndim != 0:
            raise ValueError("consensus_step must be a scalar tensor")
        if not torch.is_tensor(momentum) or momentum.ndim != 0:
            raise ValueError("momentum must be a scalar tensor")
        for value, name in (
            (af_step, "af_step"),
            (consensus_step, "consensus_step"),
            (momentum, "momentum"),
        ):
            if value.device != waveform.device:
                raise ValueError(f"{name} must be on the waveform device")
            if value.dtype != waveform.real.dtype:
                raise ValueError(f"{name} must have the waveform real dtype")
        if not velocity.is_contiguous():
            velocity = velocity.contiguous()

        # The base kernel writes ``candidate = waveform - raw_update``.
        # Positive AF steps are guaranteed by the network constructor, so the
        # ratio is well-conditioned and remains a scalar CUDA value during
        # CUDA Graph capture.
        rho = consensus_step / af_step

        if fuse_momentum:
            # The fused variant keeps the same correlation and PSL-weight
            # stages, but directly combines their gradient output with the
            # inertial recurrence.  No candidate tensor or intermediate
            # raw-update tensor is materialised.
            output = torch.empty_like(waveform)
            next_velocity = torch.empty_like(velocity)
            self._launch_correlation(waveform)
            block_cells = triton.next_power_of_2(self.cell_count)
            _local_af_softmax_gradient_kernel[(self.batch,)](
                self.residual_real,
                self.residual_imag,
                self.ambiguity_gradient_real,
                self.ambiguity_gradient_imag,
                peak_order,
                LENGTH_SQUARED=float(self.length * self.length),
                CELLS=self.cell_count,
                ORIGIN_CELL=self.origin_cell,
                BLOCK_CELLS=block_cells,
                num_warps=4,
            )
            _local_af_gradient_momentum_update_kernel[(self.batch * self.length,)](
                waveform.real,
                waveform.imag,
                consensus.real,
                consensus.imag,
                dual.real,
                dual.imag,
                velocity.real,
                velocity.imag,
                self.delays,
                self.steering_real,
                self.steering_imag,
                self.ambiguity_gradient_real,
                self.ambiguity_gradient_imag,
                self.preconditioner,
                rho,
                af_step,
                momentum,
                output.real,
                output.imag,
                next_velocity.real,
                next_velocity.imag,
                LENGTH=self.length,
                DELAYS=self.delay_count,
                CELLS=self.cell_count,
                BLOCK_CELLS=block_cells,
                num_warps=4,
            )
            return output, next_velocity

        candidate = self.forward(
            waveform,
            consensus,
            dual,
            rho,
            af_step,
            peak_order,
        )
        # Keep a Torch fallback for environments where the optional extra
        # elementwise kernel has not been compiled yet (notably the Windows
        # Triton toolchain).  The expensive AF correlation/gradient and the
        # projection/peak operators remain Triton; this two-line recurrence is
        # bandwidth-bound and contributes little to the end-to-end latency.
        raw_update = waveform - candidate
        next_velocity = momentum * velocity + raw_update
        output = waveform - next_velocity
        return output, next_velocity

    @torch.no_grad()
    def project_and_update_dual(
        self,
        waveform: torch.Tensor,
        dual: torch.Tensor,
        dual_scale: torch.Tensor | float,
        *,
        projection_mode: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the exact outer ADMM projection and dual update in one kernel."""

        if projection_mode not in {0, 1, 2, 3, 4}:
            raise ValueError("projection_mode must be one of 0, 1, 2, 3, or 4")
        waveform = self._validate_waveform(waveform)
        dual = self._validate_waveform(dual)
        if not torch.is_tensor(dual_scale):
            dual_scale = torch.tensor(
                float(dual_scale),
                device=waveform.device,
                dtype=waveform.real.dtype,
            )
        if dual_scale.ndim != 0:
            raise ValueError("dual_scale must be a scalar tensor")
        if dual_scale.device != waveform.device:
            raise ValueError("dual_scale must be on the waveform device")
        if dual_scale.dtype != waveform.real.dtype:
            dual_scale = dual_scale.to(dtype=waveform.real.dtype)

        consensus = torch.empty_like(waveform)
        next_dual = torch.empty_like(dual)
        _local_af_projection_dual_kernel[(self.batch * self.length,)](
            waveform.real,
            waveform.imag,
            dual.real,
            dual.imag,
            dual_scale,
            consensus.real,
            consensus.imag,
            next_dual.real,
            next_dual.imag,
            LENGTH=self.length,
            PROJECTION_MODE=projection_mode,
            num_warps=4,
        )
        _repair_tiny_projection[(triton.cdiv(self.batch*self.length, 256),)](
            waveform.real, waveform.imag, dual.real, dual.imag, dual_scale,
            consensus.real, consensus.imag, next_dual.real, next_dual.imag,
            COUNT=self.batch*self.length, BLOCK=256, num_warps=4)
        return consensus, next_dual

    @torch.no_grad()
    def evaluate_peak(
        self,
        waveform: torch.Tensor,
        peak_order: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the smooth local-AF value and exact local-AF peak.

        The correlation stage is shared with the inner update.  The returned
        tensors have shape ``[batch]`` and match
        ``LocalAmbiguityObjective.waveform_terms`` up to floating-point
        reduction order.
        """

        waveform = self._validate_waveform(waveform)
        if not torch.is_tensor(peak_order):
            peak_order = torch.tensor(
                float(peak_order),
                device=waveform.device,
                dtype=waveform.real.dtype,
            )
        if peak_order.ndim != 0:
            raise ValueError("peak_order must be a scalar tensor")
        if peak_order.device != waveform.device:
            raise ValueError("peak_order must be on the waveform device")
        if peak_order.dtype != waveform.real.dtype:
            peak_order = peak_order.to(dtype=waveform.real.dtype)

        self._launch_correlation(waveform)
        smooth_peak = torch.empty(
            (self.batch,), device=waveform.device, dtype=waveform.real.dtype
        )
        exact_peak = torch.empty_like(smooth_peak)
        block_cells = triton.next_power_of_2(self.cell_count)
        _local_af_peak_kernel[(self.batch,)](
            self.residual_real,
            self.residual_imag,
            peak_order,
            smooth_peak,
            exact_peak,
            LENGTH_SQUARED=float(self.length * self.length),
            CELLS=self.cell_count,
            ORIGIN_CELL=self.origin_cell,
            BLOCK_CELLS=block_cells,
            num_warps=4,
        )
        return smooth_peak, exact_peak


__all__ = ["TritonLocalAFUpdate"]
