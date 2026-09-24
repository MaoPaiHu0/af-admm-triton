"""Paired Doppler-slice analytic gradient used by the selected Consensus-ADMM."""
import torch

class PackedDopplerGradient:
    """Pair local waveform b with Doppler slice b; batch the nine pairs."""

    def __init__(self, objective, example):
        self.objective = objective
        self.shape = tuple(example.shape)
        self.dtype, self.device = (example.dtype, example.device)
        self.blocks, self.length, channels = self.shape
        if channels != 1 or self.blocks != objective.keep_mask.shape[0]:
            raise ValueError('Requires one local waveform per Doppler slice and one channel.')
        self.reference_index = objective.dense_reference_index
        self.shifted_index = objective.dense_shifted_index
        self.overlap = objective.dense_overlap_mask[None, :, :, None]
        self.reference_for_shifted = objective.gradient_reference_for_shifted_index
        self.shifted_for_reference = objective.gradient_shifted_for_reference_index
        self.shifted_valid = objective.gradient_shifted_valid[None, :, :, None]
        self.reference_valid = objective.gradient_reference_valid[None, :, :, None]
        self.keep = objective.keep_mask[:, :, None]
        self.steering = objective._dense_steering(example)[:, :, :, None]
        left, right = objective._gradient_steering(example)
        self.left_steering_conj = left[:, :, :, None].conj().resolve_conj()
        self.right_steering = right[:, :, :, None]
        self.tiny = torch.finfo(example.real.dtype).tiny
        self.norm = float(self.length * self.length)

    @torch.no_grad()
    def __call__(self, local, q):
        if tuple(local.shape) != self.shape or local.dtype != self.dtype or local.device != self.device:
            raise ValueError('The prepared operator must match the local states.')
        reference = local[:, self.reference_index, :]
        shifted = local[:, self.shifted_index, :]
        product = shifted * reference.conj() * self.overlap
        ambiguity = (self.steering * product).sum(dim=2)
        power = ambiguity.abs().square() / self.norm
        safe_power = power.clamp_min(self.tiny)
        logits = (float(q) * safe_power.log()).masked_fill(~self.keep, -torch.inf)
        weights = torch.softmax(logits.flatten(1), dim=1).reshape_as(power)
        coefficient = 2.0 * weights * ambiguity / (self.norm * safe_power)
        shifted_samples = local[:, self.reference_for_shifted, :] * self.shifted_valid
        reference_samples = local[:, self.shifted_for_reference, :] * self.reference_valid
        expanded = coefficient[:, :, None, :]
        contribution = expanded * shifted_samples * self.left_steering_conj + expanded.conj() * reference_samples * self.right_steering
        return contribution.sum(dim=1)
