"""WaveNet and MAL-Net local-AF adaptations selected for the comparison.

MAL-Net uses the recorded preset step schedule. Its factor-two phase update
is retained as part of that schedule's step-size convention.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from ..core import LocalAFWaveformDesign, LocalAFRegion, LocalAmbiguityObjective

def make_objective(length=256, *, dtype=torch.float32, device='cpu'):
    return LocalAmbiguityObjective(length, LocalAFRegion.paper_exp2(), implementation='dense').to(device=device, dtype=dtype)

def waveform(normalized_phase: torch.Tensor) -> torch.Tensor:
    phase = normalized_phase.reshape(normalized_phase.shape[0], -1, 1) * (2 * math.pi)
    return torch.complex(phase.cos(), phase.sin())

class WaveNetResidualBlock(nn.Module):

    def __init__(self, width=256):
        super().__init__()
        self.fc1, self.fc2 = (nn.Linear(width, width), nn.Linear(width, width))

    def forward(self, x):
        return torch.relu(x + torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x)))))

class WaveNetPhaseNetwork(nn.Module):
    """Six 256x256 FC layers and the two residual blocks in Fig. 2.

    The output sigmoid is interpreted as normalized phase; physical phases
    cover [0, 2*pi), not [0, 1] radians. This range convention is an explicit
    adaptation because the available diagram omits the scale factor.
    """

    def __init__(self, length=256):
        super().__init__()
        self.input = nn.Linear(length, length)
        self.blocks = nn.Sequential(WaveNetResidualBlock(length), WaveNetResidualBlock(length))
        self.output = nn.Linear(length, length)

    def forward(self, y):
        return torch.sigmoid(self.output(self.blocks(torch.sigmoid(self.input(y)))))

class MALNetLocalAF(nn.Module):
    """MAL-Net-style CCM gradient descent adapted to the local AF objective."""

    def __init__(self, design: LocalAFWaveformDesign, region: LocalAFRegion, layers: int=56):
        super().__init__()
        self.design = design
        self.layers = int(layers)
        self.objective = LocalAmbiguityObjective(design.length, region, peak_order=16.0, implementation='dense')
        self.log_steps = nn.Parameter(torch.linspace(math.log(0.04), math.log(0.008), self.layers))
        self.register_buffer('peak_orders', torch.logspace(math.log10(8.0), math.log10(128.0), self.layers), persistent=False)

    def forward(self, initial_phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if initial_phase.ndim == 2:
            initial_phase = initial_phase.unsqueeze(0)
        current = initial_phase
        history = []
        for layer in range(self.layers):
            waveform = torch.complex(torch.cos(current), torch.sin(current))
            _, peak, complex_gradient = self.objective.dense_waveform_terms_and_gradient(waveform, self.peak_orders[layer])
            phase_gradient = 2.0 * (waveform.real * complex_gradient.imag - waveform.imag * complex_gradient.real)
            step = self.log_steps[layer].exp().clamp(1e-06, 0.1)
            current = torch.remainder(current - step * phase_gradient, 2.0 * torch.pi)
            history.append(peak.detach())
        waveform = torch.complex(torch.cos(current), torch.sin(current))
        _, final_peak = self.objective.waveform_terms(waveform, self.peak_orders[-1])
        return (current, waveform, final_peak)
