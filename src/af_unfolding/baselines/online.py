"""Per-instance WaveNet optimization and repeated MAL-Net inference."""
from types import SimpleNamespace
import torch
from .networks import WaveNetPhaseNetwork, make_objective, waveform


def initial_y(seed):
    phase = 2 * torch.pi * torch.rand((1, 256, 1), generator=torch.Generator().manual_seed(seed))
    return phase.reshape(1, 256) / (2 * torch.pi)


class WaveNetEngine:
    """Restorable online Adam state; capture and warmup are solver setup."""

    def __init__(self, seed, device='cuda', graph=True):
        self.seed, self.device = seed, torch.device(device)
        torch.manual_seed(seed + 100000)
        cpu = WaveNetPhaseNetwork()
        self.initial_state = {k: v.clone() for k, v in cpu.state_dict().items()}
        self.net = cpu.to(self.device)
        self.obj = make_objective(device=self.device)
        cuda = self.device.type == 'cuda'
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, capturable=cuda, fused=cuda)
        self.y = initial_y(seed).to(self.device)
        self.best_y = self.y.clone()
        self.best_peak = torch.ones(1, device=self.device)
        self.base_peak = self.best_peak.clone()
        self.block_sum = torch.zeros(1, device=self.device)
        self.loss_value = torch.zeros(1, device=self.device)
        self.graph = None
        self.restore()
        if cuda:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.restore()
            if graph:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.step()
        else:
            for _ in range(3):
                self.step()
        self.restore()
        if cuda:
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def restore(self):
        self.net.load_state_dict(self.initial_state)
        for values in self.opt.state.values():
            for value in values.values():
                if torch.is_tensor(value):
                    value.zero_()
        self.opt.zero_grad(set_to_none=False)
        self.y.copy_(initial_y(self.seed))
        self.best_y.copy_(self.y)
        self.base_peak.copy_(self.obj.waveform_terms(waveform(self.y))[1])
        self.best_peak.copy_(self.base_peak)
        self.block_sum.zero_()
        self.loss_value.zero_()

    def step(self):
        self.opt.zero_grad(set_to_none=False)
        prediction = self.net(self.y)
        _, peak = self.obj.waveform_terms(waveform(prediction), 128)
        loss = peak.sum() * 256**2
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            better = peak < self.best_peak
            self.best_y.copy_(torch.where(better[:, None], prediction, self.best_y))
            self.best_peak.copy_(torch.minimum(peak, self.best_peak))
            self.loss_value.copy_(loss)
            self.block_sum.add_(loss)
            self.y.copy_(prediction)

    def solve(self, *, max_steps=30000, min_steps=2000):
        previous_db = float(10 * self.base_peak.log10())
        history = [previous_db]
        quiet, steps = 0, 0
        reason = 'maximum_steps'
        for start in range(0, max_steps, 100):
            self.block_sum.zero_()
            count = min(100, max_steps - start)
            for _ in range(count):
                self.step() if self.graph is None else self.graph.replay()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            steps += count
            db = float(10 * self.best_peak.log10())
            history.append(db)
            quiet = quiet + 1 if abs(previous_db - db) < .005 else 0
            previous_db = db
            if steps >= min_steps and quiet >= 10:
                reason = 'PSL_plateau_1000_steps'
                break
        return SimpleNamespace(waveform=waveform(self.best_y).detach(),
                               local_af_psl_db=10 * self.best_peak.log10(),
                               history_db=history, updates=steps, stop_reason=reason)


@torch.inference_mode()
def solve_mal(model, phase, *, max_passes=100, patience=20, tolerance_db=.01):
    current = phase.detach().clone()
    best = current.clone()
    _, peak = model.objective.waveform_terms(torch.complex(current.cos(), current.sin()))
    best_db = float((10 * peak.clamp_min(1e-30).log10()).mean())
    history, stale = [best_db], 0
    reason = 'maximum_passes'
    for index in range(max_passes):
        next_phase, _, final_peak = model(current)
        current = next_phase.detach().clone()
        db = float((10 * final_peak.clamp_min(1e-30).log10()).mean())
        history.append(db)
        if db < best_db - tolerance_db:
            best_db, best, stale = db, current.clone(), 0
        else:
            stale += 1
        if stale >= patience:
            reason = 'PSL_plateau_20_passes'
            break
    return SimpleNamespace(waveform=torch.complex(best.cos(), best.sin()),
                           local_af_psl_db=best_db, history_db=history,
                           updates=(index + 1) * model.layers, stop_reason=reason)
