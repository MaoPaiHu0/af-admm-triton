"""Portable entry points for the exact final-paper configuration."""
from importlib.resources import files
import json
import math
import numpy as np
import torch
from .core import (LocalAFRegion,LocalAFWaveformDesign,
                   OperatorAwarePureLocalAFADMMNet,CapturedPureLocalAFADMM)

def load_config():
    return json.loads(files('af_unfolding').joinpath('data/paper_final.json').read_text(encoding='utf-8'))

def initial_phase(seed=1860028, device='cpu'):
    generator=torch.Generator(device='cpu').manual_seed(int(seed))
    return (2*torch.pi*torch.rand((1,256,1),generator=generator,dtype=torch.float32)).to(device)

def build_solver(device='cuda', backend='triton', *, graph=True,
                 execution_layers=None, select_best=True):
    """Build an inference-only solver; graph outputs are reused on each call.

    CPU: backend='torch', graph=False. The released preset is single-waveform,
    batch=1, length=256. execution_layers is a prefix for verification only.
    Clone outputs if retaining them across graph replays.
    """
    device=torch.device(device)
    if backend not in ('torch','triton'):
        raise ValueError("backend must be 'torch' or 'triton'")
    if device.type!='cuda' and (graph or backend=='triton'):
        raise ValueError('Triton and CUDA Graph require a CUDA device')
    c=load_config(); p=c['parameters']; fused=backend=='triton'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=OperatorAwarePureLocalAFADMMNet(
        LocalAFWaveformDesign(c['length'],1),LocalAFRegion.paper_exp2(),
        outer_layers=c['outer_iterations'],inner_updates=c['inner_updates'],
        execution_layers=execution_layers,parameter_stages=c['parameter_stages'],
        initial_rho=.2,final_rho=.4,initial_af_step=.08,final_af_step=.03,
        initial_momentum=0,final_momentum=0,peak_order=8,final_peak_order=128,
        af_implementation='dense',gradient_implementation='triton' if fused else 'analytic',
        triton_batch=1,outer_implementation='triton' if fused else 'torch',
        triton_projection_mode=c['projection_mode'],fuse_momentum=fused,
        select_best=select_best).to(device).eval()
    with files('af_unfolding').joinpath('data/q_schedule.npy').open('rb') as f:
        schedule=np.load(f,allow_pickle=False).copy()
    with torch.no_grad():
        model.log_af_steps.fill_(math.log(p['alpha']))
        model.log_consensus_steps.fill_(math.log(p['alpha']*p['rho']))
        model.raw_momenta.fill_(math.atanh(p['mu']/model.max_abs_momentum))
        model.log_dual_scales.fill_(math.log(p['gamma']))
        indices=tuple(min(7,i*8//56) for i in range(448))
        model._layer_stage_indices=indices
        model.layer_stage_index.copy_(torch.tensor(indices,device=device))
        model.layer_peak_orders.copy_(torch.from_numpy(schedule).to(device))
    if graph:
        # Triton compilation and graph capture take place before solve timing.
        with torch.cuda.device(device):
            return CapturedPureLocalAFADMM(model,initial_phase(device=device),warmup=1)
    return model

@torch.inference_mode()
def solve(phase, solver=None):
    if phase.dtype!=torch.float32 or tuple(phase.shape)!=(1,256,1):
        raise ValueError('Final preset requires float32 phase with shape [1,256,1]')
    if not torch.isfinite(phase).all():
        raise ValueError('Phase must be finite')
    if solver is None:
        solver=build_solver(phase.device,backend='triton' if phase.is_cuda else 'torch',graph=phase.is_cuda)
    return solver(phase)
