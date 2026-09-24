"""Independent baseline gradient checks and selected-record regression."""
from pathlib import Path
import json
import pytest
import torch
from af_unfolding import initial_phase, psl_numpy
from af_unfolding.core import LocalAFRegion, LocalAFWaveformDesign, LocalAmbiguityObjective
from af_unfolding.baselines import classical as b
from af_unfolding.baselines.consensus import PackedDopplerGradient
from af_unfolding.baselines.networks import MALNetLocalAF, WaveNetPhaseNetwork
from af_unfolding.baselines.online import WaveNetEngine, solve_mal
from test_correctness import direct_torch

ROOT=Path(__file__).resolve().parents[1]
CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


@pytest.mark.parametrize('q',[8.,128.])
def test_baseline_gradient_from_independent_af_definition(q):
    torch.manual_seed(817)
    z=torch.randn(1,17,1,dtype=torch.complex128,requires_grad=True)
    ds=(-2.,0.,2.);ks=(-4,-1,0,3)
    objective=LocalAmbiguityObjective(17,LocalAFRegion(ds,ks),implementation='dense')
    value,_,actual,_=b.weighted_af_terms_and_gradient(objective,z,q)
    expected_value=direct_torch(z,ds,ks,q)
    expected=torch.autograd.grad(expected_value.sum(),z)[0]
    torch.testing.assert_close(value,expected_value,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual,expected,atol=2e-11,rtol=2e-10)


@pytest.mark.parametrize('q',[8.,128.])
def test_paired_consensus_gradient_against_independent_subproblems(q):
    torch.manual_seed(993)
    ds=(-2.,0.,2.);ks=(-4,-1,0,3)
    z=torch.randn(3,17,1,dtype=torch.complex128,requires_grad=True)
    objective=LocalAmbiguityObjective(17,LocalAFRegion(ds,ks),implementation='dense')
    actual=PackedDopplerGradient(objective,z)(z,q)
    # Each copy has a separate loss and normalization over its own slice.
    value=sum(direct_torch(z[i:i+1],(d,),ks,q).sum() for i,d in enumerate(ds))
    expected=torch.autograd.grad(value,z)[0]
    torch.testing.assert_close(actual,expected,atol=3e-11,rtol=3e-10)


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('row',json.loads((ROOT/'data/baseline_regression.json').read_text()),ids=lambda r:r['method'])
def test_selected_cuda_prefix_records(row):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    phase=initial_phase(row['seed'],'cuda')
    method=row['method']
    if method in ('admm','aiso','qgd','consensus'):
        configs=json.loads((ROOT/'src/af_unfolding/data/baselines.json').read_text())
        types=dict(admm=b.PlainADMMConfig,aiso=b.AISOConfig,qgd=b.QGDConfig,consensus=b.ConsensusADMMConfig)
        functions=dict(admm=b.solve_af_plain_admm,aiso=b.solve_af_aiso,qgd=b.solve_af_qgd,consensus=b.solve_af_consensus_admm)
        obj=LocalAmbiguityObjective(256,LocalAFRegion.paper_exp2(),peak_order=8.,implementation='dense').cuda()
        result=functions[method](obj,phase,types[method](**configs[method]),execution_layers=row['budget'])
    elif method=='wavenet':
        engine=WaveNetEngine(row['seed'],'cuda')
        result=engine.solve(max_steps=row['budget'])
        first=result.waveform.clone()
        engine.restore()
        result=engine.solve(max_steps=row['budget'])
        torch.testing.assert_close(first,result.waveform,atol=0,rtol=0)
        assert sum(p.numel() for p in engine.net.parameters())==394752
    else:
        model=MALNetLocalAF(LocalAFWaveformDesign(256,1),LocalAFRegion.paper_exp2(),layers=56).cuda().eval()
        result=solve_mal(model,phase,max_passes=row['budget'])
    wave=result.waveform.detach().cpu().numpy()
    assert abs(psl_numpy(wave)-row['psl_db'])<.002
    torch.testing.assert_close(result.waveform.abs(),torch.ones_like(result.waveform.real),atol=5e-7,rtol=0)
