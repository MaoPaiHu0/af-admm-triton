"""Independent mathematical checks and final-path GPU regression checks."""
from pathlib import Path
import numpy as np
import pytest
import torch
from af_unfolding import build_solver, initial_phase, load_config, ambiguity_numpy, psl_numpy
from af_unfolding.core import LocalAFRegion,LocalAmbiguityObjective
from af_unfolding.triton_kernels import TritonLocalAFUpdate

CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')

def direct_torch(z, ds, ks, q):
    # Deliberately use slice sums and complex exp, not the implementation tables.
    n=z.shape[1]; vals=[]
    for d in ds:
        for k in ks:
            if d==0 and k==0:
                continue
            lo,hi=max(0,-k),min(n,n-k)
            ix=torch.arange(lo,hi,device=z.device,dtype=z.real.dtype)
            r=(z[:,lo+k:hi+k,:]*z[:,lo:hi,:].conj()*
               torch.exp(-2j*torch.pi*d*ix/n)[None,:,None]).sum(1)
            vals.append((r.abs().square()/n**2).flatten())
    p=torch.stack(vals,dim=1)
    return torch.logsumexp(q*p.clamp_min(torch.finfo(p.dtype).tiny).log(),dim=1)/q

@pytest.mark.parametrize('n,ds,ks',[(13,(-2.,-.5,0.,1.5),(-12,-3,0,2,12)),(17,(-2.,0.,2.),(-4,0,3))])
def test_af_against_independent_complex128_sums(n,ds,ks):
    gen=torch.Generator().manual_seed(450)
    z=torch.complex(torch.randn(1,n,1,generator=gen,dtype=torch.float64),torch.randn(1,n,1,generator=gen,dtype=torch.float64))
    obj=LocalAmbiguityObjective(n,LocalAFRegion(ds,ks),implementation='dense')
    actual=obj.ambiguity_waveform(z)[0,:,:,0].numpy()
    np.testing.assert_allclose(actual,ambiguity_numpy(z.numpy(),ds,ks),atol=2e-12,rtol=2e-12)
    _,peak=obj.waveform_terms(z)
    assert abs(float(10*peak.log10())-psl_numpy(z.numpy(),ds,ks))<1e-11

@pytest.mark.parametrize('q',[8.,128.])
def test_analytic_gradient_vs_independent_autograd_and_directional_difference(q):
    torch.manual_seed(902)
    z=torch.randn(1,13,1,dtype=torch.complex128,requires_grad=True)
    ds=(-2.,-.5,0.,1.5);ks=(-3,-1,0,2)
    obj=LocalAmbiguityObjective(13,LocalAFRegion(ds,ks),implementation='dense')
    value,_,g=obj.dense_waveform_terms_and_gradient(z,q)
    ref=direct_torch(z,ds,ks,q)
    expected=torch.autograd.grad(ref.sum(),z)[0]
    torch.testing.assert_close(value,ref,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(g,expected,atol=3e-11,rtol=3e-10)
    direction=torch.randn_like(z);direction/=torch.linalg.vector_norm(direction)
    eps=1e-6
    fd=(direct_torch(z+eps*direction,ds,ks,q)-direct_torch(z-eps*direction,ds,ks,q))/(2*eps)
    inner=(g.conj()*direction).real.sum()
    torch.testing.assert_close(fd.squeeze(),inner,atol=2e-7,rtol=2e-6)

def test_preset_and_independent_cpu_recurrence():
    model=build_solver('cpu','torch',graph=False,execution_layers=2)
    c=load_config();p=c['parameters'];s=model.scalar_schedules()
    assert c['outer_iterations']*c['inner_updates']==7168
    assert model.layer_peak_orders.shape==(448,)
    assert float(model.layer_peak_orders[0])==8
    # CUDA float32 logspace's last ramp element is 127.99998474121094.
    assert abs(float(model.layer_peak_orders[55])-128)<2e-5
    assert bool((model.layer_peak_orders[56:]==128).all())
    assert sum(x.numel() for x in model.parameters())==32
    for key,value in [('af_steps',p['alpha']),('effective_rhos',p['rho']),('momenta',p['mu']),('dual_scales',p['gamma'])]:
        torch.testing.assert_close(s[key],torch.full_like(s[key],value),atol=2e-8,rtol=2e-7)
    phase=initial_phase(1860028)
    x=torch.complex(torch.cos(phase),torch.sin(phase));z=x.clone();u=torch.zeros_like(z)
    best=x.clone();best_psl=psl_numpy(best.numpy())
    # A separate recurrence, using an independently differentiated AF objective.
    for outer in range(2):
        v=torch.zeros_like(z)
        for _ in range(16):
            zz=z.detach().requires_grad_(True)
            grad=torch.autograd.grad(direct_torch(zz,range(-4,5),range(-10,11),model.layer_peak_orders[outer]).sum(),zz)[0]
            v=s['momenta'][0]*v+s['af_steps'][0]*grad+s['consensus_steps'][0]*(z-x+u)
            z=(z-v).detach();v=v.detach()
        a=z+u;x=a/a.abs();u=u+s['dual_scales'][0]*(z-x)
        score=psl_numpy(x.numpy())
        if score<best_psl:best=x.clone();best_psl=score
    with torch.inference_mode():actual=model(phase)
    torch.testing.assert_close(actual.waveform,best,atol=3e-5,rtol=3e-5)
    assert abs(psl_numpy(actual.waveform.numpy())-best_psl)<2e-3

@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('n,q',[(17,8.),(17,128.),(256,8.),(256,128.)])
def test_fused_k1_k2_k3_against_independent_af_and_gradient(n,q):
    torch.manual_seed(7)
    ds=tuple(float(x) for x in range(-4,5));ks=tuple(range(-10,11))
    z=(torch.randn(1,n,1,dtype=torch.complex64,device='cuda')*.2+1).requires_grad_(True)
    x=torch.exp(1j*torch.randn_like(z.real));u=torch.randn_like(z)*.01;v=torch.randn_like(z)*.02
    g=torch.autograd.grad(direct_torch(z,ds,ks,q).sum(),z)[0]
    p=load_config()['parameters'];alpha=torch.tensor(p['alpha'],device='cuda'); beta=torch.tensor(p['alpha']*p['rho'],device='cuda');mu=torch.tensor(p['mu'],device='cuda')
    operator=TritonLocalAFUpdate(n,ds,ks).cuda()
    out,vel=operator.operator_aware_update(z.detach(),x,u,v,alpha,beta,mu,torch.tensor(q,device='cuda'),fuse_momentum=True)
    candidate=z.detach()-alpha*(g+(beta/alpha)*(z.detach()-x+u))
    expected_v=mu*v+(z.detach()-candidate)
    expected_z=z.detach()-expected_v
    torch.testing.assert_close(vel,expected_v,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(out,expected_z,atol=2e-5,rtol=2e-5)
    corr=torch.complex(operator.residual_real,operator.residual_imag).cpu().numpy().reshape(len(ds),len(ks))
    np.testing.assert_allclose(corr,ambiguity_numpy(z.detach().cpu().numpy(),ds,ks),atol=1e-4,rtol=2e-5)
    assert out.data_ptr()!=z.data_ptr() and vel.data_ptr()!=v.data_ptr()

@CUDA
@pytest.mark.cuda
def test_projection_and_dual_including_zero_input():
    op=TritonLocalAFUpdate(256,tuple(float(x) for x in range(-4,5)),tuple(range(-10,11))).cuda()
    torch.manual_seed(89)
    z=torch.randn(1,256,1,dtype=torch.complex64,device='cuda')
    u=torch.randn_like(z)*.02
    z[0,0,0]=0;u[0,0,0]=0
    z[0,1,0]=complex(1e-15,-1e-15);u[0,1,0]=0
    scale=torch.tensor(.45,device='cuda')
    x,dual=op.project_and_update_dual(z,u,scale,projection_mode=3)
    torch.testing.assert_close(x.abs(),torch.ones_like(x.real),atol=3e-7,rtol=0)
    torch.testing.assert_close(dual,u+scale*(z-x),atol=2e-7,rtol=2e-6)
    _,peak=op.evaluate_peak(x,torch.tensor(128.,device='cuda'))
    assert abs(float(10*peak.log10())-psl_numpy(x.cpu().numpy()))<5e-4

@CUDA
@pytest.mark.cuda
def test_graph_matches_eager_and_resets_state_for_new_input():
    model=build_solver('cuda','triton',graph=False,execution_layers=3)
    graph=build_solver('cuda','triton',graph=True,execution_layers=3)
    a,b=initial_phase(1860000,'cuda'),initial_phase(1860028,'cuda')
    with torch.inference_mode():
        expected=model(a).waveform.clone();one=graph(a).waveform.clone()
        graph(b);two=graph(a).waveform.clone()
    torch.testing.assert_close(one,expected,atol=0,rtol=0)
    torch.testing.assert_close(two,one,atol=0,rtol=0)

def test_saved_af_figure_matches_independent_direct_sums():
    path=Path(__file__).resolve().parents[1]/'data/af_example.npz'
    with np.load(path,allow_pickle=False) as f:
        for side in ('before','after'):
            expected=ambiguity_numpy(f[side+'_waveform'],f['dopplers'],f['delays'])
            np.testing.assert_allclose(expected,f[side+'_af'],atol=1e-10,rtol=1e-10)
        assert abs(psl_numpy(f['after_waveform'])-(-51.15729256501993))<1e-8
