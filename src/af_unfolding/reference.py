"""Independent CPU complex128 AF: direct valid-overlap sums, no solver helpers."""
import numpy as np

def ambiguity_numpy(waveform,dopplers=range(-4,5),delays=range(-10,11)):
    z=np.asarray(waveform,dtype=np.complex128).reshape(-1)
    n=z.size
    out=np.empty((len(dopplers),len(delays)),dtype=np.complex128)
    for j,k in enumerate(delays):
        ix=np.arange(max(0,-k),min(n,n-k))
        product=z[ix+k]*z[ix].conj()
        for i,d in enumerate(dopplers):
            out[i,j]=np.sum(product*np.exp(-2j*np.pi*d*ix/n))
    return out

def psl_numpy(waveform,dopplers=range(-4,5),delays=range(-10,11)):
    a=ambiguity_numpy(waveform,dopplers,delays)
    p=np.abs(a)**2/np.asarray(waveform).size**2
    keep=~((np.asarray(dopplers)[:,None]==0)&(np.asarray(delays)[None,:]==0))
    return float(10*np.log10(max(float(p[keep].max()),np.finfo(np.float64).tiny)))
