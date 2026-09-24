"""Run in a fresh process to measure peak allocated tensor memory."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for p in [ROOT/'.runtime/cache',ROOT/'.runtime/temp']:p.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.runtime/cache'))
if os.name=='nt':os.environ['TEMP']=os.environ['TMP']=str(ROOT/'.runtime/temp')
import torch
from af_unfolding import build_solver,initial_phase
torch.set_num_threads(1)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
phase=initial_phase(1860028,'cuda')
solver=build_solver('cuda','triton',graph=True)
with torch.inference_mode():result=solver(phase)
torch.cuda.synchronize()
record=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            allocated_after_solve_mib=torch.cuda.memory_allocated()/2**20,
            scope='Fresh process; input + solver setup + warmup/capture + first complete replay. No separate GPU scorer. Reserved memory is a distinct allocator statistic.')
out=ROOT/'outputs/memory.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
