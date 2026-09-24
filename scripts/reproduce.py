"""Reproduce final solver quality; timings exclude setup/compilation/capture."""
from pathlib import Path
import argparse
import csv
import json
import os
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
runtime=ROOT/'.runtime'
for p in [runtime/'cache',runtime/'temp']:p.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR',str(runtime/'cache'))
if os.name=='nt':
    os.environ['TEMP']=os.environ['TMP']=str(runtime/'temp')
import numpy as np
import torch
from af_unfolding import build_solver,initial_phase,psl_numpy,load_config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--backend',default='triton',choices=['torch','triton'])
    parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--all-seeds',action='store_true')
    parser.add_argument('--seed',type=int,default=1860028)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    torch.set_num_threads(4 if args.device=='cpu' else 1)
    torch.set_num_interop_threads(1)
    start=time.perf_counter()
    solver=build_solver(args.device,args.backend,graph=not args.no_graph)
    setup=time.perf_counter()-start
    sync=torch.cuda.synchronize if args.device=='cuda' else lambda:None
    seeds=list(range(1860000,1860032)) if args.all_seeds else [args.seed]
    with (ROOT/'data/paper_reference_results.csv').open(newline='',encoding='utf-8-sig') as f:
        expected={int(r['seed']):float(r['psl_db_float64']) for r in csv.DictReader(f)}
    rows=[];args.output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        # Graph capture already warms up once; these warmups are also untimed.
        phase=initial_phase(seeds[0],args.device)
        for _ in range(3):solver(phase)
        sync()
        for seed in seeds:
            phase=initial_phase(seed,args.device)
            before=torch.complex(torch.cos(phase),torch.sin(phase)).cpu().numpy()
            wall=[];gpu=[]
            for _ in range(args.repeats):
                sync()
                if args.device=='cuda':
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                ts=time.perf_counter();result=solver(phase)
                if args.device=='cuda':b.record()
                sync();wall.append((time.perf_counter()-ts)*1000)
                if args.device=='cuda':gpu.append(a.elapsed_time(b))
            wave=result.waveform.detach().cpu().numpy().copy()
            history=result.objective_history.detach().cpu().numpy().copy()
            score=psl_numpy(wave)
            row=dict(seed=seed,initial_psl_db=psl_numpy(before),psl_db=score,
                     reference_psl_db=expected.get(seed),
                     reference_difference_db=None if seed not in expected else score-expected[seed],
                     scorer_difference_db=score-float(result.local_af_psl_db),
                     max_modulus_error=float(np.max(np.abs(np.abs(wave.astype(np.complex128))-1))),
                     final_iterate_psl_db=float(10*np.log10(history[0,-1])),
                     best_outer=int(np.argmin(history[0])),wall_ms=wall,cuda_event_ms=gpu)
            rows.append(row)
            np.savez_compressed(args.output/f'seed_{seed}.npz',phase=phase.cpu().numpy(),waveform=wave,history=history)
            print(json.dumps(row),flush=True)
    report=dict(config=load_config(),device=args.device,backend=args.backend,graph=not args.no_graph,
                torch=torch.__version__,cuda=torch.version.cuda,setup_seconds=setup,
                mean_psl_db=statistics.mean(r['psl_db'] for r in rows),
                mean_improvement_db=statistics.mean(r['initial_psl_db']-r['psl_db'] for r in rows),
                mean_wall_ms=statistics.mean(x for r in rows for x in r['wall_ms']),
                mean_cuda_event_ms=statistics.mean(x for r in rows for x in r['cuda_event_ms']) if args.device=='cuda' else None,
                timing_scope='warm device-resident solve, graph input copy included; excludes compilation, capture, transfers, scoring and file I/O',rows=rows)
    (args.output/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
