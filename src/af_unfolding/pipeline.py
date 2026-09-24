"""Experiment orchestration with one fresh process per method and seed."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

METHODS = ('admm', 'aiso', 'qgd', 'consensus', 'wavenet', 'mal', 'ours')
LABELS = dict(zip(METHODS, ('ADMM', 'AISO', 'QGD', 'Consensus-ADMM', 'WaveNet', 'MAL-Net', 'Ours')))


def paper_seeds(method):
    if method == 'ours':
        return list(range(1860000, 1860032))
    if method == 'consensus':
        return [1860028]
    return [95000] if method == 'mal' else list(range(95000, 95008))


def run_worker(args):
    import numpy as np
    import torch
    from . import build_solver, initial_phase, psl_numpy, load_config
    from .core import LocalAFRegion, LocalAFWaveformDesign, LocalAmbiguityObjective
    from .baselines import classical as b
    from .baselines.consensus import PackedDopplerGradient
    from .baselines.networks import MALNetLocalAF
    from .baselines.online import WaveNetEngine, solve_mal, waveform

    torch.set_num_threads(4 if args.device == 'cpu' else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = args.device == 'cuda'
    sync = torch.cuda.synchronize if cuda else lambda: None
    method = args.worker
    phase = initial_phase(args.seed, args.device)
    before = torch.complex(phase.cos(), phase.sin())
    prefix = 2 if args.quick else 448
    reset = lambda: None
    start = time.perf_counter()
    if method in METHODS[:4]:
        configs = json.loads(files('af_unfolding').joinpath('data/baselines.json').read_text())
        types = dict(admm=b.PlainADMMConfig, aiso=b.AISOConfig, qgd=b.QGDConfig, consensus=b.ConsensusADMMConfig)
        functions = dict(admm=b.solve_af_plain_admm, aiso=b.solve_af_aiso, qgd=b.solve_af_qgd, consensus=b.solve_af_consensus_admm)
        config = types[method](**configs[method])
        obj = LocalAmbiguityObjective(256, LocalAFRegion.paper_exp2(), peak_order=8., implementation='dense').to(args.device)
        extra = {}
        if method == 'consensus':
            extra['gradient_operator'] = PackedDopplerGradient(obj, before.expand(9, -1, -1).clone())
        def solve():
            return functions[method](obj, phase, config, execution_layers=prefix, **extra)
        functions[method](obj, phase, config, execution_layers=min(prefix, 14), **extra)
        settings = asdict(config) | dict(execution_layers=prefix, gradient='paired_slice' if method == 'consensus' else 'dense_analytic')
    elif method == 'ours':
        backend = 'triton' if cuda else 'torch'
        model = build_solver(args.device, backend, graph=cuda, execution_layers=prefix)
        @torch.inference_mode()
        def solve():
            return model(phase)
        for _ in range(3):
            solve()
        settings = load_config() | dict(backend=backend, graph=cuda, execution_layers=prefix)
    elif method == 'wavenet':
        engine = WaveNetEngine(args.seed, args.device, graph=cuda)
        before = waveform(engine.y).detach()
        reset = engine.restore
        def solve():
            return engine.solve(max_steps=100 if args.quick else 30000)
        settings = dict(learning_rate=.001, min_steps=2000, max_steps=100 if args.quick else 30000,
                        plateau_block=100, plateau_blocks=10, tolerance_db=.005,
                        parameters=sum(p.numel() for p in engine.net.parameters()), graph=cuda)
    else:
        model = MALNetLocalAF(LocalAFWaveformDesign(256, 1), LocalAFRegion.paper_exp2(), layers=56).to(args.device).eval()
        with torch.inference_mode():
            model(phase)
        def solve():
            return solve_mal(model, phase, max_passes=2 if args.quick else 100)
        settings = dict(layers=56, max_passes=2 if args.quick else 100, patience=20, tolerance_db=.01,
                        step_start=.04, step_end=.008, weights='preset_steps')
    sync()
    setup_seconds = time.perf_counter() - start
    samples = []
    initial_wave = before.detach().cpu().numpy().copy()
    for repeat in range(args.repeats):
        reset()
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        result = solve()
        if cuda:
            end.record()
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        peak_mib = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
        wave = result.waveform.detach().cpu().numpy().copy()
        score = psl_numpy(wave)
        error = float(np.abs(np.abs(wave.astype(np.complex128)) - 1).max())
        score_error = score - float(result.local_af_psl_db)
        if not np.isfinite(wave).all() or error > 2e-5 or abs(score_error) > .002:
            raise RuntimeError(f'{method}: failed independent AF/constant-modulus validation')
        samples.append(dict(repetition=repeat + 1, wall_ms=elapsed,
                            cuda_event_ms=begin.elapsed_time(end) if cuda else None,
                            peak_allocated_mib=peak_mib, psl_db=score,
                            max_modulus_error=error, scorer_difference_db=score_error))
        if hasattr(result, 'history_db'):
            history = np.asarray(result.history_db)
            updates = result.updates
            stop = result.stop_reason
        else:
            history = 10 * np.log10(result.objective_history.detach().cpu().numpy().reshape(-1).clip(1e-30))
            updates, stop = prefix * 16, 'fixed_budget'
        np.savez_compressed(args.output / f'{method}_{args.seed}_r{repeat+1}.npz',
                            initial_waveform=initial_wave, waveform=wave, history_psl_db=history)
        del result
    row = dict(method=method, label=LABELS[method], seed=args.seed, device=args.device,
               quick=args.quick, updates=updates, stop_reason=stop,
               initial_psl_db=psl_numpy(initial_wave), psl_db=statistics.mean(s['psl_db'] for s in samples),
               wall_ms=statistics.mean(s['wall_ms'] for s in samples),
               cuda_event_ms=statistics.mean(s['cuda_event_ms'] for s in samples) if cuda else None,
               setup_seconds=setup_seconds, settings=settings, samples=samples,
               environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                device=torch.cuda.get_device_name() if cuda else 'CPU'))
    (args.output / f'{method}_{args.seed}.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
    print(f"{LABELS[method]:16s} seed={args.seed} PSL={row['psl_db']:.5f} dB time={row['wall_ms']:.2f} ms updates={updates}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Seven-method local-AF experiment pipeline')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--protocol', choices=('example', 'paper'), default='example')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--quick', action='store_true', help='Reduced-budget pipeline smoke check')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.protocol == 'paper' and args.seed is not None:
        parser.error('--seed applies to --protocol example; paper uses recorded seed sets')
    if len(set(args.methods)) != len(args.methods):
        parser.error('--methods must not contain duplicates')
    import torch
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; install a CUDA PyTorch build or select --device cpu')
    if args.output is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.output = Path('outputs') / f'{args.protocol}_{"quick" if args.quick else "full"}_{stamp}'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args)
        return
    # Fail before expensive solves if the plotting dependency is absent.
    from .plotting import plot_results
    seeds = {m: paper_seeds(m) if args.protocol == 'paper' else [args.seed if args.seed is not None else 1860028] for m in args.methods}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=args.protocol,
                    quick=args.quick, device=args.device, methods=args.methods, seeds=seeds, repeats=args.repeats,
                    status='running', timing_scope='warm device-resident solve, including iteration and stopping logic; excludes setup, JIT/capture, reset, transfers, independent scoring and file I/O',
                    metric='10 log10 max |AF/N|^2 over delays -10..10, Dopplers -4..4, excluding origin')
    path = args.output / 'run.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
    rows = []
    try:
        for method in args.methods:
            for seed in seeds[method]:
                print(f'Running {LABELS[method]}, seed {seed} ...', flush=True)
                cmd = [sys.executable, '-m', 'af_unfolding.pipeline', '--worker', method,
                       '--seed', str(seed), '--device', args.device, '--output', str(args.output),
                       '--repeats', str(args.repeats)]
                if args.quick:
                    cmd.append('--quick')
                subprocess.run(cmd, env=env, check=True)
                rows.append(json.loads((args.output / f'{method}_{seed}.json').read_text()))
        summary = []
        for method in args.methods:
            group = [r for r in rows if r['method'] == method]
            summary.append(dict(method=method, label=LABELS[method], seeds=len(group),
                                psl_db=statistics.mean(r['psl_db'] for r in group),
                                wall_ms=statistics.mean(r['wall_ms'] for r in group),
                                cuda_event_ms=statistics.mean(r['cuda_event_ms'] for r in group) if args.device=='cuda' else None,
                                mean_updates=statistics.mean(r['updates'] for r in group)))
        with (args.output / 'results.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        (args.output / 'results.json').write_text(json.dumps(dict(summary=summary, runs=rows), indent=2)+'\n', encoding='utf-8')
        table = ['| Method | Seeds | PSL (dB) | Solve time (ms) |', '|---|---:|---:|---:|']
        table += [f"| {r['label']} | {r['seeds']} | {r['psl_db']:.2f} | {r['wall_ms']:.2f} |" for r in summary]
        title = 'Reduced-budget smoke check' if args.quick else f"{args.protocol.capitalize()} protocol / {args.device}"
        (args.output / 'table.md').write_text(title+'\n\n'+'\n'.join(table)+'\n', encoding='utf-8')
        latex = [r'\begin{tabular}{lrr}', r'\hline', r'Method & PSL (dB) & Solve time (ms) \\', r'\hline']
        latex += [f"{r['label']} & {r['psl_db']:.2f} & {r['wall_ms']:.2f} " + r'\\' for r in summary]
        latex += [r'\hline', r'\end{tabular}']
        (args.output / 'table.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
        plot_results(args.output, summary, rows, quick=args.quick)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('\n'+'\n'.join(table), flush=True)
    print(f'Outputs: {args.output}', flush=True)


if __name__ == '__main__':
    main()
