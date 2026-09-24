"""Figures generated from the current run's independently scored waveforms."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .reference import ambiguity_numpy

STYLE = dict(admm=('#555555', 's'), aiso=('#7b6aa8', 'D'), qgd=('#bb8844', 'v'),
             consensus=('#298c8c', '^'), wavenet=('#529a44', 'o'), mal=('#416db2', 'P'), ours=('#c93636', '*'))


def plot_results(directory, summary, rows, *, quick=False):
    directory = Path(directory)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'axes.labelcolor': 'black', 'text.color': 'black',
                         'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout='constrained')
    for row in summary:
        color, marker = STYLE[row['method']]
        axes[0].scatter(row['wall_ms'], row['psl_db'], marker=marker, color=color,
                        s=100 if row['method']=='ours' else 45, label=row['label'], zorder=3)
    axes[0].set(xscale='log', xlabel='Solve time (ms)', ylabel='Local PSL (dB)')
    axes[0].grid(which='major', alpha=.3, linewidth=.6)
    axes[0].legend(fontsize=8, frameon=True, loc='best')
    positions = np.arange(len(summary))
    bars = axes[1].bar(positions, [r['wall_ms'] for r in summary],
                       color=[STYLE[r['method']][0] for r in summary], width=.63)
    axes[1].set_xticks(positions, [r['label'] for r in summary], rotation=40, ha='right', fontsize=8)
    axes[1].set_ylabel('Solve time (ms)')
    axes[1].set_ylim(0, max(r['wall_ms'] for r in summary)*1.18)
    axes[1].bar_label(bars, labels=[f"{r['wall_ms']:.1f}" for r in summary], fontsize=7, padding=3)
    axes[1].grid(axis='y', alpha=.25, linewidth=.6)
    axes[1].set_axisbelow(True)
    axes[0].set_title('(a) PSL–time comparison', y=-.39)
    axes[1].set_title('(b) Solve time', y=-.39)
    if quick:
        fig.suptitle('Reduced-budget smoke check', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(directory/f'psl_time.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    chosen = next((r for r in rows if r['method']=='ours'), rows[0])
    with np.load(directory/f"{chosen['method']}_{chosen['seed']}_r1.npz") as data:
        waves = [data['initial_waveform'], data['waveform']]
    delays, dopplers = np.arange(-20,21), np.arange(-8,9)
    maps = [10*np.log10(np.maximum(np.abs(ambiguity_numpy(w, dopplers, delays)/w.size)**2,1e-8)) for w in waves]
    fig, axes = plt.subplots(1, 2, figsize=(7.2,3), layout='constrained')
    for axis, values, title in zip(axes, maps, ('(a) Initial waveform', f"(b) {chosen['label']}")):
        mesh = axis.pcolormesh(delays, dopplers, values, cmap='jet', vmin=-80, vmax=0, shading='nearest', rasterized=True)
        axis.add_patch(Rectangle((-10.5,-4.5),21,9,fill=False,edgecolor='black',linewidth=1,linestyle='--'))
        axis.set(xlabel='Delay bin',ylabel='Doppler bin',title=title)
    fig.colorbar(mesh,ax=axes,label='Normalized AF power (dB)',pad=.02)
    if quick:
        fig.suptitle('Reduced-budget smoke check', fontsize=10)
    for ext in ('png','pdf'):
        fig.savefig(directory/f'af_before_after.{ext}',dpi=300,bbox_inches='tight')
    np.savez_compressed(directory/'af_maps.npz',delays=delays,dopplers=dopplers,before_db=maps[0],after_db=maps[1])
    plt.close(fig)
