"""Render only the scatter and before/after AF figures used by the final paper."""
from pathlib import Path
import argparse
import csv
import os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.tri as mtri
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator,NullFormatter

def setup():
    available={x.name for x in font_manager.fontManager.ttflist}
    font='Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':font,'font.size':9.1,'font.weight':'normal',
        'mathtext.fontset':'custom','mathtext.rm':font,'mathtext.it':font+':italic',
        'mathtext.bf':font+':bold','axes.labelsize':9.1,'axes.labelweight':'normal',
        'xtick.labelsize':9.1,'ytick.labelsize':9.1,'text.color':'black','axes.labelcolor':'black',
        'axes.edgecolor':'black','axes.linewidth':.6,'xtick.color':'black','ytick.color':'black',
        'pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
    return font

def save(fig,out,name):
    fig.canvas.draw()
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text().strip():
            box=label.get_window_extent(fig.canvas.get_renderer())
            if box.width and box.height:
                assert box.x0>=-.5 and box.y0>=-.5 and box.x1<=fig.bbox.width+.5 and box.y1<=fig.bbox.height+.5,(name,label.get_text())
    fig.savefig(out/f'{name}.pdf',dpi=600)
    fig.savefig(out/f'{name}.png',dpi=300)
    plt.close(fig)

def scatter(out):
    with (ROOT/'data/scatter_points.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    fig=plt.figure(figsize=(86/25.4,2.0));ax=fig.add_axes([.165,.21,.765,.74])
    ax.set_xscale('log');ax.set_xlim(10,100000);ax.set_ylim(-52,-40)
    ax.xaxis.set_major_locator(FixedLocator([10,100,1000,10000,100000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.0f}'))
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10)));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([-52,-50,-48,-46,-44,-42,-40]))
    ax.set_xlabel('Solve time (ms)',labelpad=3);ax.set_ylabel('Local PSL (dB)',labelpad=3)
    ax.tick_params(which='both',length=0,pad=2);ax.set_axisbelow(True)
    ax.grid(which='major',color='#DEDEDE',linewidth=.5);ax.grid(which='minor',color='#EFEFEF',linewidth=.3)
    colors={'proposed':'#C70000','neural network':'#69AC43','optimization':'#4271C6'}
    offsets={'Ours':(6,6,'left'),'WaveNet':(5,0,'left'),'MAL-Net':(-5,6,'right'),
             'Consensus-ADMM':(-5,-1,'right'),'ADMM':(-6,13,'right'),'QGD':(7,1,'left'),'AISO':(7,-9,'left')}
    for row in rows:
        x,y=float(row['time_ms']),float(row['local_psl_db']);method=row['method']
        ax.scatter(x,y,marker={'proposed':'D','neural network':'o','optimization':'^'}[row['family']],s=27,color=colors[row['family']],edgecolors='white',linewidths=.3,zorder=4)
        dx,dy,ha=offsets[method]
        ax.annotate(method,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=9.1,weight='bold' if method=='Ours' else 'normal')
    save(fig,out,'scatter')

def display_interpolation(x,y,db,factor=32):
    xx,yy=np.meshgrid(x,y);nx,ny=len(x),len(y);triangles=[]
    for i in range(ny-1):
        for j in range(nx-1):
            a=i*nx+j;triangles.extend(((a,a+1,a+nx+1),(a,a+nx+1,a+nx)))
    tri=mtri.Triangulation(xx.ravel(),yy.ravel(),np.asarray(triangles))
    shown=np.clip(db,-80,0)
    interp=mtri.LinearTriInterpolator(tri,shown.ravel())
    gx,gy=np.meshgrid(np.linspace(x[0],x[-1],(nx-1)*factor+1),np.linspace(y[0],y[-1],(ny-1)*factor+1))
    values=np.asarray(interp(gx,gy))
    np.testing.assert_allclose(values[::factor,::factor],shown,atol=1e-10,rtol=0)
    return values

def af_pair(out):
    with np.load(ROOT/'data/af_example.npz',allow_pickle=False) as f:
        x,y=f['delays'],f['dopplers'];ix,iy=abs(x)<=20,abs(y)<=8
        grids=[display_interpolation(x[ix],y[iy],f[s+'_db'][np.ix_(iy,ix)]) for s in ['before','after']]
    fig=plt.figure(figsize=(86/25.4,50/25.4))
    norm=Normalize(-80,0,clip=True);cmap=matplotlib.colormaps['jet'].resampled(256)
    def axes(rect):
        a,b,w,h=rect;return fig.add_axes([a/86,b/50,w/86,h/50])
    for i,rect in enumerate([(11,18,27,27),(44.5,18,27,27)]):
        ax=axes(rect);ax.imshow(grids[i],origin='lower',extent=(-20,20,-8,8),aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-8,-4,0,4,8])
        ax.tick_params(length=2,width=.6,pad=1.4,labelleft=i==0);ax.tick_params(axis='x',pad=3.5)
        if i:ax.tick_params(axis='y',length=0)
        roi=Rectangle((-10,-4),20,8,fill=False,edgecolor='white',linewidth=.8,linestyle=(0,(3,1.8)),zorder=5)
        roi.set_path_effects([pe.Stroke(linewidth=1.3,foreground='#171717'),pe.Normal()]);ax.add_patch(roi)
        fig.text((rect[0]+rect[2]/2)/86,7.4/50,r'Delay index $k$',ha='center',va='bottom')
        fig.text((rect[0]+rect[2]/2)/86,1.4/50,['(a) Before','(b) After'][i],ha='center',va='bottom')
    fig.text(2.5/86,31.5/50,r'Doppler index $\ell$',ha='center',va='center',rotation=90)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=axes((75,18,1.7,27)),ticks=[-80,-60,-40,-20,0])
    cb.outline.set_linewidth(.6);cb.ax.tick_params(labelsize=9.1,length=2,pad=1.3);cb.ax.set_title('dB',fontsize=9.1,pad=3)
    save(fig,out,'af_before_after_single_column')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/figures');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);font=setup();scatter(a.output);af_pair(a.output)
    print(f'Final-paper scatter and AF pair generated; font={font}. AF interpolation is display-only.')
