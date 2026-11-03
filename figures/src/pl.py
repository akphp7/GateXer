import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import glob
import os
FONTS=os.environ.get('GATEXER_FONTS','fonts')   # optional folder with Tinos .ttf files; falls back to DejaVu Serif
for p in glob.glob(os.path.join(FONTS,'*.ttf')): fm.fontManager.addfont(p)
mpl.rcParams.update({'font.family':['Tinos','DejaVu Serif'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':0.5})
# Okabe-Ito colour-blind-safe palette (fill, edge)
C=dict(load=('#D6E8F5','#0072B2'),solar=('#FCEFC4','#C98A00'),wind=('#CDEBE0','#009E73'),ex=('#F4DCE9','#B0508A'),
       gate=('#F9D9C6','#D55E00'),glob=('#D9D9D9','#555555'),grey=('#F2F2F2','#555555'),white=('#FFFFFF','#333333'),key=('#FFFFFF','#333333'))
class Fig:
    def __init__(s,W,H):
        s.W,s.H=W,H; s.f=plt.figure(figsize=(W,H)); s.a=s.f.add_axes([0,0,1,1]); s.a.set_xlim(0,W); s.a.set_ylim(0,H); s.a.axis('off')
    def Y(s,y): return s.H-y
    def box(s,x,y,w,h,txt=None,c='white',fs=6.5,lw=0.7,dash=False,r=0.04,bold=False,color='#111',z=2,fill=True,ha='center',va='center',pad=0.05,ec=None,ls=1.15):
        fc,e=C[c]; e=ec or e
        p=FancyBboxPatch((x+r,s.Y(y+h)+r),w-2*r,h-2*r,boxstyle=f'round,pad={r},rounding_size={r}',fc=fc if fill else 'none',ec=e,lw=lw,ls=(0,(3,2)) if dash else '-',zorder=z); s.a.add_patch(p)
        if txt: s.text(x+(w/2 if ha=='center' else pad),y+h/2 if va=='center' else y+pad,txt,fs=fs,bold=bold,color=color,ha=ha,va=va,z=z+1,ls=ls)
        return p
    def sq(s,x,y,w,c,lw=0.5,fill=None,ec=None):
        fc,e=C[c]; s.a.add_patch(Rectangle((x,s.Y(y+w)),w,w,fc=fill or fc,ec=ec or e,lw=lw,zorder=3))
    def text(s,x,y,t,fs=6.5,bold=False,color='#111',ha='center',va='center',z=5,rot=0,italic=False,ls=1.15):
        s.a.text(x,s.Y(y),t,fontsize=fs,fontweight='bold' if bold else 'normal',fontstyle='italic' if italic else 'normal',color=color,ha=ha,va=va,zorder=z,rotation=rot,linespacing=ls)
    def arrow(s,pts,c='#222',lw=0.8,dash=False,head=True,z=4,ms=5.5):
        P=[(x,s.Y(y)) for x,y in pts]
        xs,ys=zip(*P)
        if head:
            if len(P)>2: s.a.plot(xs[:-1],ys[:-1],c=c,lw=lw,ls=(0,(3,2)) if dash else '-',zorder=z,solid_capstyle='butt')
            s.a.add_patch(FancyArrowPatch(P[-2],P[-1],arrowstyle='-|>',mutation_scale=ms,lw=lw,color=c,zorder=z,shrinkA=0,shrinkB=0,ls=(0,(3,2)) if dash and len(P)==2 else '-'))
        else: s.a.plot(xs,ys,c=c,lw=lw,ls=(0,(3,2)) if dash else '-',zorder=z)
    def series(s,x,y,w,h,fn,c,lw=0.8,n=120,z=4):
        t=np.linspace(0,1,n); s.a.plot(x+w*t,s.Y(y+h/2-np.array([fn(v) for v in t])*h/2),c=c,lw=lw,zorder=z)
    def badge(s,x,y,n,r=0.075):
        s.a.add_patch(Circle((x,s.Y(y)),r,fc='#222',ec='none',zorder=6)); s.text(x,y+0.003,str(n),fs=6.5,bold=True,color='white',z=7)
    def save(s,name):
        for ext in ('pdf','svg'): s.f.savefig(f'res/{name}.{ext}')
        s.f.savefig(f'res/{name}.png',dpi=600)
