import json,numpy as np,pandas as pd
import pl
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False})
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D=os.path.join(ROOT,'results')+os.sep
os.makedirs('res',exist_ok=True)   # outputs go to ./res
df=pd.read_csv(D+'results_all.csv'); main=df[df.stage=='main']
b0={j['grid']:j for j in map(json.loads,open(D+'boot_snb0.jsonl'))}; b1={j['grid']:j for j in map(json.loads,open(D+'boot_snb1.jsonl'))}
LBL={'eia_ERCO':'ERCOT','eia_CISO':'CAISO','eia_SWPP':'SWPP','eia_MISO':'MISO','opsd_DE':'Germany','opsd_ES':'Spain','opsd_FR':'France','opsd_DK':'Denmark','opsd_PT':'Portugal','opsd_AT':'Austria','opsd_GR':'Greece','grid_india':'India'}
MAIN=[g for g in LBL if g!='grid_india']
COL={'gx':'#D55E00','timexer':'#0072B2','itransformer':'#009E73','patchtst':'#E69F00','dlinear':'#56B4E9','snaive':'#888888','ci':'#CC79A7'}
NAME={'gx':'GateXer','timexer':'TimeXer','itransformer':'iTransformer','patchtst':'PatchTST','dlinear':'DLinear','snaive':'Seasonal naive','ci':'No sharing'}
SHORT={'itransformer':'iT','timexer':'TX','patchtst':'PT'}
def cen(j,ref):
    g=j[f'gain_vs_{ref}']; return g[0]*100,g[1]*100,g[2]*100
# ---------- Fig A: gain forest (single column)
def figA():
    rows=[]
    for g in LBL:
        j=b0[g]; p,lo,hi=cen(j,j['base']); rows.append((g,p,lo,hi,j['base']))
    main_rows=sorted([r for r in rows if r[0]!='grid_india'],key=lambda r:-r[1]); ind=[r for r in rows if r[0]=='grid_india'][0]
    fig,ax=plt.subplots(figsize=(3.5,3.05)); order=main_rows+[ind]
    for i,(g,p,lo,hi,base) in enumerate(order):
        sig=lo>0; c=COL['gx'] if sig else '#888888'
        ax.plot([lo,hi],[i,i],c=c,lw=1.2,solid_capstyle='butt'); ax.plot([p],[i],'o' if g!='grid_india' else 's',ms=4,c=c,mfc=c if g!='grid_india' else 'white',mew=1)
        ax.text(11.6,i,SHORT[base],va='center',ha='left',fontsize=6.5,color='#555')
    ax.axvline(0,c='#333',lw=0.6); ax.set_yticks(range(len(order))); ax.set_yticklabels([LBL[r[0]] for r in order]); ax.set_xlim(-6,11.5); ax.set_ylim(-0.7,len(order)-0.3)
    ax.axhline(len(order)-1.5,c='#bbb',lw=0.5,ls=(0,(3,2))); ax.set_xlabel('nMAE gain of GateXer over the strongest official baseline (%)'); ax.text(11.6,-0.95,'best\nbase',fontsize=6,ha='left',va='bottom',color='#555')
    ax.invert_yaxis(); fig.tight_layout(pad=0.3); fig.savefig('res/FigA_gain_per_grid.pdf'); fig.savefig('res/FigA_gain_per_grid.png',dpi=600); plt.close(fig)
# ---------- Fig B: per-target gain heatmap
def figB():
    G=MAIN+['grid_india']; M=np.array([[b0[g][f"pt_gain_vs_{b0[g]['base']}"][k]*100 for k in range(3)] for g in G])
    fig,ax=plt.subplots(figsize=(3.5,3.1)); v=12
    im=ax.imshow(M,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
    for i in range(len(G)):
        for k in range(3): ax.text(k,i,f'{M[i,k]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(M[i,k])>8 else '#111')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Load','Solar','Wind']); ax.xaxis.tick_top(); ax.set_yticks(range(len(G))); ax.set_yticklabels([LBL[g] for g in G])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.axhline(len(G)-1.5,c='white',lw=2)
    cb=fig.colorbar(im,ax=ax,fraction=0.06,pad=0.03); cb.set_label('gain (%)',fontsize=7); cb.ax.tick_params(labelsize=6.5,length=2)
    fig.tight_layout(pad=0.3); fig.savefig('res/FigB_gain_per_target.pdf'); fig.savefig('res/FigB_gain_per_target.png',dpi=600); plt.close(fig)
# ---------- Fig C: error by horizon
def figC():
    mods=['dlinear','patchtst','itransformer','timexer','gx']; hs=[1,6,24]
    fig,ax=plt.subplots(figsize=(3.5,2.35))
    m=main[(main.snb==0)&(main.grid!='grid_india')]
    for md in mods:
        v=[np.mean([m[m.model==md].groupby('grid')[f'h{h}_{t}'].mean().mean() for t in ('load','solar','wind')]) for h in hs]
        ax.plot(hs,v,'-o',c=COL[md],lw=1.6 if md=='gx' else 1.0,ms=4 if md=='gx' else 3,label=NAME[md],zorder=5 if md=='gx' else 3)
    ax.set_xticks(hs); ax.set_xlabel('forecast horizon (h)'); ax.set_ylabel('nMAE (mean of 11 grids)'); ax.legend(frameon=False,loc='lower right',ncol=1)
    fig.tight_layout(pad=0.3); fig.savefig('res/FigC_horizon.pdf'); fig.savefig('res/FigC_horizon.png',dpi=600); plt.close(fig)
# ---------- Fig D: forecast week
def figD():
    z=np.load(D+'week_ercot.npz'); t=np.arange(168)
    fig,axs=plt.subplots(3,1,figsize=(7.16,3.4),sharex=True)
    for k,(ax,nm) in enumerate(zip(axs,['Load','Solar','Wind'])):
        ax.plot(t,z['actual'][:,k],c='#222',lw=1.3,label='Actual')
        for m,ls in [('itransformer',(0,(3,1.5))),('timexer',(0,(1,1)))]: ax.plot(t,z[m][:,k],c=COL[m],lw=0.9,ls=ls,label=NAME[m])
        ax.plot(t,z['gx'][:,k],c=COL['gx'],lw=1.2,label=NAME['gx'])
        ax.set_ylabel(f'{nm} (MW)'); [ax.axvline(d*24,c='#ddd',lw=0.5,zorder=0) for d in range(1,7)]
    axs[0].legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.28)); axs[2].set_xlabel('hours into the test week (ERCOT, 24-h forecasts joined day by day)'); axs[2].set_xlim(0,167)
    fig.tight_layout(pad=0.3,h_pad=0.4); fig.savefig('res/FigD_week.pdf'); fig.savefig('res/FigD_week.png',dpi=600); plt.close(fig)
# ---------- Fig E: corrected protocol
def figE():
    G5=['eia_ERCO','eia_CISO','opsd_DE','opsd_ES','grid_india']; mods=['patchtst','itransformer','timexer','ci','gx']
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.16,2.45),gridspec_kw={'width_ratios':[1.25,1]})
    pl_=main[(main.snb==0)&main.grid.isin(G5)].groupby(['grid','model']).nmae_mean.mean().unstack(); co=main[(main.snb==1)&main.grid.isin(G5)].groupby(['grid','model']).nmae_mean.mean().unstack()
    # normalise each grid by its plain TimeXer so the five grids average sensibly
    ref=pl_['timexer']; pn=pl_.div(ref,axis=0).mean(); cn=co.div(ref,axis=0).mean()
    x=np.arange(len(mods)); w=0.36
    a1.bar(x-w/2,[pn[m] for m in mods],w,color='#BDBDBD',label='plain'); a1.bar(x+w/2,[cn[m] for m in mods],w,color=[COL[m] for m in mods],label='with yesterday-correction')
    for i,m in enumerate(mods): a1.text(i-w/2,pn[m]+0.003,f'{pn[m]:.3f}',ha='center',fontsize=6,rotation=90,va='bottom'); a1.text(i+w/2,cn[m]+0.003,f'{cn[m]:.3f}',ha='center',fontsize=6,rotation=90,va='bottom')
    a1.set_xticks(x); a1.set_xticklabels([NAME[m] if m!='ci' else 'No sharing' for m in mods],fontsize=6.5); a1.set_ylim(0.78,1.12); a1.set_ylabel('nMAE relative to plain TimeXer'); a1.text(0.0,1.075,'(a)  mean over 5 grids',transform=a1.transAxes,fontsize=7.5,va='bottom',ha='left')
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(fc='#BDBDBD',label='plain'),Patch(fc='#777',label='corrected')],frameon=False,loc='upper right',ncol=2,fontsize=6.5)
    for i,g in enumerate(G5):
        j=b1[g]; p,lo,hi=cen(j,'timexer'); a2.plot([lo,hi],[i,i],c='#444',lw=1.2); a2.plot([p],[i],'o',c=COL['gx'],ms=4)
    a2.axvline(0,c='#333',lw=0.6); a2.set_yticks(range(5)); a2.set_yticklabels([LBL[g] for g in G5]); a2.invert_yaxis(); a2.set_xlabel('GateXer gain over TimeXer, both corrected (%)'); a2.text(0.0,1.03,'(b)  paired 95% intervals',transform=a2.transAxes,fontsize=7.5,va='bottom',ha='left')
    fig.tight_layout(pad=0.3,w_pad=1.2); fig.savefig('res/FigE_corrected.pdf'); fig.savefig('res/FigE_corrected.png',dpi=600); plt.close(fig)
# ---------- Fig F: gate strengths
def figF():
    g=main[(main.snb==0)&(main.model=='gx')].dropna(subset=['gates']); G=MAIN
    M=np.zeros((len(G),6))
    for i,gr in enumerate(G):
        vals=[]
        for s in g[g.grid==gr].gates:
            js=json.loads(s); a=np.abs(np.array([[b['alpha_src'],b['alpha_ex']] for b in js]))  # [blocks,2,S]
            vals.append(a.mean(0).reshape(-1))
        M[i]=np.mean(vals,0)
    fig,ax=plt.subplots(figsize=(3.5,3.0)); im=ax.imshow(M,cmap='Oranges',vmin=0,vmax=0.3,aspect='auto')
    for i in range(len(G)):
        for k in range(6): ax.text(k,i,f'{M[i,k]:.2f}',ha='center',va='center',fontsize=6.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(['L','S','W']*2); ax.set_yticks(range(len(G))); ax.set_yticklabels([LBL[x] for x in G]); ax.xaxis.tick_top()
    ax.axvline(2.5,c='white',lw=2); ax.text(1,-1.15,'other sources',ha='center',fontsize=7,fontweight='bold'); ax.text(4,-1.15,'weather',ha='center',fontsize=7,fontweight='bold')
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); cb=fig.colorbar(im,ax=ax,fraction=0.06,pad=0.03); cb.set_label(r'mean $|\tanh\,\alpha|$',fontsize=7); cb.ax.tick_params(labelsize=6.5,length=2)
    fig.tight_layout(pad=0.3); fig.savefig('res/FigF_gates.pdf'); fig.savefig('res/FigF_gates.png',dpi=600); plt.close(fig)
for f in (figA,figB,figC,figD,figE,figF): f()
print('ok')
