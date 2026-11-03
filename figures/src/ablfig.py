import json,numpy as np
import pl
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False})
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D=os.path.join(ROOT,'results')+os.sep
os.makedirs('res',exist_ok=True)   # outputs go to ./res
b={j['grid']:j for j in map(json.loads,open(D+'boot_ablate4.jsonl'))}
LBL={'eia_ERCO':'ERCOT','eia_CISO':'CAISO','eia_SWPP':'SWPP','eia_MISO':'MISO','opsd_PT':'Portugal','opsd_DE':'Germany'}
G=list(LBL)
fig,ax=plt.subplots(figsize=(3.5,3.1))
spec=[('srconly','Other sources only','#0072B2','s',-0.24),('exonly','Weather only','#CC79A7','D',0.0),('gx','Both, gated (GateXer)','#D55E00','o',0.24)]
for key,nm,c,mk,off in spec:
    for i,g in enumerate(G):
        ctrl=b[g]['ctrl']; k=f'{key}_over_{ctrl}' if key!='gx' else f'gx_over_{ctrl}'
        p,lo,hi=[x*100 for x in b[g][k]]; y=i+off
        ax.plot([lo,hi],[y,y],c=c,lw=1.1,solid_capstyle='butt'); ax.plot([p],[y],mk,c=c,ms=3.6,label=nm if i==0 else None)
ax.axvline(0,c='#333',lw=0.6); ax.set_yticks(range(len(G))); ax.set_yticklabels([LBL[g]+('*' if b[g]['ctrl']=='ci' else '') for g in G]); ax.invert_yaxis()
ax.set_xlabel('nMAE gain over the no-sharing model (%)'); ax.legend(frameon=False,loc='lower right',fontsize=6.5)
ax.set_ylim(len(G)-0.45,-0.55)
fig.tight_layout(pad=0.3); fig.savefig('res/FigG_ablation_sharing.pdf'); fig.savefig('res/FigG_ablation_sharing.png',dpi=600)
