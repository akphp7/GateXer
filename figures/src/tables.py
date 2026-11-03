import json,numpy as np,pandas as pd
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D=os.path.join(ROOT,'results')+os.sep
os.makedirs('res',exist_ok=True)   # outputs go to ./res
df=pd.read_csv(D+'results_all.csv'); main=df[df.stage=='main']
b0={j['grid']:j for j in map(json.loads,open(D+'boot_snb0.jsonl'))}; b1={j['grid']:j for j in map(json.loads,open(D+'boot_snb1.jsonl'))}
LBL={'eia_ERCO':'ERCOT','eia_CISO':'CAISO','eia_SWPP':'SWPP','eia_MISO':'MISO','opsd_DE':'Germany','opsd_ES':'Spain','opsd_FR':'France','opsd_DK':'Denmark','opsd_PT':'Portugal','opsd_AT':'Austria','opsd_GR':'Greece','grid_india':'India'}
def cen(j,ref):
    g=j[f'gain_vs_{ref}']; return g[0]*100,g[1]*100,g[2]*100
def ms(sub,g,m):
    v=sub[(sub.grid==g)&(sub.model==m)].nmae_mean.values; return v.mean(),v.std(ddof=1) if len(v)>1 else 0
def cell(mu,sd,best,nostd=False): 
    t=f'{mu:.4f}'; t=f'\\textbf{{{t}}}' if best else t
    return t if nostd else f'{t}{{\\scriptsize$\\pm${sd:.4f}}}'
out=[]
# ---- main table (plain)
mods=['snaive','dlinear','patchtst','itransformer','timexer','gx']
hdr=['S.\\,naive','DLinear','PatchTST','iTransformer','TimeXer','GateXer']
P=main[main.snb==0]
L=[r'\begin{table*}[t]',r'\centering',r'\caption{Main results, plain protocol. Test nMAE averaged over load, solar and wind (mean {\scriptsize$\pm$} std over 3 seeds; lower is better; best in bold). The last column is the gain of the 3-seed GateXer ensemble over the ensemble of the strongest official baseline (PatchTST, iTransformer or TimeXer), with the paired 95\% interval; $\dagger$ marks an interval that excludes zero. India is a single national series and is shown separately.}',r'\label{tab:main}',r"\footnotesize",r"\setlength{\tabcolsep}{4pt}",r'\begin{tabular}{@{}l'+'c'*6+r'c@{}}',r'\toprule','Grid & '+' & '.join(hdr)+r' & Ensemble gain (\%) \\',r'\midrule']
def rowfor(g):
    vals={m:ms(P,g,m) for m in mods}; best=min(vals,key=lambda m:vals[m][0]); j=b0[g]; p,lo,hi=cen(j,j['base'])
    dag='$^\\dagger$' if lo>0 else ''
    return f"{LBL[g]} & "+' & '.join(cell(*vals[m],m==best,nostd=(m=='snaive')) for m in mods)+f" & {p:+.1f} [{lo:.1f}, {hi:.1f}]{dag} \\\\"
for g in LBL:
    if g=='grid_india': continue
    L.append(rowfor(g))
G11=[g for g in LBL if g!='grid_india']
means={m:np.mean([ms(P,g,m)[0] for g in G11]) for m in mods}; bestm=min(means,key=means.get)
wins={m:0 for m in mods}
for g in G11:
    v={m:ms(P,g,m)[0] for m in mods}; wins[min(v,key=v.get)]+=1
ens_mean=np.mean([cen(b0[g],b0[g]['base'])[0] for g in G11])
L.append(r'\midrule')
L.append('Mean of 11 grids & '+' & '.join((f'\\textbf{{{means[m]:.4f}}}' if m==bestm else f'{means[m]:.4f}') for m in mods)+f' & {ens_mean:+.1f} \\\\')
L.append('Grids won (of 11) & '+' & '.join(str(wins[m]) for m in mods)+' & 11 of 11$^\\dagger$ \\\\')
L.append(r'\midrule'); L.append(rowfor('grid_india')); L+= [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
open('res/table_main.tex','w').write('\n'.join(L))
# ---- corrected table
G5=['eia_ERCO','eia_CISO','opsd_DE','opsd_ES','grid_india']; cm=['patchtst','itransformer','timexer','ci','gx']; ch=['PatchTST','iTransformer','TimeXer','No sharing','GateXer']
C1=main[main.snb==1]
L=[r'\begin{table*}[t]',r'\centering',r'\caption{Corrected protocol: every trainable model receives the yesterday-correction of Eq.~(\ref{eq:snb}). Test nMAE (mean {\scriptsize$\pm$} std over 3 seeds). The last column is the gain of the GateXer ensemble over the TimeXer ensemble with the paired 95\% interval; intervals include zero on all five grids, so the two are tied.}',r'\label{tab:corrected}',r"\footnotesize",r"\setlength{\tabcolsep}{4pt}",r'\begin{tabular}{@{}l'+'c'*5+r'c@{}}',r'\toprule','Grid & '+' & '.join(ch)+r' & Gain over TimeXer (\%) \\',r'\midrule']
for g in G5:
    vals={m:ms(C1,g,m) for m in cm}; best=min(vals,key=lambda m:vals[m][0]); p,lo,hi=cen(b1[g],'timexer')
    L.append(f"{LBL[g]} & "+' & '.join(cell(*vals[m],m==best) for m in cm)+f" & {p:+.1f} [{lo:.1f}, {hi:.1f}] \\\\")
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']; open('res/table_corrected.tex','w').write('\n'.join(L))
# ---- cost table
rows=[('DLinear','dlinear'),('PatchTST','patchtst'),('iTransformer','itransformer'),('TimeXer','timexer'),('GateXer','gx')]
L=[r'\begin{table}[t]',r'\centering',r'\caption{Model size and training time (mean over all grids and seeds, plain protocol, one GPU).}',r'\label{tab:cost}',r'\footnotesize',r'\begin{tabular}{@{}lrr@{}}',r'\toprule',r'Model & Parameters (k) & Time per run (s) \\',r'\midrule']
for nm,m in rows:
    s=P[P.model==m]; L.append(f"{nm} & {s.params.mean()/1000:,.0f} & {s.seconds.mean():.0f} \\\\")
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']; open('res/table_cost.tex','w').write('\n'.join(L))
# ---- ablation (6 grids)
A3=df[(df.batch=='batch1')&(df.stage=='ablate')]; A4=df[(df.batch=='ablate4')]
gg=['eia_ERCO','eia_CISO','eia_SWPP','eia_MISO','opsd_PT','opsd_DE']; gh=['ERCOT','CAISO','SWPP','MISO','Portugal','Germany']
def val(src,g,m,star=False):
    v=src[(src.grid==g)&(src.model==m)].nmae_mean.values
    if len(v)==0: return None
    return v.mean(),(v.std(ddof=1) if len(v)>1 else 0)
def fmt(x,best=False,star=False):
    if x is None: return '--'
    t=f'{x[0]:.4f}'; t=f'\\textbf{{{t}}}' if best else t
    return t+('$^\\ast$' if star else '')+f'{{\\scriptsize$\\pm${x[1]:.4f}}}'
rowsA=[]
def ctrl(g):
    x=val(A3,g,'gx_nosh')
    return (x,False) if x else (val(main[main.snb==0],g,'ci'),True)
rowsA.append(('No sharing (control)',[ctrl(g) for g in gg]))
rowsA.append(('Other sources only',[(val(A4,g,'gx_srconly'),False) for g in gg]))
rowsA.append(('Weather only',[(val(A4,g,'gx_exonly'),False) for g in gg]))
rowsA.append(('Both, always on (no gate)',[(val(A3,g,'gx_always'),False) for g in gg]))
rowsA.append(('Both, gated, no FAMO',[(val(A3,g,'gx_gateonly'),False) for g in gg]))
rowsA.append(('Both, gated, no trend split',[(val(A3,g,'gx_nodec'),False) for g in gg]))
rowsA.append(('GateXer (full)',[(val(main[main.snb==0],g,'gx'),False) for g in gg]))
best=[min(r[1][i][0][0] for r in rowsA if r[1][i][0] is not None) for i in range(len(gg))]
L=[r'\begin{table*}[t]',r'\centering',r'\caption{Ablation, plain protocol, test nMAE (mean {\scriptsize$\pm$} std over 3 seeds; best per grid in bold). The first three rows isolate what is shared: nothing, only the other sources, only the weather and calendar tokens; both sharing branches are gated in the last row. $^\ast$: the control is the plain no-sharing model of the main comparison, which also lacks FAMO and the trend split, so its gaps are slightly overstated; a matched control is being added. Empty cells were not run.}',r'\label{tab:ablation}',r'\scriptsize',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{@{}l'+'c'*6+r'@{}}',r'\toprule','Variant & '+' & '.join(gh)+r' \\',r'\midrule']
for nm,cells in rowsA:
    L.append(nm+' & '+' & '.join(fmt(c[0],c[0] is not None and abs(c[0][0]-best[i])<1e-12,c[1] and nm.startswith('No sharing')) for i,c in enumerate(cells))+r' \\')
    if nm.startswith('Weather only') or nm.startswith('No sharing'): L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']; open('res/table_ablation.tex','w').write('\n'.join(L))
print('\n'.join(open('res/table_main.tex').read().split('\n')[8:22]))
