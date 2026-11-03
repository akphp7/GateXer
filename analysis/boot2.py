import csv,os,numpy as np,json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=os.environ.get('GATEXER_RUNS',os.path.join(ROOT,'runs'))+os.sep; A=os.path.join(ROOT,'results')+os.sep
rows=list(csv.DictReader(open(A+'results_all.csv')))
def sel(grid,model,batches,stage=None):
    return [r for r in rows if r['grid']==grid and r['model']==model and r['batch'] in batches and int(r['snb'])==0 and (stage is None or r['stage']==stage)]
def load(rs):
    Ps=[]
    for r in rs:
        z=np.load(R+r['path'].replace('.json','.npz')); Ps.append(z['test_P']); Y=z['test_Y']; idx=z['te_idx']
    return np.mean(Ps,0),Y,idx
rng=np.random.default_rng(1)
GR=['eia_ERCO','eia_CISO','eia_SWPP','eia_MISO','opsd_DE','opsd_PT']
out=open(A+'boot_ablate4.jsonl','w')
for g in GR:
    sets={'gx':sel(g,'gx',['main1','main2','batch1'],'main'),'srconly':sel(g,'gx_srconly',['ablate4']),'exonly':sel(g,'gx_exonly',['ablate4']),
          'nosh':sel(g,'gx_nosh',['batch1'],'ablate'),'ci':sel(g,'ci',['main1','main2','batch1'],'main')}
    sets={k:v for k,v in sets.items() if v}
    P={};
    for k,v in sets.items(): P[k],Y,idx=load(v)
    den=np.abs(Y).mean((0,1))
    def stat(a,b):  # gain of a over b
        gains=[];lo=[];hi=[]
        pt=None
        eb_all=np.abs(P[b]-Y).mean(1)/den; ea_all=np.abs(P[a]-Y).mean(1)/den; pt=(eb_all.mean()-ea_all.mean())/eb_all.mean()
        for ph in range(24):
            s=np.where((idx-idx[0])%24==ph)[0]; B=rng.integers(0,len(s),(1000,len(s))); eb=eb_all[s]; ea=ea_all[s]
            g0=(eb.mean()-ea.mean())/eb.mean(); bg=(eb[B].mean(1).mean(1)-ea[B].mean(1).mean(1))/eb[B].mean(1).mean(1)
            lo.append(np.percentile(bg,2.5)-g0); hi.append(np.percentile(bg,97.5)-g0)
        return [float(pt),float(pt+np.mean(lo)),float(pt+np.mean(hi))]
    res=dict(grid=g,ens={k:float((np.abs(P[k]-Y).mean(1)/den).mean()) for k in P},single={k:float(np.mean([float(r['nmae_mean']) for r in v])) for k,v in sets.items()},have=list(sets))
    ctrl='nosh' if 'nosh' in P else 'ci'; res['ctrl']=ctrl
    for a,b in [('gx','srconly'),('gx','exonly'),('srconly',ctrl),('exonly',ctrl),('gx',ctrl),('srconly','exonly')]:
        res[f'{a}_over_{b}']=stat(a,b)
    out.write(json.dumps(res)+'\n'); out.flush(); print(g,ctrl,{k:round(v,4) for k,v in res['single'].items()},{k:[round(x*100,1) for x in v] for k,v in res.items() if '_over_' in k})
