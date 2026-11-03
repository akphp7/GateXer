import csv,sys,os,numpy as np,json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=os.environ.get('GATEXER_RUNS',os.path.join(ROOT,'runs'))+os.sep; A=os.path.join(ROOT,'results')+os.sep
rows=[r for r in csv.DictReader(open(A+'results_all.csv')) if r['stage']=='main']
grids=sys.argv[1].split(','); snb=int(sys.argv[2]); out=A+f'boot_snb{snb}.jsonl'
OFF=['patchtst','itransformer','timexer']
rng=np.random.default_rng(0)
for g in grids:
    rs=[r for r in rows if r['grid']==g and int(r['snb'])==snb]
    if not rs: continue
    allm=sorted(set(r['model'] for r in rs)); ens={}
    for m in allm:
        Ps=[]
        for r in rs:
            if r['model']!=m: continue
            fp=R+r['path'].replace('.json','.npz')
            if not os.path.exists(fp): continue
            z=np.load(fp); Ps.append(z['test_P']); Y=z['test_Y']; idx=z['te_idx']
        if Ps: ens[m]=np.mean(Ps,0)
    models=list(ens.keys())
    day=(idx-idx[0])//24; ud,inv=np.unique(day,return_inverse=True); nd=len(ud)
    den=np.abs(Y).mean((0,1)); cnt=np.bincount(inv,minlength=nd).astype(float)
    DS={}
    for m in models:
        e=np.abs(ens[m]-Y).mean(1)/den
        DS[m]=np.stack([np.bincount(inv,weights=e[:,k],minlength=nd) for k in range(3)],1)
    E={m:DS[m].sum(0)/cnt.sum() for m in models}
    phases=range(24); Bs={}
    def phase_stats(ref):
        gains=[];los=[];his=[];dl=[];dh=[]
        for ph in phases:
            sel=np.where((idx-idx[0])%24==ph)[0]; ns=len(sel); B=rng.integers(0,ns,(1000,ns))
            er=np.abs(ens[ref][sel]-Y[sel]).mean(1)/den; eg=np.abs(ens['gx'][sel]-Y[sel]).mean(1)/den
            g0=(er.mean()-eg.mean())/er.mean(); bg=(er[B].mean(1).mean(1)-eg[B].mean(1).mean(1))/er[B].mean(1).mean(1)
            gains.append(g0); los.append(np.percentile(bg,2.5)-g0); his.append(np.percentile(bg,97.5)-g0)
        return np.mean(los),np.mean(his)
    offm=[m for m in OFF if m in models]
    base=min(offm,key=lambda m:E[m].mean())
    res=dict(grid=g,snb=snb,n_days=int(nd),base=base,single={m:float(np.mean([float(r['nmae_mean']) for r in rs if r['model']==m])) for m in allm},
             ens={m:float(E[m].mean()) for m in models},ens_pt={m:[float(v) for v in E[m]] for m in models})
    for ref in offm+(['ci'] if 'ci' in models else []):
        pt=(E[ref].mean()-E['gx'].mean())/E[ref].mean(); dlo,dhi=phase_stats(ref)
        res[f'gain_vs_{ref}']=[float(pt),float(pt+dlo),float(pt+dhi)]
        res[f'pt_gain_vs_{ref}']=[float((E[ref][k]-E['gx'][k])/E[ref][k]) for k in range(3)]
    open(out,'a').write(json.dumps(res)+'\n'); print(g,base,[round(x,4) for x in res[f'gain_vs_{base}']])
