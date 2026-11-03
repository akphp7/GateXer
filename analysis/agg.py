import json,glob,os,re,csv,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS=sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,'runs')
os.chdir(RUNS)   # run folders: <batch>/.../<grid>/<model>_s<seed>.json
rows=[]
for p in glob.glob('**/*.json',recursive=True):
    if os.path.basename(p) in ('best.json',) or 'tune' in p.split(os.sep) or 'Improve' in p or p.startswith('results_smoke'): continue
    try: d=json.load(open(p))
    except: continue
    if 'metrics' not in d or 'args' not in d: continue
    m=re.match(r'(.+)_s(\d+)\.json',os.path.basename(p)); 
    if not m: continue
    a=d['args']; me=d['metrics']; ex=d.get('extra',{}) or {}
    grid=os.path.basename(os.path.dirname(p)); batch=p.split(os.sep)[0]
    stage='ablate' if '/ablate' in p.replace(os.sep,'/') else 'main'
    nm=me['nmae']; nh=me.get('nmae_by_h',{})
    rows.append(dict(batch=batch,stage=stage,grid=grid,model=m.group(1),seed=int(m.group(2)),snb=int(a.get('snb',0) or 0),
      nmae_load=nm[0],nmae_solar=nm[1],nmae_wind=nm[2],nmae_mean=sum(nm)/3,val=me.get('val_score'),
      **{f'h{h}_{t}':nh[h][i] for h in ('1','6','24') if h in nh for i,t in enumerate(('load','solar','wind'))},
      ramp_mean=sum(me['ramp_mae'])/3 if 'ramp_mae' in me else None,params=ex.get('params'),seconds=d.get('seconds'),epochs=ex.get('epochs_run'),
      gates=json.dumps(d.get('gates')) if d.get('gates') else '',path=p))
keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(os.path.join(ROOT,'results','results_all.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows))
import collections; print(collections.Counter((r['batch'],r['stage'],r['snb']) for r in rows))
