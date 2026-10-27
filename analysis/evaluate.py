"""Step B: re-rank NVARC candidates on the eval set with properties fixed in Step A."""
import json, sys, numpy as np
from common import *
from props import PROPS, active_props, violations
rel=json.load(open(__import__('pathlib').Path(__file__).parent/'reliability.json'))
T=load_tasks_shared(); W=weight_map(T); TRU=truth_map(T); N=len(T)
runs=load_runs_shared()
pooled={}
for r in runs:
    for k,v in r.items(): pooled.setdefault(k,[]).extend(v)
CTX={}
def ctx_for(task, thr):
    key=(task,thr)
    if key not in CTX:
        props=[p for p in PROPS if rel[p.__name__]<=thr]
        CTX[key]=active_props(T[task]['train'],props)
    return CTX[key]

def rank(bk, cands, mode, thr):
    items=group(cands)
    if mode=='base': return sorted(items,key=lambda x:-kgmon(x))
    task,i=bk.rsplit('_',1); act,ctx=ctx_for(task,thr)
    tin=np.array(T[task]['test'][int(i)]['input'])
    nv=[len(violations(tin,it[0],act,ctx)) for it in items]
    if mode=='veto':   # lexicographic: fewest violations first, then kgmon
        order=sorted(range(len(items)),key=lambda j:(nv[j],-kgmon(items[j])))
    elif mode=='veto0':  # only demote violators below all clean candidates (same as veto with 0/1)
        order=sorted(range(len(items)),key=lambda j:(nv[j]>0,-kgmon(items[j])))
    return [items[j] for j in order]

def ev(res, mode, thr=0.01):
    sc=0; lost=set(); hits=set()
    for bk,c in res.items():
        r=rank(bk,c,mode,thr)
        h=[np.array_equal(x[0],TRU[bk]) for x in r]
        if any(h[:2]): sc+=W[bk]; hits.add(bk)
        elif any(h): lost.add(bk)
    return 100*sc/N, hits, lost

for name,res in [('seed0',runs[0]),('seed1',runs[1]),('pooled',pooled)]:
    b,bh,bl=ev(res,'base')
    print(f"\n== {name}: baseline {b:.2f}%  lost-in-selection {len(bl)}")
    for mode in ['veto0','veto']:
        for thr in [0.005,0.01,0.02,0.05]:
            s,h,l=ev(res,mode,thr)
            rec=sorted(h-bh); broke=sorted(bh-h)
            print(f"  {mode:5s} thr<={thr:<5}  {s:.2f}% ({s-b:+.2f})  recovered {len(rec)} {rec}  broke {len(broke)} {broke}")
