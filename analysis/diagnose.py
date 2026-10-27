import json, numpy as np
from common import *
from props import PROPS, active_props, violations
rel=json.load(open(__import__('pathlib').Path(__file__).parent/'reliability.json'))
props=[p for p in PROPS if rel[p.__name__]<=0.01]
T=load_tasks_shared(); TRU=truth_map(T)
runs=load_runs_shared(); pooled={}
for r in runs:
    for k,v in r.items(): pooled.setdefault(k,[]).extend(v)
tot_wrong=flag_wrong=tot_right=flag_right=0
rows=[]
for bk,c in pooled.items():
    t,i=bk.rsplit('_',1); act,ctx=active_props(T[t]['train'],props)
    tin=np.array(T[t]['test'][int(i)]['input'])
    items=sorted(group(c),key=lambda x:-kgmon(x))
    h=[np.array_equal(x[0],TRU[bk]) for x in items]
    v=[violations(tin,x[0],act,ctx) for x in items]
    for hh,vv in zip(h,v):
        if hh: tot_right+=1; flag_right+=bool(vv)
        else: tot_wrong+=1; flag_wrong+=bool(vv)
    if any(h) and not any(h[:2]):
        r=h.index(True)
        above=v[:r]
        # near-miss: cell agreement of top-1 with truth
        tr=TRU[bk]; top=items[0][0]
        nm = f"{100*(top==tr).mean():.0f}%" if top.shape==tr.shape else "shape≠"
        rows.append((bk,len(act),r+1,len(items),sum(bool(a) for a in above),r, items[0][1], items[r][1], nm))
print(f"all candidates: wrong {tot_wrong}, flagged {flag_wrong} ({100*flag_wrong/tot_wrong:.1f}%); right {tot_right}, flagged {flag_right}")
print("\nlost output   active  truth-rank/of  above-truth flagged/total  top1-views truth-views  top1-cell-agree")
for r in rows: print(f"{r[0]:12s} {r[1]:4d}   {r[2]:3d}/{r[3]:<3d}    {r[4]:3d}/{r[5]:<3d}    {r[6]:3d}  {r[7]:3d}   {r[8]}")
