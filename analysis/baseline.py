import numpy as np
from common import *
T=load_tasks_shared(); W=weight_map(T); TR=truth_map(T); N=len(T)
print("tasks",N,"outputs",len(TR))
runs=load_runs_shared()
pooled={}
for r in runs:
    for k,v in r.items(): pooled.setdefault(k,[]).extend(v)
def ev(res,label):
    sc=orc=0;lost=[]
    for bk,c in res.items():
        items=sorted(group(c),key=lambda x:-kgmon(x))
        hit=[np.array_equal(i[0],TR[bk]) for i in items]
        if any(hit): orc+=W[bk]
        if any(hit[:2]): sc+=W[bk]
        elif any(hit): lost.append((bk,hit.index(True)+1,len(items)))
    print(f"{label}: score {100*sc/N:.2f}%  oracle {100*orc/N:.2f}%  lost-in-selection {len(lost)}")
    return lost
for j,r in enumerate(runs): ev(r,f"seed{j}")
lost=ev(pooled,"pooled")
print(sorted(lost))
