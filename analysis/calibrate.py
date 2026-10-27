from pathlib import Path
"""Step A: measure each property's reliability on the 1000 public TRAINING tasks (no eval data used).
For each task with >=2 train pairs: induce active props from train pairs, check true test output."""
import json, numpy as np

from props import PROPS, active_props, violations
from common import TRAIN_DIR as TR
stats={p.__name__:[0,0] for p in PROPS}
n=0
for f in sorted(TR.glob('*.json')):
    t=json.load(open(f))
    if len(t['train'])<2: continue
    act,ctx=active_props(t['train'])
    for te in t['test']:
        n+=1
        bad=set(violations(np.array(te['input']),np.array(te['output']),act,ctx))
        for p in act:
            stats[p.__name__][0]+=1
            stats[p.__name__][1]+= p.__name__ in bad
print('test outputs',n)
print(f"{'property':26s} active  truth-violations  rate")
rel={}
for k,(a,v) in sorted(stats.items(),key=lambda x:-x[1][0]):
    r=v/a if a else float('nan'); rel[k]=r
    print(f"{k:26s} {a:5d}  {v:5d}  {100*r:6.2f}%")
json.dump(rel,open(Path(__file__).parent/'reliability.json','w'),indent=1)
