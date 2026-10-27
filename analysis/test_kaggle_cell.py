"""Apply the exact Kaggle cell (kaggle/verifier_cell.py) to NVARC's real harness submissions and score before/after."""
import copy, json, sys
from pathlib import Path
from common import load_tasks_shared, ROOT, CAND_DIRS
sys.path.insert(0, str(ROOT / "kaggle"))
import verifier_cell as V

T = load_tasks_shared()
CH = {k: {"train": v["train"], "test": [{"input": t["input"]} for t in v["test"]]} for k, v in T.items()}
SUBS = [ROOT / "data/harness/runs/s0/eval_submission_seed0.json", ROOT / "data/harness/runs/s1/eval_submission_seed1.json"]


def solved(sub):
    return {(k, i) for k, v in T.items() for i, t in enumerate(v["test"])
            if i < len(sub.get(k, [])) and t["output"] in (sub[k][i].get("attempt_1"), sub[k][i].get("attempt_2"))}


def pct(outs):
    return 100 * sum(1 / len(T[k]["test"]) for k, _ in outs) / len(T)


for seed, (sp, cd) in enumerate(zip(SUBS, CAND_DIRS)):
    sub = json.load(open(sp)); new = copy.deepcopy(sub)
    n = V.apply_verifier(CH, new, V.load_pool(str(cd)), log=lambda m: None)
    a, b = solved(sub), solved(new)
    print(f"seed{seed}: NVARC {pct(a):.2f}% -> verifier {pct(b):.2f}%  changed {n}  gained {sorted(b - a)}  lost {sorted(a - b)}")
