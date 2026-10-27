"""Score the two submission files from our Kaggle commit run (NVARC adaptive pooled, eval mode)."""
import json
from common import load_tasks, EXCLUDE, ROOT

R = ROOT / "results/kaggle_run"
orig = json.load(open(R / "submission_nvarc_original.json"))
ver = json.load(open(R / "submission_with_verifier.json"))
T = load_tasks()


def solved(sub, tasks):
    outs = {f"{k}_{i}" for k, v in tasks.items() for i, t in enumerate(v["test"])
            if i < len(sub.get(k, [])) and t["output"] in (sub[k][i].get("attempt_1"), sub[k][i].get("attempt_2"))}
    return 100 * sum(1 / len(tasks[o.rsplit("_", 1)[0]]["test"]) for o in outs) / len(tasks), outs


for label, tasks in [("115 shared tasks", {k: v for k, v in T.items() if k not in EXCLUDE}), ("all 120 tasks", T)]:
    so, oo = solved(orig, tasks); sv, ov = solved(ver, tasks)
    print(f"{label}: without {so:.2f}% ({len(oo)} outputs) -> with verifier {sv:.2f}% ({len(ov)})  gained {sorted(ov - oo)}  lost {sorted(oo - ov)}")
print("outputs changed:", sum(orig[k][i] != ver[k][i] for k in orig for i in range(len(orig[k]))))
