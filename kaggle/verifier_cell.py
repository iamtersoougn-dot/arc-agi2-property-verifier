# === Property verifier (appended as the LAST cell of the NVARC notebook) ===
# Runs after NVARC has written submission.json. For each test output it checks NVARC's two
# attempts against invariants that hold on EVERY train pair of that task. The invariants were
# calibrated on the 1,000 public ARC-AGI-2 training tasks; only the 12 that were wrong about the
# true answer <=1% of the time are used. If an attempt breaks an invariant AND a non-violating
# candidate exists in NVARC's own candidate pool, the attempt is replaced by the best such
# candidate (ranked by NVARC's own score: #views - mean augmented NLL).
# Attempts that pass are never touched. If anything goes wrong, NVARC's submission is kept as is.
import bz2, glob, json, os, pickle, shutil, traceback
from collections import Counter
import numpy as np


def _colors(g): return set(np.unique(g).tolist())
def _bg(g): return Counter(g.ravel().tolist()).most_common(1)[0][0]
def _ratio(o, i): return (o.shape[0] / i.shape[0], o.shape[1] / i.shape[1])


def _is_sub(small, big):
    h, w = small.shape; H, W = big.shape
    if h > H or w > W: return False
    return any(np.array_equal(big[a:a + h, b:b + w], small) for a in range(H - h + 1) for b in range(W - w + 1))


def _ctx(train):
    ins = [np.array(p["input"]) for p in train]; outs = [np.array(p["output"]) for p in train]
    return {"ratios": {_ratio(o, i) for i, o in zip(ins, outs)},
            "dshape": {(o.shape[0] - i.shape[0], o.shape[1] - i.shape[1]) for i, o in zip(ins, outs)},
            "newc": set().union(*[_colors(o) - _colors(i) for i, o in zip(ins, outs)]),
            "pal": set().union(*[_colors(o) for o in outs])}


def _fg_kept(i, o, c):
    if o.shape != i.shape: return False
    m = i != _bg(i)
    return bool(np.array_equal(o[m], i[m]))


def _cmap(i, o, c):
    if o.shape != i.shape: return False
    f = {}
    for a, b in zip(i.ravel().tolist(), o.ravel().tolist()):
        if f.setdefault(a, b) != b: return False
    return True


PROPS = {
    "same_shape": lambda i, o, c: o.shape == i.shape,
    "const_ratio": lambda i, o, c: len(c["ratios"]) == 1 and _ratio(o, i) in c["ratios"],
    "const_dshape": lambda i, o, c: len(c["dshape"]) == 1 and (o.shape[0] - i.shape[0], o.shape[1] - i.shape[1]) in c["dshape"],
    "out_cols_subset_in": lambda i, o, c: _colors(o) <= _colors(i),
    "out_cols_eq_in": lambda i, o, c: _colors(o) == _colors(i),
    "new_cols_known": lambda i, o, c: (_colors(o) - _colors(i)) <= c["newc"],
    "palette_known": lambda i, o, c: _colors(o) <= (c["pal"] | _colors(i)),
    "fg_kept": _fg_kept,
    "cellwise_colormap": _cmap,
    "color_counts_preserved": lambda i, o, c: Counter(i.ravel().tolist()) == Counter(o.ravel().tolist()),
    "fg_count_nondecreasing": lambda i, o, c: int((o != _bg(o)).sum()) >= int((i != _bg(i)).sum()),
    "in_is_subgrid_of_out": lambda i, o, c: _is_sub(i, o),
}


def _ok(p, i, o, c):
    try: return bool(p(i, o, c))
    except Exception: return False


def active_props(train):
    if len(train) < 2: return [], None  # one example is too little evidence to induce invariants
    c = _ctx(train)
    act = [p for p in PROPS.values() if all(_ok(p, np.array(t["input"]), np.array(t["output"]), c) for t in train)]
    return act, c


def n_violations(test_in, grid, act, c):
    return sum(not _ok(p, test_in, grid, c) for p in act)


def load_pool(cand_dir):
    """{ 'taskid_i': [(grid, n_views, mean_aug_nll)] } from NVARC's raw candidate files."""
    raw = {}
    for fn in os.listdir(cand_dir):
        key = fn.split(".")[0]
        with open(os.path.join(cand_dir, fn), "rb") as fh:
            for s in pickle.loads(bz2.decompress(fh.read())):
                g = np.asarray(s["solution"])
                raw.setdefault(key, {}).setdefault((g.shape, g.tobytes()), []).append(s)
    pool = {}
    for key, groups in raw.items():
        pool[key] = [(np.asarray(gs[0]["solution"]), len(gs), float(np.mean([np.mean(x["score_aug"]) for x in gs])))
                     for gs in groups.values()]
    return pool


def apply_verifier(challenges, submission, pool, log=print):
    changed = 0
    for tid, task in challenges.items():
        act, c = active_props(task["train"])
        if not act or tid not in submission: continue
        for i, test in enumerate(task["test"]):
            if i >= len(submission[tid]): continue
            tin = np.array(test["input"]); slot = submission[tid][i]
            atts = [slot.get("attempt_1"), slot.get("attempt_2")]
            bad = [a is not None and np.array(a).ndim == 2 and n_violations(tin, np.array(a), act, c) > 0 for a in atts]
            if not any(bad): continue
            used = {json.dumps(a) for a, b in zip(atts, bad) if not b}
            clean = [g for g, n, nll in sorted(pool.get(f"{tid}_{i}", []), key=lambda x: -(x[1] - x[2]))
                     if n_violations(tin, g, act, c) == 0 and json.dumps(g.tolist()) not in used]
            new = list(atts)
            for j in (0, 1):
                if bad[j] and clean:
                    new[j] = clean.pop(0).tolist()
            if bad[0] and not bad[1] and new[0] == atts[0]:  # no clean replacement: put the clean attempt first
                new = [atts[1], atts[0]]
            if new != atts:
                changed += 1
                log(f"verifier: {tid}_{i} replaced {['a1' if bad[0] else '', 'a2' if bad[1] else '']}")
                slot["attempt_1"], slot["attempt_2"] = new
    return changed


if __name__ == "__main__" and os.path.isdir("/kaggle"):
    try:
        rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
        name = "arc-agi_test_challenges.json" if rerun else "arc-agi_evaluation_challenges.json"
        ch_path = glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]
        challenges = json.load(open(ch_path))
        sub_path = "submission.json" if os.path.exists("submission.json") else "/kaggle/working/submission.json"
        submission = json.load(open(sub_path))
        shutil.copy(sub_path, "/kaggle/working/submission_nvarc_original.json")
        pool = load_pool("/kaggle/inference_outputs")
        n = apply_verifier(challenges, submission, pool)
        json.dump(submission, open(sub_path, "w"))
        if os.path.abspath(sub_path) != "/kaggle/working/submission.json" and os.path.isdir("/kaggle/working"):
            json.dump(submission, open("/kaggle/working/submission.json", "w"))
        print(f"verifier: {n} test outputs changed; original kept as submission_nvarc_original.json")
    except Exception:
        traceback.print_exc()
        print("verifier failed; NVARC submission left unchanged")
