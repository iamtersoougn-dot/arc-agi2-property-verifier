"""Shared loading + NVARC selection score, re-implemented from the NVARC harness write-up.

score_kgmon(grid) = (#augmented views that produced the grid) - mean(augmented NLL over those views)

Paths (override with environment variables):
  ARC_DATA   folder holding ARC-AGI-2 'training/' and 'evaluation/'  (default: data/ARC-AGI-2/data)
  ARC_CANDS  folder holding cands_seed0/ and cands_seed1/              (default: data/cands)
"""
import bz2
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("ARC_DATA", ROOT / "data/ARC-AGI-2/data"))
EVAL_DIR = DATA / "evaluation"
TRAIN_DIR = DATA / "training"
CANDS = Path(os.getenv("ARC_CANDS", ROOT / "data/cands"))
CAND_DIRS = [CANDS / "cands_seed0", CANDS / "cands_seed1"]

# Kaggle's copy of these eval tasks has different test pairs from the public GitHub copy.
EXCLUDE = {"4a21e3da", "abc82100", "b6f77b65", "f560132c", "faa9f03d"}


def load_tasks():
    return {p.stem: json.load(open(p)) for p in sorted(EVAL_DIR.glob("*.json"))}


def load_tasks_shared():
    return {k: v for k, v in load_tasks().items() if k not in EXCLUDE}


def hashable(g):
    g = np.asarray(g)
    return (g.shape, g.tobytes())


def load_run(d):
    """{task_i: [candidate dicts]} from one NVARC candidate export folder."""
    res = {}
    for f in os.listdir(d):
        bk = f.split(".")[0]
        with open(os.path.join(d, f), "rb") as fh:
            for s in pickle.loads(bz2.decompress(fh.read())):
                res.setdefault(bk, []).append(s)
    return res


def load_runs_shared():
    runs = [load_run(d) for d in CAND_DIRS]
    return [{k: v for k, v in r.items() if k.rsplit("_", 1)[0] not in EXCLUDE} for r in runs]


def group(cands):
    """[(grid, n_views, mean_aug_nll)] per distinct grid."""
    g = {}
    for c in cands:
        g.setdefault(hashable(c["solution"]), []).append(c)
    return [(np.asarray(gs[0]["solution"]), len(gs), float(np.mean([np.mean(c["score_aug"]) for c in gs])))
            for gs in g.values()]


def kgmon(item):
    _, n, nll = item
    return n - nll


def weight_map(tasks):
    return {f"{k}_{i}": 1 / len(t["test"]) for k, t in tasks.items() for i in range(len(t["test"]))}


def truth_map(tasks):
    return {f"{k}_{i}": np.array(t["test"][i]["output"]) for k, t in tasks.items() for i in range(len(t["test"]))}
