"""Property verifier: induce invariants that hold on EVERY train pair, check candidates against them.

Each property is f(inp, out, ctx) -> bool. ctx carries task-level facts learned from the
train pairs (e.g. the palette of allowed new colors). A property is "active" for a task only
if it holds on all train pairs; a candidate violates it if it fails on (test_input, candidate).
"""
from collections import Counter

import numpy as np


def bg(g):
    return Counter(g.ravel().tolist()).most_common(1)[0][0]


def colors(g):
    return set(np.unique(g).tolist())


def is_subgrid(small, big):
    h, w = small.shape
    H, W = big.shape
    if h > H or w > W:
        return False
    for i in range(H - h + 1):
        for j in range(W - w + 1):
            if np.array_equal(big[i:i + h, j:j + w], small):
                return True
    return False


def ratio(o, i):
    return (o.shape[0] / i.shape[0], o.shape[1] / i.shape[1])


# --- task-level context learned from train pairs --------------------------------------
def learn_ctx(train):
    ins = [np.array(p["input"]) for p in train]
    outs = [np.array(p["output"]) for p in train]
    new_cols = [colors(o) - colors(i) for i, o in zip(ins, outs)]
    return {
        "out_shapes": {o.shape for o in outs},
        "ratios": {ratio(o, i) for i, o in zip(ins, outs)},
        "new_colors_union": set().union(*new_cols),
        "out_palette": set().union(*[colors(o) for o in outs]),
        "n_out_colors": {len(colors(o)) for o in outs},
        "dshape": {(o.shape[0] - i.shape[0], o.shape[1] - i.shape[1]) for i, o in zip(ins, outs)},
    }


# --- properties ---------------------------------------------------------------------------
def same_shape(i, o, c):
    return o.shape == i.shape


def const_out_shape(i, o, c):
    return len(c["out_shapes"]) == 1 and o.shape in c["out_shapes"]


def const_ratio(i, o, c):
    return len(c["ratios"]) == 1 and ratio(o, i) in c["ratios"]


def const_dshape(i, o, c):
    return len(c["dshape"]) == 1 and (o.shape[0] - i.shape[0], o.shape[1] - i.shape[1]) in c["dshape"]


def out_cols_subset_in(i, o, c):
    return colors(o) <= colors(i)


def out_cols_eq_in(i, o, c):
    return colors(o) == colors(i)


def new_cols_known(i, o, c):
    # any color the output introduces must be one some train output introduced
    return (colors(o) - colors(i)) <= c["new_colors_union"]


def palette_known(i, o, c):
    return colors(o) <= (c["out_palette"] | colors(i))


def n_out_colors_const(i, o, c):
    return len(c["n_out_colors"]) == 1 and len(colors(o)) in c["n_out_colors"]


def bg_preserved(i, o, c):
    return bg(o) == bg(i)


def fg_kept(i, o, c):
    # same shape, and every non-background input cell is unchanged
    if o.shape != i.shape:
        return False
    m = i != bg(i)
    return bool(np.array_equal(o[m], i[m]))


def only_bg_changes(i, o, c):  # alias with stricter reading: changes happen only on bg cells
    return fg_kept(i, o, c) and bool((o != i).any())


def bg_kept(i, o, c):
    # same shape, every background input cell is unchanged (only objects edited)
    if o.shape != i.shape:
        return False
    m = i == bg(i)
    return bool(np.array_equal(o[m], i[m]))


def cellwise_colormap(i, o, c):
    # out = f(in) cell by cell for some color function f
    if o.shape != i.shape:
        return False
    f = {}
    for a, b in zip(i.ravel().tolist(), o.ravel().tolist()):
        if f.setdefault(a, b) != b:
            return False
    return True


def color_counts_preserved(i, o, c):
    return Counter(i.ravel().tolist()) == Counter(o.ravel().tolist())


def fg_count_nondecreasing(i, o, c):
    return int((o != bg(o)).sum()) >= int((i != bg(i)).sum())


def fg_count_nonincreasing(i, o, c):
    return int((o != bg(o)).sum()) <= int((i != bg(i)).sum())


def out_is_subgrid(i, o, c):
    return is_subgrid(o, i)


def sym_lr(i, o, c):
    return np.array_equal(o, o[:, ::-1])


def sym_ud(i, o, c):
    return np.array_equal(o, o[::-1, :])


def sym_diag(i, o, c):
    return o.shape[0] == o.shape[1] and np.array_equal(o, o.T)


def in_is_subgrid_of_out(i, o, c):
    return is_subgrid(i, o)


PROPS = [same_shape, const_out_shape, const_ratio, const_dshape, out_cols_subset_in, out_cols_eq_in,
         new_cols_known, palette_known, n_out_colors_const, bg_preserved, fg_kept, bg_kept,
         cellwise_colormap, color_counts_preserved, fg_count_nondecreasing, fg_count_nonincreasing,
         out_is_subgrid, sym_lr, sym_ud, sym_diag, in_is_subgrid_of_out]


def active_props(train, props=PROPS):
    ctx = learn_ctx(train)
    act = []
    for p in props:
        ok = True
        for pair in train:
            i, o = np.array(pair["input"]), np.array(pair["output"])
            try:
                if not p(i, o, ctx):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            act.append(p)
    return act, ctx


def violations(test_in, cand, act, ctx):
    bad = []
    for p in act:
        try:
            if not p(test_in, cand, ctx):
                bad.append(p.__name__)
        except Exception:
            bad.append(p.__name__)
    return bad
