#!/usr/bin/env python3
"""
npviz - type numpy code on the left, see every array as sheets of paper on the right.

    pip install numpy PyQt6 moderngl
    python npviz.py

How it works
  * Write normal numpy code in the editor (np is already imported):
        a = np.arange(24)
        b = a.reshape(2, 3, 4)
        a.shape = (6, 4)
        b.T                      # a bare expression is shown too
        a.shape                  # non-array results print in the console
  * Every array your code creates or changes becomes a "step". Click a step to watch
    it animate out of the array it came from.
  * Each element is a thin sheet. Layout reads like print(a):
        last axis -> columns (left to right)
        2nd-last  -> rows (top to bottom)
        3rd-last  -> pages (front to back)
        4th-last+ -> whole blocks side by side / below / behind, and so on
  * Colour "origin" (default) gives each element a colour from where it started and
    keeps that colour as it moves, so you can follow it through reshape / .T / flatten.
  * Hover a sheet: its index, value, and which element it came from.

Built for big arrays:
  * positions are computed on the GPU from each element's index (8 bytes per element
    uploaded for a still array, 24 per element while animating)
  * your code runs on a background thread, and only the lines after the first edited
    line are re-run (unchanged lines keep their results, random numbers included)
  * arrays are only copied before a line that could change them in place, so views
    (reshape, .T, slicing) cost no memory
  * the "how did the elements move" analysis only runs for the step you look at
  * up to 300k elements are drawn as 3D sheets, up to MAX_SHOW (8M) as GPU squares,
    beyond that every k-th element per axis is shown

Keys: Ctrl+Enter run (live mode runs by itself), in the 3D view: space replay, f fit, r reset rotation.
Mouse in 3D: left-drag rotate, right-drag pan, wheel zoom, double-click fit.
"""
import ast
import contextlib
import io
import itertools
import math
import queue
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import moderngl

warnings.simplefilter("ignore")
H_, V_ = QtCore.Qt.Orientation.Horizontal, QtCore.Qt.Orientation.Vertical
MONO = "Consolas" if sys.platform == "win32" else ("Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")
MAX_SHOW = 8_000_000        # elements drawn before subsampling kicks in
CUBE_MAX = 300_000          # above this, GPU squares instead of 3D sheets
MAX_LABELS = 3000           # numbers on sheets only for arrays up to this size
MAXD = 12                   # max dimensions the GPU layout handles
TEX_W = 4096                # per-element colour/value data lives in a TEX_W-wide float texture


def mono_font(px=12):
    f = QtGui.QFont(MONO)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    f.setPixelSize(px)
    return f


# =========================================================================== examples
EXAMPLES = {
    "reshape basics": """\
a = np.arange(24)          # 24 numbers in one row
b = a.reshape(4, 6)        # same 24, cut into 4 rows of 6
c = a.reshape(2, 3, 4)     # 2 pages, each 3 rows x 4 columns
d = c.reshape(6, -1)       # -1 means "work it out" -> (6, 4)
e = c.reshape(-1)          # back to one row
c.shape                    # prints (2, 3, 4) in the console
""",
    "flatten / ravel": """\
a = np.arange(12).reshape(3, 4)
f = a.flatten()            # copy, read row by row (C order)
r = a.ravel()              # same order, but a view (no copy) when possible
fF = a.flatten(order='F')  # read column by column (Fortran order)
tf = a.T.flatten()         # transpose first -> same as order='F'
""",
    "transpose vs reshape": """\
a = np.arange(6).reshape(2, 3)
t = a.T                    # rows become columns: elements MOVE
r = a.reshape(3, 2)        # same shape as a.T, but elements keep their order!
""",
    "in-place .shape": """\
a = np.arange(12)
a.shape = (3, 4)           # changes a itself, no copy
a.shape = (2, 2, 3)
a.shape = (-1,)
""",
    "views share memory": """\
a = np.arange(12).reshape(3, 4)
r = a.ravel()              # a view of a
c = a.flatten()            # a copy of a
a[0, 0] = 100              # changes a ... and r (shared memory), not c
""",
    "3D axes": """\
a = np.arange(24).reshape(2, 3, 4)   # (pages, rows, columns)
s = np.swapaxes(a, 0, 2)             # (4, 3, 2)
m = np.moveaxis(a, 0, -1)            # (3, 4, 2): first axis goes last
p = a.transpose(1, 0, 2)             # (3, 2, 4)
""",
    "add / remove axes": """\
a = np.arange(4)
row = a[np.newaxis, :]     # (1, 4): a row
col = a[:, np.newaxis]     # (4, 1): a column
e = np.expand_dims(a, 0)   # same as a[None]
s = np.squeeze(col)        # drop length-1 axes -> (4,)
""",
    "indexing & slicing": """\
a = np.arange(24).reshape(4, 6)
row = a[1]                 # one row
col = a[:, 2]              # one column
blk = a[1:3, 2:5]          # a block
odd = a[:, ::2]            # every other column
big = a[a > 15]            # boolean mask -> always 1D
""",
    "stack / concatenate": """\
a = np.arange(6).reshape(2, 3)
b = np.arange(6, 12).reshape(2, 3)
v = np.concatenate([a, b], axis=0)   # (4, 3)
h = np.concatenate([a, b], axis=1)   # (2, 6)
s = np.stack([a, b])                 # (2, 2, 3): new axis in front
""",
    "sum along an axis": """\
a = np.arange(24).reshape(2, 3, 4)
s0 = a.sum(axis=0)         # squash the pages   -> (3, 4)
s1 = a.sum(axis=1)         # squash the rows    -> (2, 4)
s2 = a.sum(axis=2)         # squash the columns -> (2, 3)
k = a.sum(axis=1, keepdims=True)   # (2, 1, 4)
""",
    "big: 6 million elements": """\
rng = np.random.default_rng(0)
v = rng.random((200, 100, 100, 3))   # 200 frames of 100x100 RGB
t = v.transpose(0, 3, 1, 2)          # channels first
f = v.reshape(200, -1)               # one row per frame
g = v.mean(axis=-1)                  # greyscale
s = v[::2]                           # every other frame
""",
}

# =========================================================================== colour maps
_CMAPS = {
    "turbo": [(48, 18, 59), (70, 107, 227), (40, 187, 236), (49, 242, 153), (162, 252, 60),
              (237, 208, 58), (251, 128, 34), (208, 47, 8), (122, 4, 3)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)],
    "paper": [(70, 70, 90), (230, 226, 210)],
}


def _lut(stops, n=256):
    s = np.asarray(stops, np.float64)
    xp, x = np.linspace(0, 1, len(s)), np.linspace(0, 1, n)
    return np.stack([np.interp(x, xp, s[:, c]) for c in range(3)], 1).astype(np.uint8)


LUTS = {k: _lut(v) for k, v in _CMAPS.items()}
AXIS_COLS = ["#ff5f5f", "#5fdf5f", "#5f9fff", "#ffbf3f", "#df5fff", "#3fdfdf", "#ff8fbf", "#bfbf5f"]

# =========================================================================== element mapping (pure numpy)
REDUCERS = [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min), ("prod", np.prod),
            ("std", np.std), ("var", np.var), ("median", np.median), ("any", np.any), ("all", np.all),
            ("argmax", np.argmax), ("argmin", np.argmin)]


class _KeyGrab:
    def __getitem__(self, k):
        return k


def numeric(x):
    return isinstance(x, np.ndarray) and (np.issubdtype(x.dtype, np.number) or x.dtype == bool)


def _as_array(v):
    """arrays as they are; numpy scalars (tot = a.sum()) as 0-d arrays; anything else None."""
    if numeric(v):
        return v
    if isinstance(v, np.generic) and (np.issubdtype(type(v), np.number) or isinstance(v, np.bool_)):
        return np.asarray(v)
    return None


def displayable(raw):
    if np.iscomplexobj(raw):
        return np.abs(raw)
    if raw.dtype == bool:
        return raw.view(np.uint8)
    return raw


def _same(x, y):
    """exact equality (nan == nan), cheap shape/dtype checks first."""
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    try:
        return bool(np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"))
    except Exception:
        return False


def _vals_equal(x, y):
    if x.shape != y.shape:
        return False
    try:
        eq = x == y
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            eq |= np.isnan(x) & np.isnan(y)
        return bool(np.all(eq))
    except Exception:
        return False


def _check_prov(prov, pall, R):
    if prov.shape != (R.size,) or prov.size == 0 or prov.min() < -1 or prov.max() >= pall.size:
        return None
    ok = prov >= 0
    if ok.all():
        return prov if _vals_equal(pall[prov], R.reshape(-1)) else None
    if not ok.any() or not _vals_equal(pall[prov[ok]], R.reshape(-1)[ok]):
        return None
    return prov


def _quiet_eval(expr, env):
    with contextlib.redirect_stdout(io.StringIO()):
        return eval(compile(expr, "<trace>", "eval"), env)


def _trace(expr, env, names, srcs, R):
    """Re-run expr with the source arrays replaced by element ids. Exact for anything that only
    moves / copies elements (reshape, .T, flatten, slicing, stack, concatenate, repeat, ...)."""
    sizes = [s.size for s in srcs]
    N = sum(sizes)
    if N == 0 or R.size == 0 or R.size > 64 * N + 64:
        return None
    it = np.int32 if N < 2**31 - 2 else np.int64
    offs = np.cumsum([0] + sizes)

    def run(labels):
        e = dict(env)
        for k, (nm, s) in enumerate(zip(names, srcs)):
            e[nm] = labels[offs[k]:offs[k + 1]].reshape(s.shape)
        try:
            T = np.asarray(_quiet_eval(expr, e))
        except Exception:
            return None
        if T.shape != R.shape or T.dtype.kind not in "iu":
            return None
        return T.reshape(-1)

    T = run(np.arange(1, N + 1, dtype=it))
    if T is None:
        return None
    T = T.astype(it, copy=False)
    T -= 1
    pall = np.concatenate([s.reshape(-1) for s in srcs]) if len(srcs) > 1 else srcs[0].reshape(-1)
    prov = _check_prov(T, pall, R)
    if prov is None:
        return None
    # second run with scrambled ids (an affine bijection, no big random permutation needed):
    # real data movement relabels consistently, arithmetic that happened to look like ids doesn't
    m = 1_000_003 if N % 1_000_003 else 999_983
    c = 12345 % N
    lab2 = ((np.arange(N, dtype=np.int64) * m + c) % N + 1).astype(it)
    T2 = run(lab2)
    if T2 is None:
        return None
    pos = np.arange(prov.size) if prov.size <= 400_000 else \
        np.random.default_rng(7).integers(0, prov.size, 400_000)
    pv = prov[pos].astype(np.int64)
    expect = np.where(pv >= 0, (pv * m + c) % N + 1, 0)
    return prov if np.array_equal(T2[pos], expect) else None


def _index_prov(expr, env, name, P, R):
    """name[<key>] where the key depends on values (a[a > 5]): evaluate the key on the real data."""
    m = re.match(r"^\s*" + re.escape(name) + r"\s*\[(.*)\]\s*$", expr, re.S)
    if not m:
        return None
    e = dict(env)
    e[name], e["_K"] = P, _KeyGrab()
    it = np.int32 if P.size < 2**31 else np.int64
    try:
        key = _quiet_eval("_K[" + m.group(1) + "]", e)
        T = np.asarray(np.arange(P.size, dtype=it).reshape(P.shape)[key])
    except Exception:
        return None
    if T.shape != R.shape:
        return None
    return _check_prov(T.reshape(-1), P.reshape(-1), R)


def _perm_prov(P, R):
    """Same multiset of values (sort / shuffle): match values up, ties in order."""
    n = P.size
    if n != R.size or n < 2 or n > 20_000_000 or P.dtype != R.dtype:
        return None
    try:   # cheap rejections before any sorting
        if P.dtype.kind in "iuf":
            if np.nanmin(P) != np.nanmin(R) or np.nanmax(P) != np.nanmax(R):
                return None
            if not np.isclose(np.nansum(P, dtype=np.float64), np.nansum(R, dtype=np.float64)):
                return None
    except Exception:
        pass
    p, r = P.reshape(-1), R.reshape(-1)
    op, orr = np.argsort(p, kind="stable"), np.argsort(r, kind="stable")
    if not _vals_equal(p[op], r[orr]):
        return None
    prov = np.empty(n, np.int64)
    prov[orr] = op
    return prov


def _reduce_axes(P, R, prefer=None):
    """Which axes were collapsed, and by which function? Checked on a small corner of the
    output (first few entries of every kept axis), so it is cheap even for huge arrays.
    prefer: the axes written in the code (axis=...), tried first."""
    nd = P.ndim
    if nd == 0 or R.size >= P.size or R.size == 0:
        return None
    cands = []
    for r in range(1, nd + 1):
        for ax in itertools.combinations(range(nd), r):
            keep = tuple(s for i, s in enumerate(P.shape) if i not in ax)
            kd = tuple(1 if i in ax else s for i, s in enumerate(P.shape))
            if R.shape == keep:
                cands.append((ax, False))
            elif R.shape == kd:
                cands.append((ax, True))
    if not cands:
        return None
    if prefer is not None:
        cands.sort(key=lambda c: c[0] != tuple(prefer))
    for ax, kd in cands:
        sl_p = tuple(slice(None) if i in ax else slice(0, 3) for i in range(nd))
        Ps = P[sl_p]
        if kd:
            Rs = R[sl_p]
        else:
            Rs = R[tuple(slice(0, 3) for _ in R.shape)]
        Rf = np.asarray(Rs, np.float64)
        for name, f in REDUCERS:
            if name in ("argmax", "argmin") and len(ax) > 1:
                continue
            try:
                v = f(Ps, axis=ax[0] if len(ax) == 1 else ax, keepdims=kd)
                if np.shape(v) == Rf.shape and np.allclose(np.asarray(v, np.float64), Rf, rtol=1e-5, atol=1e-8,
                                                           equal_nan=True):
                    return ax, kd, name
            except Exception:
                pass
    return cands[0] + (None,)


_RED_NAMES = {"sum", "mean", "max", "min", "prod", "std", "var", "median", "any", "all", "argmax", "argmin", "ptp",
              "amax", "amin", "nansum", "nanmean", "nanmax", "nanmin", "nanstd", "nanvar", "nanmedian", "nanprod",
              "nanargmax", "nanargmin", "average", "count_nonzero"}


def _calls_on(src, name, attrs):
    """yield (call node, axis node or None, 'method'|'function') for name.attr(...) / np.attr(name, ...)."""
    try:
        tree = ast.parse(src)
    except Exception:
        return
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in attrs):
            continue
        v = node.func.value
        method = isinstance(v, ast.Name) and v.id == name
        func = isinstance(v, ast.Name) and v.id in ("np", "numpy") and bool(node.args) and \
            isinstance(node.args[0], ast.Name) and node.args[0].id == name
        if not (method or func):
            continue
        ax = next((k.value for k in node.keywords if k.arg == "axis"), None)
        if ax is None:
            pos = (node.args[0] if node.args else None) if method else (node.args[1] if len(node.args) > 1 else None)
            ax = pos
        yield node, ax, ("method" if method else "function")


def _reduction_hint(src, name, ndim):
    """-> (written as a reduction?, axes tuple or None)"""
    for node, ax, _ in _calls_on(src or "", name, _RED_NAMES):
        if ax is None:
            return True, tuple(range(ndim))
        try:
            a = ast.literal_eval(ax)
        except Exception:
            return True, None
        if a is None:
            return True, tuple(range(ndim))
        a = (a,) if isinstance(a, int) else tuple(a)
        try:
            return True, tuple(sorted(int(x) % ndim for x in a))
        except Exception:
            return True, None
    return False, None


def _sort_prov(src, name, P, R):
    """np.sort(a, axis=k) / a.sort(axis=k): follow numpy's order within each row/column, so equal
    values never jump to another row."""
    if R.size != P.size or P.ndim == 0:
        return None
    for node, ax, _ in _calls_on(src or "", name, {"sort"}):
        try:
            axis = -1 if ax is None else ast.literal_eval(ax)
            if axis is None:
                prov = np.argsort(P.reshape(-1), kind="stable")
            else:
                ids = np.arange(P.size).reshape(P.shape)
                prov = np.take_along_axis(ids, np.argsort(P, axis=axis, kind="stable"), axis=axis).reshape(-1)
        except Exception:
            return None
        return _check_prov(prov.astype(np.int64), P.reshape(-1), R)
    return None


def reduce_map(pshape, ax, n_out):
    keep = tuple(1 if i in ax else s for i, s in enumerate(pshape))
    it = np.int32 if n_out < 2**31 else np.int64
    return np.broadcast_to(np.arange(n_out, dtype=it).reshape(keep), pshape).reshape(-1)


def relate(expr, env, names, srcs, R, hint=None):
    """How does R relate to the source arrays? -> (kind, prov, red, names_used)
    hint: the source line (used to read axis=... and spot sorts / reductions)."""
    P = srcs[0]
    prov, used_n = None, names[:1]
    src = expr or hint or ""
    is_red, axes = _reduction_hint(src, names[0], P.ndim)
    if is_red and R.size < P.size:                 # written as sum/max/argmax/...: always a reduction
        red = _reduce_axes(displayable(P), displayable(R), prefer=axes)
        if red is not None:
            return "reduce", None, red, used_n
    prov = _sort_prov(src, names[0], P, R)
    if prov is not None:
        return "move", prov, None, used_n
    if expr:
        prov = _index_prov(expr, env, names[0], P, R)
        if prov is None:
            prov = _trace(expr, env, names, srcs, R)
            used_n = names
            if prov is None and len(names) > 1:     # other arrays may be indices (a[idx]): keep them real
                prov = _trace(expr, env, names[:1], srcs[:1], R)
                used_n = names[:1]
    if prov is None:
        used_n = names[:1]
        if R.size == P.size and _vals_equal(P.reshape(-1), R.reshape(-1)):   # a.shape = (...) etc.
            prov = np.arange(P.size, dtype=np.int32 if P.size < 2**31 else np.int64)
        else:
            prov = _perm_prov(P, R)
    if prov is not None:
        return "move", prov, None, used_n
    red = _reduce_axes(displayable(P), displayable(R), prefer=axes)
    if red is not None:
        return "reduce", None, red, used_n
    if R.shape == P.shape:
        return "elementwise", None, None, used_n
    return "new", None, None, used_n


# =========================================================================== running user code
@dataclass
class Step:
    name: str                  # variable name (or the expression for bare expressions)
    label: str                 # source line(s)
    line: int
    raw: np.ndarray            # the array - the live object until something could change it, then a copy
    live: bool = True
    expr: str = None           # expression that produced it (for the element mapping)
    cand: list = field(default_factory=list)   # [(name, step_idx)] candidate source arrays
    env: dict = None           # other variables at that moment (shallow)
    mem: str = ""
    # filled in lazily by analyse()
    analysed: bool = False
    kind: str = ""
    srcs: list = field(default_factory=list)   # [(name, step_idx)] actually used, in prov order
    prov: np.ndarray = None
    red: tuple = None
    desc: str = ""
    _origin: np.ndarray = None
    _range: tuple = None
    _raw_range: tuple = None

    @property
    def parent(self):
        return self.cand[0][1] if self.cand else -1


def analyse(steps, i):
    """Work out how step i relates to its source array (only done for steps you look at)."""
    st = steps[i]
    if st.analysed:
        return
    for _, j in st.cand:
        analyse(steps, j)
    if st.cand:
        names = [n for n, _ in st.cand]
        srcs = [steps[j].raw for _, j in st.cand]
        env = dict(st.env)
        for n, j in st.cand:
            env[n] = steps[j].raw
        kind, prov, red, used = relate(st.expr, env, names, srcs, st.raw, hint=st.label)
        st.kind, st.prov, st.red = kind, prov, red
        st.srcs = [(n, j) for n, j in st.cand if n in used]
    else:
        st.kind = "new"
    st.desc = describe(st, steps)
    st.analysed = True


def origin_of(steps, i):
    """colour key per element that follows elements around (computed on demand)."""
    st = steps[i]
    if st._origin is None:
        analyse(steps, i)
        n = st.raw.size
        if st.kind == "move":
            pool = np.concatenate([origin_of(steps, j) for _, j in st.srcs])
            st._origin = np.where(st.prov >= 0, pool[np.maximum(st.prov, 0)], -1).astype(np.float32)
        elif st.kind == "elementwise":
            st._origin = origin_of(steps, st.parent)
        else:
            st._origin = np.linspace(0, 1, n, dtype=np.float32) if n > 1 else np.zeros(n, np.float32)
    return st._origin


def value_range(st):
    if st._range is None:
        d = displayable(st.raw)
        lo = hi = 0.0
        if d.size:
            lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
            if not (math.isfinite(lo) and math.isfinite(hi)):
                f = d[np.isfinite(d)]
                lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
        st._raw_range = (lo, hi) if d.size else None
        st._range = (lo, hi if hi > lo else lo + 1.0)
    return st._range


def shared_range(steps):
    """one min..max over every step, so "value" colours can be compared between steps."""
    rs = [st._raw_range for st in steps if value_range(st) and st._raw_range is not None]
    if not rs:
        return (0.0, 1.0)
    lo, hi = min(r[0] for r in rs), max(r[1] for r in rs)
    return (lo, hi if hi > lo else lo + 1.0)


def _mem_note(new, olds, names, inplace):
    if inplace:
        return f"{names[0]} itself was changed in place (same object)"
    for o, nm in zip(olds, names):
        try:
            if np.shares_memory(new, o, max_work=10_000):
                return f"VIEW of {nm}: shares {nm}'s memory, nothing was copied"
        except Exception:
            if np.may_share_memory(new, o):
                return f"probably a VIEW of {nm} (shares memory)"
    return "new memory (a copy)" if olds else "new array"


def describe(st, steps):
    R = st.raw
    if not st.srcs:
        return f"NEW ARRAY  shape {R.shape}"
    P = steps[st.srcs[0][1]].raw
    pn = st.srcs[0][0]
    ins = ", ".join(f"{n} {steps[i].raw.shape}" for n, i in st.srcs)
    sh = f"{ins}  ->  {st.name} {R.shape}"
    if st.kind == "move":
        prov, n0 = st.prov, P.size
        if prov.size == n0 and prov[0] == 0 and np.array_equal(prov, np.arange(n0)):
            if P.shape == R.shape:
                return f"SAME  {sh}\nNothing moved, nothing changed."
            return (f"RESHAPE  {sh}\nSame {n0} elements in the SAME order: numpy reads them row by row "
                    f"(last axis fastest) and just cuts that row into the new shape.")
        sizes = [steps[i].raw.size for _, i in st.srcs]
        cnt = np.bincount(prov[prov >= 0].astype(np.intp), minlength=sum(sizes))   # intp: 32-bit in the browser
        used0 = int(np.count_nonzero(cnt[:n0]))
        dup, new = int(np.count_nonzero(cnt > 1)), int(np.count_nonzero(prov < 0))
        others = []
        off = n0
        for (nm, _), sz in zip(st.srcs[1:], sizes[1:]):
            k = int(np.count_nonzero(cnt[off:off + sz]))
            if k:
                others.append(f"{k} from {nm}")
            off += sz
        if others:
            head, body = "COMBINE", f"{R.size} elements: {used0} from {pn}, " + ", ".join(others) + "."
        elif R.size == n0 and used0 == n0 and not dup and not new:
            head, body = "REARRANGE", f"Same {n0} elements in a NEW order - every element moved (follow the colours)."
        elif used0 < n0 and not dup and not new:
            head, body = "SELECT", f"Picked {used0} of {n0} elements, the other {n0 - used0} are dropped."
        else:
            head, body = "COPY", f"{R.size} out from {used0} of {n0} in"
            if dup:
                body += f", {dup} copied more than once"
            if new:
                body += f", {new} new (not from {pn})"
            body += "."
        return f"{head}  {sh}\n{body}"
    if st.kind == "reduce":
        ax, kd, name = st.red
        axs = ax[0] if len(ax) == 1 else ax
        k = P.size // max(R.size, 1)
        return (f"REDUCE over axis {axs}  {sh}\nEach output = {name or 'f'}( {k} elements along axis {axs} )"
                + (" - keepdims leaves a length-1 axis" if kd else " - that axis disappears") + ".")
    if st.kind == "elementwise":
        return f"ELEMENTWISE  {sh}\nNothing moved, the values changed."
    return f"NEW ARRAY  {sh}\nNo element-to-element link found."


# statements that cannot change an existing array in place don't need any copies
_MUTATING = {"sort", "fill", "resize", "put", "itemset", "partition", "setflags", "byteswap", "setfield",
             "shuffle", "copyto", "place", "putmask", "fill_diagonal", "__setitem__", "__iadd__", "__isub__",
             "__imul__", "__itruediv__", "update", "append", "extend", "insert", "pop", "clear", "remove"}
_SAFE_CALLS = {"print", "len", "range", "list", "tuple", "int", "float", "abs", "min", "max", "sum", "sorted",
               "round", "str", "repr", "type", "isinstance", "zip", "enumerate", "slice", "bool", "divmod",
               "pow", "dict", "set", "any", "all", "reversed", "map", "filter", "complex", "hash", "id"}


def _safety(node):
    """'safe': can't change existing arrays; 'local': may change the arrays it names; 'all': anything."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
        return "safe"
    if isinstance(node, ast.Expr) or (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)):
        root, level = node.value, "safe"
    else:
        root, level = node, "local"
    for n in ast.walk(root):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                          ast.ClassDef, ast.NamedExpr)):
            return "all"
        if isinstance(n, ast.Call):
            if any(k.arg == "out" for k in n.keywords):
                level = "local"
            f = n.func
            if isinstance(f, ast.Attribute):
                if f.attr in _MUTATING:
                    level = "local"
            elif isinstance(f, ast.Name):
                if f.id not in _SAFE_CALLS:
                    return "all"
            else:
                return "all"
    return level


def _names_in(node, env):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in out and numeric(env.get(n.id)):
            out.append(n.id)
    return out[:4]


@dataclass
class _Snap:
    ns: dict
    latest: dict
    nsteps: int
    ncon: int


class Runner:
    """Runs code statement by statement and records every array change as a Step.
    Keeps the state after every statement, so a re-run only executes from the first changed line."""

    def __init__(self, max_steps=400, timeout=10.0):
        self.max_steps = max_steps
        self.timeout = timeout
        self.segs, self.snaps, self.steps, self.console = [], [], [], []

    def _fresh(self):
        return {"np": np, "numpy": np, "__name__": "__npviz__"}

    def _restore(self, k):
        """state after statement k (1-based) of the previous run, or None if it can't be trusted."""
        sn = self.snaps[k - 1]
        ns, latest = dict(sn.ns), dict(sn.latest)
        steps = self.steps[:sn.nsteps]
        for name, (obj, si, *_) in latest.items():
            st = steps[si]
            if st.raw is obj:            # never protected -> nothing could have changed it
                continue
            if not _same(obj, st.raw):   # a later line of the previous run changed it: put it back in place
                try:
                    if obj.size != st.raw.size or not obj.flags.writeable:
                        return None
                    if obj.shape != st.raw.shape:
                        obj.shape = st.raw.shape
                    obj[...] = st.raw
                except Exception:
                    return None
            st.raw, st.live = obj, True  # identical again: share the live object, drop the copy
        return ns, latest, steps, self.console[:sn.ncon]

    def run(self, code):
        """-> (steps, console lines, error line or None, statements re-used)"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, [f"line {e.lineno}: SyntaxError: {e.msg}"], e.lineno, 0
        nodes = tree.body
        segs = [((ast.get_source_segment(code, n) or ""), n.lineno) for n in nodes]
        k = 0
        while k < min(len(segs), len(self.segs), len(self.snaps)) and segs[k] == self.segs[k]:
            k += 1
        state = None
        while k > 0 and state is None:
            state = self._restore(k)
            if state is None:
                k = 0
        if state is None:
            ns, latest, steps, console = self._fresh(), {}, [], []
        else:
            ns, latest, steps, console = state
        reused = k
        snaps = self.snaps[:k]
        # objects referenced by steps that still hold the live object (not copied)
        shared = {}
        for i, st in enumerate(steps):
            if st.live:
                shared.setdefault(id(st.raw), (st.raw, []))[1].append(i)
        out = io.StringIO()
        err = None

        def flush():
            s = out.getvalue()
            if s:
                console.extend(s.rstrip("\n").split("\n"))
                out.seek(0)
                out.truncate()

        def add(name, label, line, val, expr, src_names, before, inplace=False):
            if len(steps) >= self.max_steps:
                return None
            cand = [(n, latest[n][1]) for n in src_names if n in latest]
            st = Step(name, label, line, val, expr=expr, cand=cand, env=before)
            st.mem = _mem_note(val, [before[n] for n, _ in cand if n in before], [n for n, _ in cand], inplace) \
                if cand else "new array"
            steps.append(st)
            shared.setdefault(id(val), (val, []))[1].append(len(steps) - 1)
            return len(steps) - 1

        def protect(objs):
            """copy arrays that the next statement might change (and everything sharing their memory)."""
            done = []
            for key, (obj, idxs) in list(shared.items()):
                if any(obj is o or np.may_share_memory(obj, o) for o in objs):
                    cp = np.array(obj, copy=True)
                    for i in idxs:
                        steps[i].raw, steps[i].live = cp, False
                    del shared[key]
                    done.append((obj, cp, idxs))
            return done

        deadline = time.perf_counter() + self.timeout

        def tracer(frame, event, arg):        # only your own code is traced, numpy internals are not
            if time.perf_counter() > deadline:
                raise TimeoutError(f"stopped after {self.timeout:.0f} s - is there an infinite loop?")
            return tracer if frame.f_code.co_filename.startswith("line ") else None

        old_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            for si in range(k, len(nodes)):
                node = nodes[si]
                seg = segs[si][0]
                before = dict(ns)
                level = _safety(node)
                guarded = []
                if level == "all":
                    guarded = protect([o for o, _ in shared.values()])
                elif level == "local":
                    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                    guarded = protect([ns[n] for n in names if isinstance(ns.get(n), np.ndarray)])
                try:
                    with contextlib.redirect_stdout(out):
                        if isinstance(node, ast.Expr):
                            val = eval(compile(ast.Expression(node.value), f"line {node.lineno}", "eval"), ns)
                        else:
                            exec(compile(ast.Module([node], type_ignores=[]), f"line {node.lineno}", "exec"), ns)
                            val = None
                except Exception as e:
                    flush()
                    console.append(f"line {node.lineno}: {type(e).__name__}: {e}")
                    err = node.lineno
                    break
                flush()
                if isinstance(node, ast.Expr):
                    if isinstance(val, np.generic) and not isinstance(val, (np.str_, np.bytes_)):
                        val = np.asarray(val)
                    if numeric(val):
                        add(seg if len(seg) <= 24 else "out", seg, node.lineno, val, seg, _names_in(node.value, before), before)
                    elif val is not None:
                        console.append(f"{seg}  ->  {val!r}")
                # guarded arrays that did NOT change go back to being shared (copy freed)
                changed_objs = set()
                for obj, cp, idxs in guarded:
                    if _same(obj, cp):
                        for i in idxs:
                            steps[i].raw, steps[i].live = obj, True
                        shared.setdefault(id(obj), (obj, []))[1].extend(idxs)
                    else:
                        changed_objs.add(id(obj))
                # every ndarray variable that is new or changed becomes a step
                for name, v0 in list(ns.items()):
                    v = None if name.startswith("_") else _as_array(v0)
                    if v is None:
                        continue
                    if v is not v0:                      # a numpy scalar: compare by the scalar object
                        old = latest.get(name)
                        if old is not None and old[2] is v0:
                            continue
                    old = latest.get(name)
                    if old is not None and old[0] is v and id(v) not in changed_objs:
                        ost = steps[old[1]]
                        if level == "safe" or ost.live:
                            continue                     # can't have changed
                        if _same(v, ost.raw):            # unchanged: share the live object again
                            ost.raw, ost.live = v, True
                            shared.setdefault(id(v), (v, []))[1].append(old[1])
                            continue
                    expr, src = None, []
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                            isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
                        expr, src = ast.get_source_segment(code, node.value), _names_in(node.value, before)
                    elif old is not None:
                        src = [name]
                    i = add(name, seg, node.lineno, v, expr, src, before, inplace=old is not None and old[0] is v)
                    if i is not None:
                        latest[name] = (v, i, v0)
                for name in [n for n in latest if _as_array(ns.get(n)) is None]:
                    del latest[name]
                snaps.append(_Snap(dict(ns), dict(latest), len(steps), len(console)))
        finally:
            sys.settrace(old_trace)
        self.segs, self.snaps, self.steps, self.console = segs[:len(snaps)], snaps, steps, console
        return steps, list(console), err, reused


def run_code(code):
    """one-off helper (no caching): -> (steps, console, error line)"""
    steps, con, err, _ = Runner().run(code)
    return (steps or []), con, err


# =========================================================================== layout (shared by GPU and CPU)
DIRS = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)


@dataclass
class AxisInfo:
    ax: int
    n: int
    d: int
    level: int
    cols: int
    start: np.ndarray = None
    end: np.ndarray = None
    inner: np.ndarray = None     # size of one block of this level (for placing its arrow)


@dataclass
class Layout:
    nd: int
    n: np.ndarray        # (MAXD,) int32   axis length
    st: np.ndarray       # (MAXD,) int32   element stride (C order)
    cols: np.ndarray     # (MAXD,) int32   wrap width (= n when not wrapped)
    a: np.ndarray        # (MAXD,3) float32 offset per step along the axis
    b: np.ndarray        # (MAXD,3) float32 offset per wrapped row
    c: np.ndarray        # (3,) centre
    axes: list
    half: np.ndarray
    count: int


LAYOUTS = ["numpy order", "layers first"]


def axis_order(nd, order="numpy order"):
    """which axis goes to columns, rows, pages, then outer blocks (repeating columns/rows/pages).
    numpy order:  like print(a) - last axis = columns, 2nd-last = rows, 3rd-last = pages, a[0] is one block.
    layers first: like learnbyvisualize - axis 0 = layers stacked in depth (a[i] is one face), axis 1 = rows,
                  axis 2 = columns, axis 3 = whole cubes side by side (a[..., k] is one cube), axis 4+ further out.
    1D and 2D look the same in both; 3D too (pages, rows, columns) - they differ from 4D on."""
    if order == "layers first" and nd >= 4:
        return [2, 1, 0] + list(range(nd - 1, 2, -1))
    return list(range(nd - 1, -1, -1))


def _grid_cols(k, c0, lo=1):
    """columns for wrapping k blocks into a grid: no empty slots if possible, close to c0."""
    c0 = max(float(c0), 1.0)
    cands = range(max(lo, int(c0 * 0.6)), int(min(k, math.ceil(c0 * 1.6))) + 1)
    if not cands:
        return int(min(k, max(lo, round(c0))))
    return min(cands, key=lambda c: (-(-k // c) * c - k, abs(c - c0)))


def make_layout(shape, gap=1.5, page=1.4, wrap=True, order="numpy order"):
    """position of element i = sum_k (ik % cols_k) * a_k + (ik // cols_k) * b_k - c,
    with ik = (i // st_k) % n_k. The vertex shader evaluates exactly this."""
    nd = len(shape)
    if nd > MAXD:
        raise ValueError(f"can only draw up to {MAXD} dimensions")
    n = np.ones(MAXD, np.int32)
    stv = np.ones(MAXD, np.int32)
    cols = np.ones(MAXD, np.int32)
    A = np.zeros((MAXD, 3), np.float32)
    B = np.zeros((MAXD, 3), np.float32)
    count = int(np.prod(shape, dtype=np.int64)) if nd else 1
    if nd == 0 or count == 0:
        return Layout(nd, n, stv, cols, A, B, np.zeros(3, np.float32), [], np.full(3, .5, np.float32), count)
    ext = np.zeros(3)
    lo = np.zeros(3)
    hi = np.zeros(3)
    infos = []
    for L, ax in enumerate(axis_order(nd, order)):
        k, d, lvl = shape[ax], L % 3, L // 3
        inner = ext.copy()
        e = 1 if d == 0 else 0
        bgap = gap * lvl + 0.25 * float(ext.max())     # gap between whole blocks grows with the block size
        stepd = (page if d == 2 else 1.0) if lvl == 0 else ext[d] + 1 + bgap
        # only whole blocks (4D+) are wrapped into a grid; a 1D array is always one straight line
        do_wrap = wrap and lvl >= 1 and k > 12
        a = stepd * DIRS[d]
        if do_wrap:
            stepe = ext[e] + 1 + (bgap if lvl >= 1 else gap * 0.5)
            cl = _grid_cols(k, math.sqrt(k * stepe / stepd))
            rows = -(-k // cl)
            b = stepe * DIRS[e]
            ext[d] += (cl - 1) * stepd
            ext[e] += (rows - 1) * stepe
        else:
            cl, rows, b = k, 1, np.zeros(3, np.float32)
            ext[d] += (k - 1) * stepd
        corners = np.array([[0, 0], [cl - 1, 0], [0, rows - 1], [cl - 1, rows - 1]], np.float64)
        pts = corners[:, :1] * a + corners[:, 1:] * b
        lo += pts.min(0)
        hi += pts.max(0)
        n[ax], stv[ax], cols[ax], A[ax], B[ax] = k, int(np.prod(shape[ax + 1:], dtype=np.int64)), cl, a, b
        infos.append(AxisInfo(ax, k, d, lvl, cl, inner=inner))
        if L == min(nd, 3) - 1:
            ext0 = ext.copy()                           # size of one block (the first cube)
    c = ((lo + hi) / 2).astype(np.float32)
    # axis arrows: one corner just outside element [0, 0, ...] (top-left-front), every arrow of a level
    # starts there and runs along an edge of the array; outer block levels use a corner further out
    for inf in infos:
        last = inf.cols - 1 if inf.cols > 1 else inf.n - 1
        m = 0.2 + 1.1 * inf.level
        # inner axes: just in front of the top-left corner; outer (block) axes: along the top-BACK edge,
        # so seen from above they pass behind/above the blocks instead of through them
        z = 0.15 + 0.3 * inf.level if (inf.level == 0 or inf.d == 2) else -inf.inner[2] - 0.15 - 0.3 * inf.level
        origin = -c + np.array([-(0.5 + m), 0.5 + m, z])
        if inf.level == 0 and inf.d == 0:
            # columns arrow along the BOTTOM-front edge: along the top its label lands on the pages behind
            origin = -c + np.array([-0.5, -(ext0[1] + 0.5 + m), z])
        off = (last % inf.cols) * A[inf.ax] + (last // inf.cols) * B[inf.ax]
        span = (1.0 if inf.d < 2 else 0.6) + 2 * m
        inf.start = origin
        inf.end = origin + off + span * DIRS[inf.d]
        if inf.cols < inf.n and inf.cols == 1:       # wrapped into a single column: arrow follows the column
            inf.end = origin + off + span * DIRS[1 if inf.d == 0 else 0]
    return Layout(nd, n, stv, cols, A, B, c, sorted(infos, key=lambda x: x.ax),
                  ((hi - lo) / 2 + .5).astype(np.float32), count)


def positions(L, idx):
    """CPU version of the shader layout (used for labels on small arrays)."""
    idx = np.asarray(idx, np.int64)
    p = np.zeros((idx.size, 3)) - L.c
    for k in range(L.nd):
        ik = (idx // int(L.st[k])) % int(L.n[k])
        p += (ik % int(L.cols[k]))[:, None] * L.a[k] + (ik // int(L.cols[k]))[:, None] * L.b[k]
    return p


@dataclass
class Geom:
    sv: np.ndarray        # (m,2) float32: colour key, value in 0..1 (-5 = nan/inf) - a view into pad
    ids: np.ndarray       # display index -> raw flat index (None = identity)
    dshape: tuple
    sub: int
    L: Layout
    pad: np.ndarray = None   # (rows*TEX_W, 2) float32, uploaded as an RG32F texture

    @property
    def axes(self):
        return self.L.axes

    @property
    def half(self):
        return self.L.half


def build_geom(steps, i, cmode, gap, page, wrap=True, maxn=MAX_SHOW, order="numpy order", vrange=None):
    st = steps[i]
    raw = st.raw
    shape = raw.shape
    sub, ids = 1, None
    view = raw
    if raw.size > maxn:
        sub = 2
        while np.prod([-(-s // sub) for s in shape], dtype=np.int64) > maxn:
            sub += 1
        sl = tuple(slice(None, None, sub) for _ in shape)
        view = raw[sl]
        ids = np.arange(raw.size, dtype=np.int64).reshape(shape)[sl].reshape(-1)
    L = make_layout(view.shape, gap, page, wrap, order)
    m = view.size
    lo, hi = vrange if (cmode == "value" and vrange is not None) else value_range(st)
    pad = np.zeros((max(1, -(-m // TEX_W)) * TEX_W, 2), np.float32)
    sv = pad[:m]
    v = sv[:, 1]
    np.subtract(np.asarray(displayable(view), np.float32).reshape(-1), np.float32(lo), out=v)
    v *= np.float32(1.0 / (hi - lo))
    bad = ~np.isfinite(v)
    np.clip(v, 0, 1, out=v)
    if bad.any():
        v[bad] = -5.0
    s = sv[:, 0]
    if cmode == "origin":
        o = origin_of(steps, i)
        s[:] = o if ids is None else o[ids]
    elif cmode in ("value", "value per array"):
        s[:] = v
        if bad.any():
            s[bad] = -1
    else:
        f = np.arange(m, dtype=np.int64) if ids is None else ids
        if cmode == "flat index":
            s[:] = f / max(raw.size - 1, 1)
        else:  # "axis k"
            k = int(cmode.split()[-1])
            if k >= raw.ndim or shape[k] <= 1:
                s[:] = 0
            else:
                stride = int(np.prod(shape[k + 1:], dtype=np.int64))
                s[:] = ((f // stride) % shape[k]) / (shape[k] - 1)
    return Geom(sv, ids, view.shape, sub, L, pad)


def build_transition(P, Q, GP, GQ):
    """-> (refs, ref, dref). refs (k,2) int32: instance k starts as element ref[k] of P and ends as
    element dref[k] of Q (-1 = not there, drawn at scale 0). Positions, colours and values are all
    looked up on the GPU from these two numbers."""
    kind = Q.kind
    n0, nq = len(GP.sv), len(GQ.sv)
    i32 = np.int32
    if GP.sub > 1 or GQ.sub > 1:
        kind = "elementwise" if (kind == "elementwise" and GP.dshape == GQ.dshape) else "fade"
    if kind == "move":
        prov = Q.prov.astype(i32, copy=False)
        newm = (prov < 0) | (prov >= n0)
        ref = np.where(newm, i32(-1), prov).astype(i32, copy=False)
        used = np.zeros(n0, bool)
        used[prov[~newm]] = True
        drop = np.flatnonzero(~used).astype(i32)
        if drop.size:
            ref = np.concatenate([drop, ref])
            dref = np.concatenate([np.full(drop.size, -1, i32), np.arange(nq, dtype=i32)])
        else:
            dref = np.arange(nq, dtype=i32)
    elif kind == "reduce":
        ax, kd, _ = Q.red
        ref, dref = np.arange(n0, dtype=i32), reduce_map(P.raw.shape, ax, Q.raw.size).astype(i32, copy=False)
    elif kind == "elementwise" and n0 == nq:
        ref = dref = np.arange(n0, dtype=i32)
    else:
        ref = np.concatenate([np.arange(n0, dtype=i32), np.full(nq, -1, i32)])
        dref = np.concatenate([np.full(n0, -1, i32), np.arange(nq, dtype=i32)])
    return _refs(ref, dref), ref, dref


def _refs(ref, dref):
    out = np.empty((len(ref), 2), np.int32)
    out[:, 0], out[:, 1] = ref, dref
    return out


# =========================================================================== GL
_LAY_UNI = ("uniform int {P}_nd; uniform int {P}_n[MAXD]; uniform int {P}_st[MAXD]; uniform int {P}_cols[MAXD];"
            " uniform vec3 {P}_a[MAXD]; uniform vec3 {P}_b[MAXD]; uniform vec3 {P}_c;\n")
_LAY_FN = """
vec3 lay{P}(int i) {{
    vec3 p = -{P}_c;
    for (int k = 0; k < MAXD; k++) {{
        if (k >= {P}_nd) break;
        int ik = (i / {P}_st[k]) % {P}_n[k];
        p += float(ik % {P}_cols[k]) * {P}_a[k] + float(ik / {P}_cols[k]) * {P}_b[k];
    }}
    return p;
}}
"""
VS = (f"#define MAXD {MAXD}\n" + _LAY_UNI.format(P="A") + _LAY_UNI.format(P="B")
      + _LAY_FN.format(P="A") + _LAY_FN.format(P="B") + """
in int sref; in int dref;
uniform sampler2D svA; uniform sampler2D svB; uniform int texW;
vec2 fetchA(int i) { return texelFetch(svA, ivec2(i % texW, i / texW), 0).rg; }
vec2 fetchB(int i) { return texelFetch(svB, ivec2(i % texW, i / texW), 0).rg; }
uniform mat4 mvp; uniform float t; uniform float stagger; uniform int n; uniform int u_static;
uniform float size; uniform float thick; uniform int hover;
uniform sampler2D lut; uniform float px_scale; uniform float arc;
out vec3 v_col; flat out int v_id; out vec3 v_loc; out float v_shade;
#ifdef CUBES
in vec3 in_vert; in vec3 in_norm;
#define IID gl_InstanceID
#else
#define IID gl_VertexID
#endif
float vis(float v) { return v < -9.0 ? 0.0 : 1.0; }
vec3 col(float s, float v) {
    if (v < -4.0 && v > -6.0) return vec3(1.0, 0.0, 1.0);
    if (s < 0.0) return vec3(0.42);
    return texture(lut, vec2(clamp(s, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
}
void main() {
    int I = IID;
    vec3 p0 = vec3(0.0), p1 = vec3(0.0);
    float S0, V0, S1, V1;
    if (u_static == 1) {
        vec2 q = fetchB(I);
        p1 = layB(I); p0 = p1; S0 = q.x; S1 = q.x; V0 = q.y; V1 = q.y;
    } else {
        bool a0 = sref >= 0, a1 = dref >= 0;
        vec2 q0 = a0 ? fetchA(sref) : vec2(0.0, -10.0);
        vec2 q1 = a1 ? fetchB(dref) : vec2(0.0, -10.0);
        if (!a0) q0.x = q1.x;
        if (!a1) q1.x = q0.x;
        S0 = q0.x; V0 = q0.y; S1 = q1.x; V1 = q1.y;
        if (a1) p1 = layB(dref);
        p0 = a0 ? layA(sref) : p1;
        if (!a1) p1 = p0;
    }
    float d = stagger * float(I) / float(max(n, 1));
    float k = clamp((t - d) / max(1.0 - stagger, 1e-4), 0.0, 1.0);
    k = k * k * (3.0 - 2.0 * k);
    float sc = mix(vis(V0), vis(V1), k) * size;
    vec3 c0 = col(S0, V0), c1 = col(S1, V1);
    vec3 c = V0 < -9.0 ? c1 : (V1 < -9.0 ? c0 : mix(c0, c1, k));
    if (I == hover) { c = mix(c, vec3(1.0), 0.55); sc = size * 1.15; }
    vec3 p = mix(p0, p1, k);
    vec3 dv = p1 - p0;
    float dl = length(dv);
    if (dl > 1e-3) {   // sideways arc so elements swapping places don't pass through each other
        vec3 side = cross(dv, vec3(0.0, 1.0, 0.0));
        if (length(side) < 1e-3 * dl) side = cross(dv, vec3(1.0, 0.0, 0.0));
        p += normalize(side) * (arc * min(dl, 40.0) * 4.0 * k * (1.0 - k));
    }
    v_col = c; v_id = I;
#ifdef CUBES
    gl_Position = mvp * vec4(p + in_vert * vec3(sc, sc, max(sc * thick, 0.002)), 1.0);
    v_loc = in_vert;
    vec3 LD = normalize(vec3(0.35, 0.7, 0.8));
    v_shade = 0.55 + 0.45 * max(dot(in_norm, LD), 0.0);
#else
    gl_Position = mvp * vec4(p, 1.0);
    gl_PointSize = max(sc * px_scale / gl_Position.w, 1.0);
    v_loc = vec3(0.0); v_shade = 1.0;
#endif
    if (sc <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}
""")
FS = """
in vec3 v_col; flat in int v_id; in vec3 v_loc; in float v_shade;
uniform int id_pass;
out vec4 f;
void main() {
    if (id_pass == 1) {
        int i = v_id + 1;
        f = vec4(float(i & 255), float((i >> 8) & 255), float((i >> 16) & 255), float((i >> 24) & 255)) / 255.0;
        return;
    }
    float sh = v_shade;
#ifdef CUBES
    vec2 a = abs(v_loc.xy) * 2.0;
    if (abs(v_loc.z) > 0.49 && max(a.x, a.y) > 0.93) sh *= 0.55;    // thin border on the sheet's faces
#else
    vec2 q = abs(gl_PointCoord - 0.5) * 2.0;
    sh = max(q.x, q.y) > 0.88 ? 0.55 : 1.0;
#endif
    f = vec4(v_col * sh, 1.0);
}
"""


def _box():
    v = []
    for ax in range(3):
        for sgn in (1, -1):
            nrm = np.zeros(3)
            nrm[ax] = sgn
            u, w = [i for i in range(3) if i != ax]
            quad = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[u], p[w] = sgn * .5, a * .5, b * .5
                quad.append(p)
            for i in (0, 1, 2, 0, 2, 3):
                v.append(np.concatenate([quad[i], nrm]))
    return np.array(v, np.float32)


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(fovy / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])


def look_at(eye, target, up=(0, 1, 0)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def _surface_format():
    f = QtGui.QSurfaceFormat()
    f.setVersion(3, 3)
    f.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    f.setDepthBufferSize(24)
    f.setSamples(4)
    return f


def _gl_context():
    errs = []
    for kw in ({}, {"libgl": "libGL.so.1"}, {"backend": "egl"},
               {"backend": "egl", "libgl": "libGL.so.1", "libegl": "libEGL.so.1"}):
        try:
            return moderngl.create_context(require=330, **kw)
        except Exception as e:
            errs.append(f"{kw}: {e}")
    raise RuntimeError("could not attach moderngl to the Qt OpenGL context:\n" + "\n".join(errs))


def _set_u(prog, name, value):
    u = prog.get(name, None)
    if u is not None:
        u.value = value


def _set_layout(prog, P, L):
    _set_u(prog, f"{P}_nd", int(L.nd))
    for nm, arr in (("n", L.n), ("st", L.st), ("cols", L.cols)):
        u = prog.get(f"{P}_{nm}", None)
        if u is not None:
            u.write(np.ascontiguousarray(arr, np.int32).tobytes())
    for nm, arr in (("a", L.a), ("b", L.b)):
        u = prog.get(f"{P}_{nm}", None)
        if u is not None:
            u.write(np.ascontiguousarray(arr, np.float32).tobytes())
    u = prog.get(f"{P}_c", None)
    if u is not None:
        u.value = tuple(float(x) for x in L.c)


class Overlay(QtWidgets.QWidget):
    """2D text/lines drawn over the GL view (labels on sheets + axis arrows)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.labels, self.guides, self.title, self.phase = [], [], "", ""

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        f = mono_font(13)
        p.setFont(f)
        if self.title:
            p.setPen(QtGui.QColor(220, 220, 220))
            p.drawText(QtCore.QPointF(12, 22), self.title)
        if self.phase:
            p.setPen(QtGui.QColor(255, 200, 90) if self.phase in ("before", "preparing...") else QtGui.QColor(150, 200, 255))
            p.drawText(QtCore.QPointF(12, 42), self.phase)
        f.setPixelSize(12)
        p.setFont(f)
        for x0, y0, x1, y1, text, col in self.guides:
            p.setPen(QtGui.QPen(QtGui.QColor(col), 2))
            p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy)
            if ln > 4:
                ux, uy = dx / ln, dy / ln
                for s in (1, -1):
                    p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x1 - 9 * ux + s * 5 * uy, y1 - 9 * uy - s * 5 * ux))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text)
            ux, uy = ((dx / ln, dy / ln) if ln > 1e-6 else (1.0, 0.0))
            if abs(uy) > abs(ux):          # mostly vertical arrow: label left of a down tip, above an up tip
                tx, ty = (x1 - tw - 8, y1 + 4) if uy > 0 else (x1 - tw / 2, y1 - 8)
            elif ux >= 0:
                tx, ty = x1 + 8, y1 + 4
            else:
                tx, ty = x1 - tw - 8, y1 + 4
            p.drawText(QtCore.QPointF(tx, ty), text)
        last, fm = None, None
        for x, y, text, px in self.labels:
            if px != last:
                f.setPixelSize(px)
                p.setFont(f)
                fm = p.fontMetrics()
                last = px
            pt = QtCore.QPointF(x - fm.horizontalAdvance(text) / 2, y + px * 0.36)
            p.setPen(QtGui.QColor(0, 0, 0, 230))
            p.drawText(pt + QtCore.QPointF(1, 1), text)
            p.setPen(QtGui.QColor(255, 255, 255))
            p.drawText(pt, text)
        p.end()


class View(QOpenGLWidget):
    picked = QtCore.pyqtSignal(int)
    anim_done = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float)
    key = QtCore.pyqtSignal(int)
    dbl = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(True)
        self.overlay = Overlay(self)
        self.ctx = None
        self.yaw, self.pitch, self.dist = 0.38, 0.32, 20.0
        self.target = np.zeros(3)
        self.fov = math.radians(40)
        self.inst, self.inst_static = None, True     # inst: (k,2) int32 refs while animating
        self.lay_a = self.lay_b = None
        self.geo_a = self.geo_b = None
        self._buf, self._vao = None, None
        self._tex = {}                                # id(Geom) -> texture (small LRU)
        self.count = 0
        self.cubes = True
        self.size, self.thick = 0.86, 0.08
        self.t, self.stagger, self.duration, self.arc = 1.0, 0.3, 2.0, 0.15
        self.moving, self._hold = False, 0.0
        self.hover = -1
        self.lut = LUTS["turbo"]
        self._lut_dirty = True
        self._upload_needed = False
        self.label_src = None          # static: (pos (m,3), [text])
        self.anim_labels = None        # moving: (p0, p1, vis0, vis1, texts0, texts1)
        self.axes_info, self.show_axes = [], True
        self._axes_pair = None
        self.static = True
        self.render_ms = self.upload_ms = 0.0
        self._idfbo, self._idbuf, self._idbuf_cache, self._idkey = None, None, None, None
        self._fbo, self._fbo_id = None, None
        self._drag, self._mouse = None, None
        self.rot_locked = False
        self._gen = 0
        self._anim = QtCore.QTimer(self)
        self._anim.setInterval(0)
        self._anim.timeout.connect(self._step_anim)
        self._pick_timer = QtCore.QTimer(self)
        self._pick_timer.setSingleShot(True)
        self._pick_timer.timeout.connect(self._do_pick)
        self._zoom = QtCore.QTimer(self)
        self._zoom.setInterval(15)
        self._zoom.timeout.connect(self._zoom_step)

    # ---------------------------------------------------------------- data
    def set_static(self, G):
        self.lay_a = self.lay_b = G.L
        self.geo_a = self.geo_b = G
        self.inst, self.inst_static = None, True
        self.static, self.t, self.moving = True, 1.0, False
        self.anim_labels, self._axes_pair = None, None
        self.overlay.phase = ""
        self._anim.stop()
        self._upload_needed = True
        self.update()

    def set_transition(self, GA, GB, refs, hold=0.0, labels=None, start=True, axes=None):
        """hold: seconds to show the 'before' state (with labels) before anything moves.
        axes: (axes of the start array, axes of the end array) - arrows switch halfway."""
        self.lay_a, self.lay_b = GA.L, GB.L
        self.geo_a, self.geo_b = GA, GB
        self.inst, self.inst_static = refs, False
        self._axes_pair = axes
        if axes:
            self.axes_info = axes[0]
        self.static, self.t, self.moving = False, 0.0, False
        self._hold = hold
        self.anim_labels = labels
        self.label_src = None
        self.overlay.phase = "before" if hold > 0 else ""
        self._t0 = time.perf_counter()
        self.hover = -1
        self._upload_needed = True
        if start:
            self._anim.start()
        self.update()

    def scrub(self, t):
        self._anim.stop()
        self.t = float(t)
        self.moving = 0.0 < self.t < 1.0
        self.overlay.phase = "scrubbing"
        self._swap_axes()
        self.update()

    def _swap_axes(self):
        if self._axes_pair:
            self.axes_info = self._axes_pair[0] if self.t < 0.5 else self._axes_pair[1]

    def set_cmap(self, name):
        self.lut = LUTS[name]
        self._lut_dirty = True
        self.update()

    def _step_anim(self):
        el = time.perf_counter() - self._t0
        if el < self._hold:
            self.t = 0.0
        else:
            if not self.moving:
                self.moving = True
                self.overlay.phase = "moving"
            self.t = min((el - self._hold) / max(self.duration, 1e-3), 1.0)
        self._swap_axes()
        self.progress.emit(self.t)
        self.update()
        if self.t >= 1.0:
            self._anim.stop()
            self.moving = False
            self.overlay.phase = ""
            self.anim_done.emit()

    # ---------------------------------------------------------------- GL
    def initializeGL(self):
        self.ctx = _gl_context()
        self.progs = {}
        for name, define in (("cubes", "#define CUBES\n"), ("points", "")):
            self.progs[name] = self.ctx.program(vertex_shader="#version 330\n" + define + VS,
                                                fragment_shader="#version 330\n" + define + FS)
        self.box_vbo = self.ctx.buffer(_box().tobytes())
        self.lut_tex = self.ctx.texture((256, 1), 3, self.lut.tobytes())
        self.lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lut_tex.repeat_x = self.lut_tex.repeat_y = False

    def _texture(self, G):
        """per-element (colour key, value) of a Geom as an RG32F texture; a few are kept for re-use."""
        key = id(G)
        if key in self._tex:
            tex = self._tex.pop(key)[0]
        else:
            tex = self.ctx.texture((TEX_W, len(G.pad) // TEX_W), 2, G.pad.tobytes(), dtype="f4")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._tex[key] = (tex, G)               # holding G keeps id(G) unique while cached
        keep = {id(self.geo_a), id(self.geo_b), key}
        for k in list(self._tex):
            if len(self._tex) <= 4:
                break
            if k not in keep:
                self._tex.pop(k)[0].release()
        return tex

    def _upload(self):
        t0 = time.perf_counter()
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        self._gen += 1
        if self.geo_b is None:
            self.count = 0
            return
        self.tex_a, self.tex_b = self._texture(self.geo_a), self._texture(self.geo_b)
        self.count = len(self.geo_b.sv) if self.inst_static else len(self.inst)
        if not self.count:
            return
        self.cubes = self.count <= CUBE_MAX
        content = []
        if not self.inst_static:
            self._buf = self.ctx.buffer(np.ascontiguousarray(self.inst, np.int32))
            content.append((self._buf, "1i 1i/i" if self.cubes else "1i 1i", "sref", "dref"))
        if self.cubes:
            content.insert(0, (self.box_vbo, "3f 3f", "in_vert", "in_norm"))
        self._vao = self.ctx.vertex_array(self.progs["cubes" if self.cubes else "points"], content)
        self.upload_ms = (time.perf_counter() - t0) * 1e3

    def matrices(self):
        dpr = self.devicePixelRatioF()
        w, h = max(int(self.width() * dpr), 1), max(int(self.height() * dpr), 1)
        cp = math.cos(self.pitch)
        eye = self.target + self.dist * np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        P = perspective(self.fov, w / h, max(self.dist * 0.01, 1e-3), self.dist * 6 + 100)
        return P @ look_at(eye, self.target), P, w, h

    def _render(self, prog, mvp, P, h, id_pass):
        prog["mvp"].write(mvp.T.astype("f4").tobytes())
        for k, v in (("t", self.t), ("stagger", 0.0 if self.static else self.stagger), ("size", self.size),
                     ("thick", self.thick), ("px_scale", h * P[1, 1] / 2), ("arc", self.arc)):
            _set_u(prog, k, float(v))
        _set_u(prog, "n", int(self.count))
        _set_u(prog, "u_static", 1 if self.inst_static else 0)
        _set_u(prog, "hover", int(self.hover if (self.static and not id_pass) else -1))
        _set_u(prog, "id_pass", int(id_pass))
        _set_u(prog, "lut", 0)
        _set_u(prog, "svA", 1)
        _set_u(prog, "svB", 2)
        _set_u(prog, "texW", TEX_W)
        _set_layout(prog, "A", self.lay_a)
        _set_layout(prog, "B", self.lay_b)
        self.lut_tex.use(0)
        self.tex_a.use(1)
        self.tex_b.use(2)
        if self.cubes:
            self._vao.render(moderngl.TRIANGLES, instances=self.count)
        else:
            self._vao.render(moderngl.POINTS, vertices=self.count)

    def paintGL(self):
        t0 = time.perf_counter()
        if self._upload_needed:
            self._upload()
            self._upload_needed = False
        if self._lut_dirty:
            self.lut_tex.write(np.ascontiguousarray(self.lut).tobytes())
            self._lut_dirty = False
        mvp, P, w, h = self.matrices()
        fid = self.defaultFramebufferObject()
        if fid != self._fbo_id:
            self._fbo, self._fbo_id = self.ctx.detect_framebuffer(fid), fid
        self._fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.08, 0.09, 1.0, depth=1.0)
        self.ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)   # Qt may leave blending on
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._idbuf = None
        if self.count and self._vao is not None:
            prog = self.progs["cubes" if self.cubes else "points"]
            self._render(prog, mvp, P, h, 0)
            if self._labels_now() is not None:
                key = (mvp.tobytes(), self._gen, self.size, self.thick, w, h, self.t)
                if key != self._idkey or self._idbuf_cache is None:
                    self._render_ids(prog, mvp, P, w, h, read_all=True)
                    self._idbuf_cache, self._idkey = self._idbuf, key
                    self._fbo.use()
                self._idbuf = self._idbuf_cache
        self._update_overlay(mvp, P, w, h)
        self.render_ms = (time.perf_counter() - t0) * 1e3

    def _render_ids(self, prog, mvp, P, w, h, read_all):
        if self._idfbo is None or self._idfbo.size != (w, h):
            if self._idfbo is not None:
                self._idfbo.release()
            self._idfbo = self.ctx.simple_framebuffer((w, h), components=4)
        self._idfbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self._idfbo.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._render(prog, mvp, P, h, 1)
        if read_all:
            b = np.frombuffer(self._idfbo.read(components=4), "<u4").reshape(h, w)[::-1]
            self._idbuf = b.astype(np.int64) - 1

    @staticmethod
    def _project(mvp, pts, w, h):
        hp = np.c_[pts, np.ones(len(pts))] @ mvp.T
        wv = hp[:, 3]
        ok = wv > 1e-6
        ws = np.where(ok, wv, 1)
        return (hp[:, 0] / ws * .5 + .5) * w, (.5 - hp[:, 1] / ws * .5) * h, wv, ok

    def _labels_now(self):
        """-> (positions, texts, scale) of every sheet right now, or None. While animating this
        repeats the vertex shader's maths on the CPU so the numbers ride along with the sheets."""
        if self.static or self.anim_labels is None:
            if not self.static or self.label_src is None:
                return None
            pos, texts = self.label_src
            return pos, texts, np.ones(len(texts))
        p0, p1, vis0, vis1, t0, t1 = self.anim_labels
        n = len(p0)
        if n != self.count:
            return None
        st = self.stagger
        d = st * np.arange(n) / max(n, 1)
        k = np.clip((self.t - d) / max(1.0 - st, 1e-4), 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)
        dv = p1 - p0
        pos = p0 + dv * k[:, None]
        dl = np.linalg.norm(dv, axis=1)
        side = np.cross(dv, [0.0, 1.0, 0.0])
        flat = np.linalg.norm(side, axis=1) < 1e-3 * np.maximum(dl, 1e-9)
        side[flat] = np.cross(dv[flat], [1.0, 0.0, 0.0])
        sn = np.linalg.norm(side, axis=1)
        move = dl > 1e-3
        side[move] /= sn[move, None]
        pos[move] += side[move] * (self.arc * np.minimum(dl[move], 40.0) * 4 * k[move] * (1 - k[move]))[:, None]
        scale = vis0 + (vis1 - vis0) * k
        texts = [b if kk >= 0.5 else a for a, b, kk in zip(t0, t1, k.tolist())]
        return pos, texts, scale

    def _update_overlay(self, mvp, P, w, h):
        dpr = self.devicePixelRatioF()
        ov = self.overlay
        ov.labels, ov.guides = [], []
        if self.show_axes and self.axes_info:
            pts = np.array([p for a in self.axes_info for p in (a.start, a.end)], np.float64)
            x, y, _, ok = self._project(mvp, pts, w, h)
            for i, a in enumerate(self.axes_info):
                if ok[2 * i] and ok[2 * i + 1]:
                    txt = f"axis {a.ax}  ({a.n})" + (f"  as {-(-a.n // a.cols)}x{a.cols} grid" if a.cols < a.n else "")
                    ov.guides.append((x[2 * i] / dpr, y[2 * i] / dpr, x[2 * i + 1] / dpr, y[2 * i + 1] / dpr, txt,
                                      AXIS_COLS[a.ax % len(AXIS_COLS)]))
        lab = self._labels_now()
        if lab is not None and self._idbuf is not None:
            pos, texts, scale = lab
            x, y, wv, ok = self._project(mvp, np.asarray(pos, np.float64), w, h)
            px = scale * self.size * (h * P[1, 1] / 2) / np.maximum(wv, 1e-6) / dpr
            xi, yi = x.astype(np.int64), y.astype(np.int64)
            idx = np.flatnonzero(ok & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (px >= 13))
            if idx.size:
                # always centred on the sheet; if the centre is hidden behind another sheet, no label
                seen = idx[self._idbuf[yi[idx], xi[idx]] == idx]
                for i in seen:
                    t = texts[i]
                    fpx = min(15.0, px[i] * 0.42, px[i] * 1.1 / (max(len(t), 1) * 0.62))
                    if fpx >= 7:
                        ov.labels.append((x[i] / dpr, y[i] / dpr, t, int(fpx)))
                ov.labels.sort(key=lambda r_: r_[3])
        ov.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.overlay.resize(e.size())

    # ---------------------------------------------------------------- picking
    def _do_pick(self):
        if not self.static or not self.count or self._mouse is None or self.ctx is None or self._vao is None:
            return
        dpr = self.devicePixelRatioF()
        x, y = int(self._mouse[0] * dpr), int(self._mouse[1] * dpr)
        mvp, P, w, h = self.matrices()
        if not (0 <= x < w and 0 <= y < h):
            return
        if self._idbuf is not None and self._idbuf.shape == (h, w):
            i = int(self._idbuf[y, x])
        else:
            self.makeCurrent()
            prog = self.progs["cubes" if self.cubes else "points"]
            self._render_ids(prog, mvp, P, w, h, read_all=False)
            b = np.frombuffer(self._idfbo.read(viewport=(x, h - 1 - y, 1, 1), components=4), "<u4")
            i = int(b[0]) - 1
            self.doneCurrent()
        if i != self.hover:
            self.hover = i
            self.update()
        self.picked.emit(i)

    # ---------------------------------------------------------------- camera
    def zoom_to(self, d, secs=0.6):
        self._z = (self.dist, d, time.perf_counter(), secs)
        self._zoom.start()

    def _zoom_step(self):
        d0, d1, t0, secs = self._z
        k = min((time.perf_counter() - t0) / secs, 1.0)
        self.dist = d0 + (d1 - d0) * k * k * (3 - 2 * k)
        self.update()
        if k >= 1.0:
            self._zoom.stop()

    def reset_rotation(self, yaw=0.38, pitch=0.32):
        self.yaw, self.pitch = yaw, pitch
        self.update()

    def fit_dist(self, half):
        """distance at which a sphere around the array fits both the width and the height of the view."""
        aspect = max(self.width(), 1) / max(self.height(), 1)
        half_fov = min(self.fov / 2, math.atan(math.tan(self.fov / 2) * aspect))
        return (float(np.linalg.norm(half)) + 0.5) / math.sin(half_fov)

    def fit(self, half, reset_angles=False):
        self._zoom.stop()
        self.target = np.zeros(3)
        self.dist = self.fit_dist(half)
        if reset_angles:
            self.yaw, self.pitch = 0.38, 0.32
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.position(), e.buttons(), e.modifiers())

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.dbl.emit()

    def keyPressEvent(self, e):
        self.key.emit(e.key())

    def mouseMoveEvent(self, e):
        p = e.position()
        if self._drag is None:
            self._mouse = (p.x(), p.y())
            self._pick_timer.start(12)
            return
        p0, btn, mod = self._drag
        dx, dy = p.x() - p0.x(), p.y() - p0.y()
        self._drag = (p, btn, mod)
        B = QtCore.Qt.MouseButton
        if self.rot_locked or (btn & B.RightButton) or (btn & B.MiddleButton) or \
                (mod & QtCore.Qt.KeyboardModifier.ShiftModifier):
            s = self.dist * 2 * math.tan(self.fov / 2) / max(self.height(), 1)
            cp = math.cos(self.pitch)
            fwd = -np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
            right = np.cross(fwd, [0, 1, 0])
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(right, fwd)
            self.target = self.target - right * dx * s + up * dy * s
        else:
            self.yaw -= dx * 0.008
            self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.55, 1.55))
        self.update()

    def wheelEvent(self, e):
        self._zoom.stop()
        self.dist = max(self.dist * 0.88 ** (e.angleDelta().y() / 120), 0.2)
        self.update()

    def leaveEvent(self, e):
        self._mouse = None
        if self.hover != -1:
            self.hover = -1
            self.update()
        self.picked.emit(-1)


# =========================================================================== code editor
class _LineNumbers(QtWidgets.QWidget):
    def __init__(self, ed):
        super().__init__(ed)
        self.ed = ed

    def sizeHint(self):
        return QtCore.QSize(self.ed.gutter_width(), 0)

    def paintEvent(self, e):
        self.ed.paint_gutter(e)


class _Highlighter(QtGui.QSyntaxHighlighter):
    KW = r"\b(def|for|in|if|else|elif|while|return|import|from|as|and|or|not|None|True|False|print|lambda)\b"

    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color, bold=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            return f
        self.rules = [(re.compile(self.KW), fmt("#569cd6", True)),
                      (re.compile(r"\bnp\b"), fmt("#4ec9b0")),
                      (re.compile(r"\b\d+(\.\d+)?\b"), fmt("#b5cea8")),
                      (re.compile(r"'[^']*'|\"[^\"]*\""), fmt("#ce9178")),
                      (re.compile(r"#.*$"), fmt("#6a9955"))]

    def highlightBlock(self, text):
        for rx, f in self.rules:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)


class CodeEdit(QtWidgets.QPlainTextEdit):
    run_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFont(mono_font(14))
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.gutter = _LineNumbers(self)
        self.blockCountChanged.connect(self._margins)
        self.updateRequest.connect(self._scroll_gutter)
        self._margins()
        self._hl = _Highlighter(self.document())
        self.step_line, self.err_line = None, None
        self.setStyleSheet("QPlainTextEdit {background: #1a1b1e; color: #e6e6e6; "
                           "selection-background-color: #264f78; selection-color: #ffffff;}")

    def gutter_width(self):
        return 14 + self.fontMetrics().horizontalAdvance("9") * max(2, len(str(self.blockCount())))

    def _margins(self, *_):
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _scroll_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self.gutter.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, e):
        p = QtGui.QPainter(self.gutter)
        p.fillRect(e.rect(), QtGui.QColor(30, 31, 34))
        block = self.firstVisibleBlock()
        n = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        fh = self.fontMetrics().height()
        while block.isValid() and top <= e.rect().bottom():
            if block.isVisible() and bottom >= e.rect().top():
                ln = n + 1
                p.setPen(QtGui.QColor(255, 107, 107) if ln == self.err_line else
                         QtGui.QColor(235, 235, 235) if ln == self.step_line else QtGui.QColor(95, 100, 110))
                p.drawText(0, top, self.gutter.width() - 6, fh, QtCore.Qt.AlignmentFlag.AlignRight, str(ln))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            n += 1

    def mark(self, step_line=None, err_line=None):
        self.step_line, self.err_line = step_line, err_line
        sels = []
        for ln, col in ((step_line, QtGui.QColor(42, 46, 56)), (err_line, QtGui.QColor(80, 32, 36))):
            if ln is None:
                continue
            b = self.document().findBlockByNumber(ln - 1)
            if not b.isValid():
                continue
            s = QtWidgets.QTextEdit.ExtraSelection()
            s.format.setBackground(col)
            s.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            s.cursor = QtGui.QTextCursor(b)
            sels.append(s)
        self.setExtraSelections(sels)
        self.gutter.update()

    def keyPressEvent(self, e):
        K, M = QtCore.Qt.Key, QtCore.Qt.KeyboardModifier
        if e.key() in (K.Key_Return, K.Key_Enter) and (e.modifiers() & M.ControlModifier):
            self.run_requested.emit()
            return
        if e.key() == K.Key_Tab:
            self.insertPlainText("    ")
            return
        if e.key() in (K.Key_Return, K.Key_Enter):
            line = self.textCursor().block().text()
            indent = line[:len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(e)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(e)


# =========================================================================== background worker
class Worker(QtCore.QObject):
    """One background thread: runs your code and prepares the GPU data, so the window never freezes.
    Only the newest run matters; older queued work is dropped."""
    done = QtCore.pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.busy = False
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, kind, fn):
        self.q.put((kind, fn))

    def _loop(self):
        while True:
            jobs = [self.q.get()]
            while True:
                try:
                    jobs.append(self.q.get_nowait())
                except queue.Empty:
                    break
            runs = [i for i, j in enumerate(jobs) if j[0] == "run"]
            if runs:
                jobs = jobs[runs[-1]:]
            preps = [j for j in jobs if j[0] == "prep"]
            jobs = [j for j in jobs if j[0] != "prep"] + preps[-1:]
            for kind, fn in jobs:
                self.busy = True
                try:
                    res = fn()
                except Exception as e:
                    import traceback
                    res = RuntimeError(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                self.busy = False
                self.done.emit(kind, res)


@dataclass
class Prep:
    gen: int
    i: int
    animate: bool
    GQ: Geom
    GP: Geom = None
    refs: np.ndarray = None
    anim_labels: tuple = None
    static_labels: tuple = None
    ms: float = 0.0


def _texts(R, flat, mode):
    """label text for elements `flat` (flat indices into R; -1 = blank)."""
    flat = np.asarray(flat, np.int64)
    ok = flat >= 0
    out = [""] * len(flat)
    if mode == "values":
        txt = [fmt_val(x) for x in R.flat[flat[ok]].tolist()] if ok.any() else []
    elif R.ndim == 0:
        txt = ["()"] * int(ok.sum())
    else:
        idx = np.unravel_index(flat[ok], R.shape)
        txt = [",".join(map(str, t)) for t in zip(*[a.tolist() for a in idx])]
    for j, t in zip(np.flatnonzero(ok).tolist(), txt):
        out[j] = t
    return out


def _sig(steps, i):
    """identity of a step across re-runs (no line number: adding a line above isn't a change)"""
    s = steps[i]
    nth = sum(1 for t in steps[:i] if t.name == s.name and t.label == s.label)
    return (s.name, s.label, s.raw.shape, s.raw.dtype.str, nth)


# =========================================================================== main window
def _row(*ws):
    w = QtWidgets.QWidget()
    l = QtWidgets.QHBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.setSpacing(4)
    for x in ws:
        if isinstance(x, int):
            l.addStretch(x)
        else:
            l.addWidget(x)
    return w


def _lbl(t, w=None):
    q = QtWidgets.QLabel(t)
    if w:
        q.setFixedWidth(w)
    return q


def _combo(items):
    c = QtWidgets.QComboBox()
    c.addItems(items)
    return c


class Viz(QtWidgets.QMainWindow):
    def __init__(self, code=None):
        super().__init__()
        self.setWindowTitle("npviz")
        self.steps, self.cur, self.gen = [], -1, 0
        self._shown_sig = None
        self._prep = None
        self._geoms = {}
        self._geo_lock = threading.Lock()
        self.runner = Runner()
        self.worker = Worker()
        self.worker.done.connect(self._on_done)
        self._times = {"run": 0.0, "prep": 0.0}
        self._build_ui()
        self.editor.setPlainText(code if code is not None else EXAMPLES["reshape basics"])
        self.run(select_line=1)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.setStyleSheet(f"QWidget {{font-family: '{MONO}'; font-size: 12px;}}"
                           "QLabel#what {color: #7ee08f; font-weight: bold;}"
                           "QLabel#h {color: #6aa6ff; font-weight: bold;}")
        left = QtWidgets.QWidget()
        L = QtWidgets.QVBoxLayout(left)
        L.setContentsMargins(6, 6, 6, 6)
        L.setSpacing(4)

        self.examples = _combo(["examples..."] + list(EXAMPLES))
        self.examples.model().item(0).setEnabled(False)
        self.examples.activated.connect(self._load_example)
        b_run = QtWidgets.QPushButton("run  (Ctrl+Enter)")
        b_run.clicked.connect(lambda: self.run())
        self.live = QtWidgets.QCheckBox("live")
        self.live.setChecked(True)
        self.live.setToolTip("re-run automatically shortly after you stop typing")
        L.addWidget(_row(self.examples, b_run, self.live))

        self.editor = CodeEdit()
        self.editor.run_requested.connect(lambda: self.run())
        self.editor.textChanged.connect(self._on_text)

        steps_box = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(steps_box)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        h = _lbl("steps  (click one to see it animate)")
        h.setObjectName("h")
        sl.addWidget(h)
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.currentRowChanged.connect(self._on_step_clicked)
        sl.addWidget(self.steps_list)

        info_box = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(info_box)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(3)
        self.what = QtWidgets.QLabel()
        self.what.setObjectName("what")
        self.what.setWordWrap(True)
        il.addWidget(self.what)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        il.addWidget(self.info)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("console: print() output, non-array results (a.shape), errors")
        il.addWidget(self.console, 1)

        split = QtWidgets.QSplitter(V_)
        split.addWidget(self.editor)
        split.addWidget(steps_box)
        split.addWidget(info_box)
        split.setSizes([360, 170, 300])
        L.addWidget(split, 1)

        self.cmode = _combo(["origin", "value", "value per array", "flat index"] + [f"axis {k}" for k in range(6)])
        self.cmode.setToolTip("origin: colour = where the element started (follows it through moves)\n"
                              "value: colour by value, one scale for every step (so a * 2 visibly changes)\n"
                              "value per array: colour by value, each array stretched over the full colour range\n"
                              "flat index: position in memory order\naxis k: colour by index along axis k")
        self.cmode.currentIndexChanged.connect(self.refresh)
        self.cmap = _combo(list(LUTS))
        self.cmap.currentTextChanged.connect(lambda n: self.view.set_cmap(n))
        self.labels = _combo(["values", "indices", "no labels"])
        self.labels.currentIndexChanged.connect(self.refresh)
        self.axes_cb = QtWidgets.QCheckBox("axes")
        self.axes_cb.setChecked(True)
        self.axes_cb.toggled.connect(lambda on: (setattr(self.view, "show_axes", on), self.view.update()))
        L.addWidget(_row(_lbl("colour"), self.cmode, self.cmap, self.labels, self.axes_cb))
        self.layout_c = _combo(LAYOUTS)
        self.layout_c.setToolTip("numpy order: like print(a) - last axis = columns, 2nd-last = rows, 3rd-last = pages;\n"
                                 "    a[0] is one block, a[1] the next...\n"
                                 "layers first: like learnbyvisualize - axis 0 = layers in depth, axis 1 = rows,\n"
                                 "    axis 2 = columns, axis 3 = whole cubes side by side (a[..., k] is one cube)")
        self._settings = QtCore.QSettings("npviz", "npviz")
        saved = str(self._settings.value("layout", "numpy order"))
        self.layout_c.setCurrentText(saved if saved in LAYOUTS else "numpy order")
        self.layout_c.currentTextChanged.connect(lambda t: self._settings.setValue("layout", t))
        self.layout_c.currentIndexChanged.connect(lambda _: self.refresh(fit=True))
        L.addWidget(_row(_lbl("layout"), self.layout_c, 1))

        def slider(name, lo, hi, val, cb, fmt=None):
            s = QtWidgets.QSlider(H_)
            s.setRange(lo, hi)
            s.setValue(val)
            vl = _lbl(fmt(val) if fmt else "", 36)
            s.valueChanged.connect(lambda v: (cb(v), vl.setText(fmt(v) if fmt else "")))
            return [_lbl(name), s] + ([vl] if fmt else [])
        self.size_s = slider("sheet", 20, 100, 86, lambda v: self._set_view("size", v / 100))
        self.thick_s = slider("thick", 1, 100, 8, lambda v: self._set_view("thick", v / 100))
        self.page_s = slider("page gap", 10, 40, 14, lambda v: self.refresh())
        L.addWidget(_row(*self.size_s, *self.thick_s, *self.page_s))

        h2 = _lbl("animation")
        h2.setObjectName("h")
        L.addWidget(h2)
        self.speed_s = slider("move time", 2, 80, 20, lambda v: None, lambda v: f"{v / 10:.1f}s")
        self.speed_s[0].setToolTip("how long the elements take to move")
        self.hold_s = slider("hold before", 0, 50, 12, lambda v: None, lambda v: f"{v / 10:.1f}s")
        self.hold_s[0].setToolTip("how long to show the starting array (with labels) before anything moves")
        L.addWidget(_row(*self.speed_s, *self.hold_s))
        self.move_c = _combo(["in a wave", "all together", "one by one"])
        self.move_c.setToolTip("in a wave: elements start one after another, overlapping\n"
                               "all together: everything moves at once\n"
                               "one by one: each element moves on its own, in the new array's order (slower)")
        self.loop_cb = QtWidgets.QCheckBox("loop")
        self.loop_cb.setToolTip("keep replaying, pausing at the start and the end")
        self.loop_cb.toggled.connect(lambda on: self.replay() if on else self._loop_timer.stop())
        b_replay = QtWidgets.QPushButton("replay")
        b_replay.clicked.connect(self.replay)
        L.addWidget(_row(_lbl("move"), self.move_c, self.loop_cb, b_replay, 1))
        b_rot, b_front, b_fit = (QtWidgets.QPushButton(t) for t in ("reset rotation", "front", "fit view"))
        b_rot.setToolTip("back to the default angle (keeps zoom and pan)  [r]")
        b_front.setToolTip("look at the array straight on")
        b_fit.setToolTip("re-centre, default angle, zoom to fit  [f / double-click]")
        b_rot.clicked.connect(lambda: self.view.reset_rotation())
        b_front.clicked.connect(lambda: self.view.reset_rotation(0.0, 0.0))
        b_fit.clicked.connect(lambda: self.fit(reset_angles=True))
        self.lock_cb = QtWidgets.QCheckBox("lock rotation")
        self.lock_cb.setToolTip("left-drag pans instead of rotating")
        self.lock_cb.toggled.connect(lambda on: setattr(self.view, "rot_locked", on))
        L.addWidget(_row(_lbl("view"), b_rot, b_front, b_fit, self.lock_cb, 1))
        self.scrub = QtWidgets.QSlider(H_)
        self.scrub.setRange(0, 1000)
        self.scrub.setToolTip("drag to move the animation by hand")
        self.scrub.sliderMoved.connect(self._on_scrub)
        self.scrub.sliderReleased.connect(lambda: self._on_scrub(self.scrub.value(), released=True))
        L.addWidget(_row(_lbl("scrub"), self.scrub))
        self.stats = _lbl("")
        self.stats.setStyleSheet("color: #8a8f98;")
        L.addWidget(self.stats)

        self.view = View()
        self.view.picked.connect(self._on_pick)
        self.view.anim_done.connect(self._on_anim_done)
        self.view.progress.connect(self._on_progress)
        self.view.key.connect(self._view_key)
        self.view.dbl.connect(lambda: self.fit(reset_angles=True))
        self.hover = QtWidgets.QLabel(" ")
        self.hover.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hover.setStyleSheet("padding: 3px;")
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self.view, 1)
        rl.addWidget(self.hover)

        main = QtWidgets.QSplitter(H_)
        main.addWidget(left)
        main.addWidget(right)
        main.setSizes([500, 1000])
        self.setCentralWidget(main)

        self._run_timer = QtCore.QTimer(self)
        self._run_timer.setSingleShot(True)
        self._run_timer.timeout.connect(lambda: self.run())
        self._loop_timer = QtCore.QTimer(self)
        self._loop_timer.setSingleShot(True)
        self._loop_timer.timeout.connect(self.replay)
        self._stat_timer = QtCore.QTimer(self)
        self._stat_timer.timeout.connect(self._update_stats)
        self._stat_timer.start(300)

    def _set_view(self, k, v):
        setattr(self.view, k, v)
        self.view.update()

    def _update_stats(self):
        v = self.view
        G = self._prep.GQ if self._prep is not None else None
        sub = f" (every {G.sub}th per axis)" if G is not None and G.sub > 1 else ""
        busy = "  working..." if self.worker.busy else ""
        self.stats.setText(f"{v.count:,} {'sheets' if v.cubes else 'points'}{sub} | run {self._times['run']:.0f}ms "
                           f"| prepare {self._times['prep']:.0f}ms | upload {v.upload_ms:.0f}ms{busy}")

    def _view_key(self, k):
        K = QtCore.Qt.Key
        if k == K.Key_Space:
            self.replay()
        elif k == K.Key_F:
            self.fit(reset_angles=True)
        elif k == K.Key_R:
            self.view.reset_rotation()
        elif k in (K.Key_Up, K.Key_Left):
            self.select(self.cur - 1)
        elif k in (K.Key_Down, K.Key_Right):
            self.select(self.cur + 1)

    def _load_example(self, i):
        if i > 0:
            self.editor.setPlainText(EXAMPLES[self.examples.itemText(i)])
            self.run(select_line=1)

    # ------------------------------------------------------------------ running (background)
    def _on_text(self):
        if self.live.isChecked():
            self._run_timer.start(500)

    def run(self, select_line=None):
        self._run_timer.stop()
        code = self.editor.toPlainText()
        runner = self.runner

        def job():
            t0 = time.perf_counter()
            steps, console, err, reused = runner.run(code)
            for i, st in enumerate(steps or []):        # small steps: analyse now so the list shows kinds
                if st.raw.size <= 200_000:
                    analyse(steps, i)
            return code, select_line, steps, console, err, reused, (time.perf_counter() - t0) * 1e3
        self.worker.submit("run", job)

    def _on_done(self, kind, res):
        if isinstance(res, Exception):
            self.console.appendPlainText(f"[internal error] {res}")
            return
        if kind == "run":
            self._apply_run(*res)
        else:
            self._apply_prep(res)

    def _apply_run(self, code, select_line, steps, console, err, reused, ms):
        self._times["run"] = ms
        footer = [f"[ran in {ms:.0f} ms, {len(steps or [])} steps, {reused} unchanged lines re-used]"]
        if steps is None or (err is not None and self.steps and select_line is None):
            self.console.setPlainText("\n".join(console + footer + ["(display unchanged - showing the last run that worked)"]))
            self.editor.mark(self.editor.step_line, err)
            return
        old_sigs = {_sig(self.steps, i) for i in range(len(self.steps))}
        new_sigs = [_sig(steps, i) for i in range(len(steps))]
        shown = self._shown_sig
        if select_line is None and self.steps and shown is not None and shown not in new_sigs \
                and all(sg in old_sigs for sg in new_sigs):
            self.console.setPlainText("\n".join(console + footer + ["(display unchanged while you edit)"]))
            return
        self.console.setPlainText("\n".join(console + footer))
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
        self.steps = steps
        self.gen += 1
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        for i in range(len(steps)):
            self.steps_list.addItem(self._step_text(i))
        self.steps_list.blockSignals(False)
        self.editor.mark(None, err)
        if not steps:
            self.cur, self._shown_sig, self._prep = -1, None, None
            z = np.zeros((TEX_W, 2), np.float32)
            self.view.set_static(Geom(z[:0], None, (0,), 1, make_layout((0,)), z))
            self.view.axes_info, self.view.label_src = [], None
            self.view.overlay.title = ""
            self.what.setText("")
            self.info.setText("")
            return
        if select_line is not None:
            i = next((j for j, s in enumerate(steps) if s.line == select_line), len(steps) - 1)
            self.select(i, animate=True, move_cursor=False, fit=True)
            return
        new = [i for i in range(len(steps)) if new_sigs[i] not in old_sigs]
        if new:
            line = self.editor.textCursor().blockNumber() + 1
            on_line = [i for i in new if steps[i].line <= line <= steps[i].line + steps[i].label.count("\n")]
            self.select(on_line[-1] if on_line else new[-1], animate=True, move_cursor=False)
        else:
            same = [i for i in range(len(steps)) if new_sigs[i] == shown]
            self.select(same[-1] if same else len(steps) - 1, animate=False, move_cursor=False)

    def _step_text(self, i):
        s = self.steps[i]
        return f"L{s.line:<3} {s.name:<8} {str(s.raw.shape):<16} {s.kind if s.analysed else '...'}"

    def _on_step_clicked(self, i):
        if 0 <= i < len(self.steps):
            self.select(i, move_cursor=True)

    # ------------------------------------------------------------------ preparing a step (background)
    def _geom(self, steps, i, cmode, page, order, vrange=None):
        key = (id(steps[i]), cmode, page, order, vrange)
        with self._geo_lock:
            g = self._geoms.get(key)
        if g is None:
            g = build_geom(steps, i, cmode, 1.5, page / 10, True, MAX_SHOW, order, vrange)
            with self._geo_lock:
                if len(self._geoms) > 6:
                    self._geoms.pop(next(iter(self._geoms)))
                self._geoms[key] = g
        return g

    def select(self, i, animate=True, move_cursor=True, fit=False):
        if not (0 <= i < len(self.steps)):
            return
        self._loop_timer.stop()
        self.cur = i
        st = self.steps[i]
        self.steps_list.blockSignals(True)
        self.steps_list.setCurrentRow(i)
        self.steps_list.blockSignals(False)
        self.editor.mark(st.line, self.editor.err_line)
        if move_cursor:
            c = QtGui.QTextCursor(self.editor.document().findBlockByNumber(st.line - 1))
            self.editor.setTextCursor(c)
        R = st.raw
        self.what.setText(st.desc if st.analysed else "working out how the elements moved...")
        self.info.setText(f"{st.name}:  shape {R.shape}   ndim {R.ndim}   size {R.size:,}   dtype {R.dtype}\n"
                          f"strides {R.strides} bytes   (C-contiguous {R.flags.c_contiguous})\n{st.mem}")
        self.hover.setText(" ")
        self._shown_sig = _sig(self.steps, i)
        self._fit_next = fit
        steps, gen = self.steps, self.gen
        cmode, page, lmode = self.cmode.currentText(), self.page_s[1].value(), self.labels.currentText()
        order = self.layout_c.currentText()
        if self.view.static and self.view.count:
            self.view.overlay.phase = "preparing..."
            self.view.update()

        def job():
            t0 = time.perf_counter()
            analyse(steps, i)
            st = steps[i]
            vr = shared_range(steps) if cmode == "value" else None
            GQ = self._geom(steps, i, cmode, page, order, vr)
            res = Prep(gen, i, animate, GQ)
            small = lmode != "no labels" and GQ.sub == 1
            if small and len(GQ.sv) <= MAX_LABELS:
                res.static_labels = (positions(GQ.L, np.arange(len(GQ.sv))), _texts(st.raw, np.arange(st.raw.size), lmode))
            if st.parent >= 0:
                P = steps[st.parent]
                GP = self._geom(steps, st.parent, cmode, page, order, vr)
                refs, ref, dref = build_transition(P, st, GP, GQ)
                res.GP, res.refs = GP, refs
                if small and GP.sub == 1 and len(refs) <= MAX_LABELS:
                    res.anim_labels = self._anim_labels(GP, GQ, ref, dref, _texts(P.raw, ref, lmode),
                                                        _texts(st.raw, dref, lmode))
            else:
                n = len(GQ.sv)
                ref, dref = np.full(n, -1, np.int32), np.arange(n, dtype=np.int32)
                res.refs = _refs(ref, dref)
                if res.static_labels is not None:
                    t = res.static_labels[1]
                    res.anim_labels = self._anim_labels(GQ, GQ, ref, dref, t, t)
            res.ms = (time.perf_counter() - t0) * 1e3
            return res
        self.worker.submit("prep", job)

    @staticmethod
    def _anim_labels(GP, GQ, ref, dref, t0, t1):
        p0 = positions(GP.L, np.maximum(ref, 0))
        p1 = positions(GQ.L, np.maximum(dref, 0))
        a0, a1 = ref >= 0, dref >= 0
        p0[~a0] = p1[~a0]
        p1[~a1] = p0[~a1]
        return p0, p1, a0.astype(float), a1.astype(float), t0, t1

    def _apply_prep(self, res):
        if res.gen != self.gen or res.i != self.cur:
            return                                   # stale (a newer run or another step was picked)
        self._times["prep"] = res.ms
        self._prep = res
        st = self.steps[res.i]
        self.what.setText(st.desc)
        item = self.steps_list.item(res.i)
        if item is not None:
            item.setText(self._step_text(res.i))
        if getattr(self, "_fit_next", False):
            self.view.fit(res.GQ.half, reset_angles=True)
            self._fit_next = False
        if res.animate:
            self._play_from(res)
        else:
            self._show_static_from(res)

    def _play_from(self, res, start=True):
        st = self.steps[res.i]
        self._loop_timer.stop()
        if res.GP is not None:
            P = self.steps[st.parent]
            hold = self.hold_s[1].value() / 10
            self.view.overlay.title = f"{st.cand[0][0]} {P.raw.shape}   ->   {st.name} {st.raw.shape}"
            self._ensure_visible(np.maximum(res.GP.half, res.GQ.half))
            GA, axes = res.GP, (res.GP.axes, res.GQ.axes)
        else:
            hold, GA, axes = 0.0, res.GQ, None
            self.view.overlay.title = f"{st.name}   shape {st.raw.shape}"
            self._ensure_visible(res.GQ.half)
        n = len(res.refs)
        mode = self.move_c.currentText()
        dur = self.speed_s[1].value() / 10
        if mode == "all together":
            self.view.stagger = 0.0
        elif mode == "in a wave":
            self.view.stagger = 0.35
        else:
            dur *= float(np.clip(n / 8, 1, 6))
            self.view.stagger = float(np.clip(1 - 1.5 / max(n, 1), 0, 0.985))
        self.view.duration = dur
        self.view.set_transition(GA, res.GQ, res.refs, hold=hold, labels=res.anim_labels, start=start, axes=axes)
        self.scrub.setValue(0)

    def _show_static_from(self, res):
        st = self.steps[res.i]
        self.view.set_static(res.GQ)
        self.view.axes_info = res.GQ.axes
        self.view.label_src = res.static_labels
        self.view.overlay.title = f"{st.name}   shape {st.raw.shape}"
        self._ensure_visible(res.GQ.half)
        self.scrub.blockSignals(True)
        self.scrub.setValue(1000)
        self.scrub.blockSignals(False)

    def replay(self):
        if self._prep is not None and self._prep.i == self.cur and self._prep.gen == self.gen:
            self._play_from(self._prep)
        elif self.cur >= 0:
            self.select(self.cur, animate=True, move_cursor=False)

    def _on_progress(self, t):
        if not self.scrub.isSliderDown():
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(t * 1000))
            self.scrub.blockSignals(False)

    def _on_scrub(self, v, released=False):
        if self._prep is None or self._prep.i != self.cur:
            return
        if self.view.static:
            self._play_from(self._prep, start=False)
            self.scrub.setValue(v)
        self._loop_timer.stop()
        t = v / 1000
        if released and t >= 1.0:
            self._show_static_from(self._prep)
        else:
            self.view.scrub(t)

    def _on_anim_done(self):
        if self._prep is not None:
            self._show_static_from(self._prep)
        if self.loop_cb.isChecked():
            self._loop_timer.start(int(max(self.hold_s[1].value() / 10, 0.6) * 1000))

    def refresh(self, *_, fit=False):
        if self.cur >= 0:
            self.select(self.cur, animate=False, move_cursor=False, fit=fit)

    def _ensure_visible(self, half):
        """Never moves or rotates the camera; only zooms out (smoothly) if the array wouldn't fit."""
        need = self.view.fit_dist(half)
        if need > self.view.dist * 1.1:
            self.view.zoom_to(need * 1.05)

    def fit(self, reset_angles=False):
        if self._prep is not None:
            self.view.fit(self._prep.GQ.half, reset_angles)

    # ------------------------------------------------------------------ hover
    def hover_text(self, gi):
        if self._prep is None or self._prep.i != self.cur:
            return " "
        st, G = self.steps[self.cur], self._prep.GQ
        if not (0 <= gi < len(G.sv)):
            return " "
        R = st.raw
        fi = int(gi if G.ids is None else G.ids[gi])
        idx = np.unravel_index(fi, R.shape) if R.ndim else ()
        txt = f"{st.name}[{', '.join(map(str, idx))}] = {fmt_val(R[idx])}"
        if st.parent < 0 or not st.analysed:
            return txt + f"      (flat position {fi})"
        P = self.steps[st.parent].raw
        pn = st.cand[0][0]
        if st.kind == "move":
            pf = int(st.prov[fi])
            if pf < 0:
                return txt + "      <- new"
            off = 0
            for nm, si in st.srcs:
                S = self.steps[si].raw
                if pf < off + S.size:
                    sidx = np.unravel_index(pf - off, S.shape) if S.ndim else ()
                    return txt + f"      <- {nm}[{', '.join(map(str, sidx))}]   (flat {fi} <- flat {pf - off})"
                off += S.size
        if st.kind == "reduce":
            ax, kd, name = st.red
            it = iter(idx)
            parts = []
            for k in range(P.ndim):
                if k in ax:
                    parts.append(":")
                    if kd:
                        next(it)
                else:
                    parts.append(str(next(it)))
            return txt + f"      = {name or 'f'}({pn}[{', '.join(parts)}])"
        if st.kind == "elementwise":
            return txt + f"      was {fmt_val(P[idx])}"
        return txt

    def _on_pick(self, gi):
        self.hover.setText(" " if gi < 0 or self.cur < 0 else self.hover_text(gi))


def fmt_val(x):
    if isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (complex, np.complexfloating)):
        return f"{x:.3g}"
    x = float(x)
    if x != x:
        return "nan"
    if math.isfinite(x) and x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:.3g}"


def _dark_theme(app):
    app.setStyle("Fusion")
    R, G, C = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup, QtGui.QColor
    p = QtGui.QPalette()
    for role, col in ((R.Window, "#202124"), (R.WindowText, "#d8d8d8"), (R.Base, "#17181b"),
                      (R.AlternateBase, "#222327"), (R.Text, "#d8d8d8"), (R.Button, "#2b2d31"),
                      (R.ButtonText, "#d8d8d8"), (R.Highlight, "#2f5f9f"), (R.HighlightedText, "#ffffff"),
                      (R.ToolTipBase, "#2b2d31"), (R.ToolTipText, "#d8d8d8"), (R.PlaceholderText, "#7a7a7a"),
                      (R.BrightText, "#ff6b6b"), (R.Light, "#3a3c42"), (R.Mid, "#2a2c30"), (R.Dark, "#151618")):
        p.setColor(role, C(col))
    for role in (R.Text, R.WindowText, R.ButtonText):
        p.setColor(G.Disabled, role, C("#6a6a6a"))
    app.setPalette(p)


def _app():
    QtGui.QSurfaceFormat.setDefaultFormat(_surface_format())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    _dark_theme(app)
    return app


def main():
    app = _app()
    code = None
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            code = fh.read()
    w = Viz(code)
    w.resize(1500, 950)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
