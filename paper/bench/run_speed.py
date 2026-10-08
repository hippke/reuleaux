"""Cost per evaluation by topological case -> speed.json (run pinned: taskset -c 2)"""
import json, platform, time
import numpy as np
from common import *

rng = np.random.default_rng(7)
pool = np.vstack([fam_general(rng, 60000), fam_similar(rng, 30000), fam_hierarchical(rng, 30000)])
case = np.empty(len(pool), dtype=np.int64); nv = np.empty(len(pool), dtype=np.int64)
batch_case(shift(pool), case, nv)
classes = {"empty": case == 0, "disk": case == 1, "lens": case == 2,
           "triangle": (case == 3) & (nv == 3), "quadrilateral": (case == 3) & (nv == 4)}
M = 20000
sets = {}
for k, m in classes.items():
    rows = pool[m]
    reps = int(np.ceil(M / len(rows)))
    sets[k] = np.ascontiguousarray(np.tile(rows, (reps, 1))[:M])
    print(k, len(rows), "distinct")
mix = np.ascontiguousarray(pool[:M])

def best(fn, X, rep=9):
    out = np.empty(X.shape[0]); fn(X, out)          # compile / warm
    t = min(_t(fn, X, out) for _ in range(rep))
    return 1e9 * t / X.shape[0], out

def _t(fn, X, out):
    t0 = time.perf_counter(); fn(X, out); return time.perf_counter() - t0

methods = {
    "stable": batch_stable_shift,
    "vector": batch_vector_shift,
    "generic": batch_generic,
    "fewell": lambda X, out: fewell_batch(X, True, out),
}
res = {"cpu": platform.processor() or "x86_64", "numba": __import__("numba").__version__, "ns": {}}
for cname, X in list(sets.items()) + [("mixture", mix)]:
    res["ns"][cname] = {}
    ref = np.empty(X.shape[0]); batch_stable_shift(X, ref)
    for mname, fn in methods.items():
        if mname == "fewell" and cname != "triangle":
            continue
        ns, out = best(fn, X, rep=9)
        res["ns"][cname][mname] = ns
    print(cname, {k: round(v, 1) for k, v in res["ns"][cname].items()})
json.dump(res, open(os.path.join(HERE, "speed.json"), "w"), indent=1)
