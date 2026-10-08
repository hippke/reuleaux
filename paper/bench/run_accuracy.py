"""Accuracy of every method against the 50-digit reference, with the
componentwise condition number of every configuration -> accuracy.npz"""
import time
import numpy as np
from common import *

U = 2.0 ** -53
rng = np.random.default_rng(2026)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
fams = {
    "general": fam_general(rng, 4 * N),
    "similar": fam_similar(rng, N),
    "hierarchical": fam_hierarchical(rng, N),
    "corner": fam_corner(rng, N, -7.0, -0.3)[0],
    "sliver": fam_sliver_lens(rng, N, -7.0, -0.3),
}
METHODS = ("stable", "generic", "vector", "fewell", "fewell_eq16")
res = {}
t0 = time.time()
for name, X in fams.items():
    Y = shift(X)
    out = {m: run(m, Y if m != "generic" else X) for m in METHODS}
    out["lens_textbook"] = np.array([textbook_lens_for(row) for row in X])
    ref = np.full(len(X), np.nan)
    kap = np.full(len(X), np.nan)
    case = []
    reflex = np.zeros(len(X), bool)
    for k, row in enumerate(X):
        cs = [tuple(row[3 * c:3 * c + 3]) for c in range(3)]
        ov = tc.overlap(*cs)
        case.append(ov.case + ("*" if ov.perturbed else ""))
        reflex[k] = ov.reflex
        if ov.area > 0 and not ov.perturbed:
            A, K = mp_area(row, True)
            if A is not None and K is not None:
                ref[k] = float(A)
                kap[k] = float(K)
    case = np.array(case)
    rmin = np.min(X[:, [2, 5, 8]], axis=1)
    res[name] = dict(X=X, ref=ref, kappa=kap, case=case, reflex=reflex, rmin=rmin, **out)
    pos = np.isfinite(ref) & (ref > 0)
    print(f"{name:13s} n={len(X)}, positive {pos.sum()}, cases "
          + ", ".join(f"{c} {n}" for c, n in zip(*np.unique(case, return_counts=True)))
          + f"  [{time.time() - t0:.0f} s]")
    for m in METHODS + ("lens_textbook",):
        e = np.abs(out[m] - ref) / ref
        ok = pos & np.isfinite(e)
        if not ok.any():
            continue
        nrm = e[ok] / (kap[ok] * U)
        print(f"    {m:14s} n={ok.sum():5d}  max rel {e[ok].max():.1e}  "
              f"max err/(kappa u) {nrm.max():.2e}  median {np.median(nrm):.2e}")
np.savez_compressed(os.path.join(HERE, "accuracy.npz"),
                    **{f"{f}__{k}": v for f, d in res.items() for k, v in d.items()})
