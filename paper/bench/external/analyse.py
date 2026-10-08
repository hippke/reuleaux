"""Compare external codes with this work on the shared configurations."""
import json, os, sys, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import batch_stable_shift, ks          # noqa: E402

U = 2.0 ** -53
d = np.load(os.path.join(HERE, "configs.npz"), allow_pickle=True)
X, ref, kap, fam = d["X"], d["ref"], d["kappa"], d["fam"]
rmin = X[:, [2, 5, 8]].min(axis=1)
unit = math.pi * rmin ** 2
ours = np.empty(len(X)); batch_stable_shift(X, ours)

gef_blocked = np.load(os.path.join(HERE, "out_gefera.npz"))["blocked"]
gef = d["L01"] + d["L02"] - gef_blocked
tools = {
    "this work": ours,
    "eulerr 7.1 (R/Rust)": np.loadtxt(os.path.join(HERE, "out_eulerr.txt")),
    "venn.js (JS)": np.array(json.load(open(os.path.join(HERE, "out_venn.json")))),
    "matplotlib-venn 1.1.2": np.load(os.path.join(HERE, "out_mvenn.npy")),
    "gefera (via flux)": gef,
    "Shapely q=8": np.load(os.path.join(HERE, "out_shapely_q8.npy")),
    "Shapely q=64": np.load(os.path.join(HERE, "out_shapely_q64.npy")),
    "Shapely q=512": np.load(os.path.join(HERE, "out_shapely_q512.npy")),
}
pos = ref > 0
zero = ref == 0
print(f"configurations {len(X)}: {pos.sum()} with A > 0, {zero.sum()} empty\n")
hdr = f"{'tool':24s} {'NaN/exc':>8s} {'gross':>7s} {'max rel (A>0)':>14s} {'max err/ku':>11s} {'med err/ku':>11s} {'p99 err/ku':>11s}"
print(hdr)
rows = {}
for name, a in tools.items():
    bad = ~np.isfinite(a)
    err_abs = np.abs(a - ref)
    # gross failure: relative error above 1e-3 where A > 0, or a non-zero
    # result (above 1e-9 of the smallest disk) where the overlap is empty
    wrong = np.isfinite(a) & ((pos & (err_abs > 1e-3 * ref)) | (zero & (err_abs > 1e-9 * unit)))
    ok = pos & np.isfinite(a)
    rel = err_abs[ok] / ref[ok]
    nk = rel / (kap[ok] * U)
    rows[name] = dict(nan=int(bad.sum()), wrong=int(wrong.sum()), maxrel=float(np.nanmax(rel)),
                      over100=int((nk > 100).sum()),
                      maxnk=float(nk.max()), mednk=float(np.median(nk)), p99=float(np.percentile(nk, 99)))
    r = rows[name]
    print(f"{name:24s} {r['nan']:8d} {r['wrong']:7d} {r['maxrel']:14.2e} {r['maxnk']:11.2e} {r['mednk']:11.2e} {r['p99']:11.2e}  >100ku: {r['over100']}")
    # per-family wrong counts
    wf = {f: int((wrong & (fam == f)).sum() + (bad & (fam == f)).sum()) for f in np.unique(fam)}
    rows[name]["wrong_by_family"] = wf
print()
for name in tools:
    print(f"{name:24s} failures (NaN + wrong) by family:", rows[name]["wrong_by_family"])

# gefera's native quantity: area blocked = |S ∩ (P ∪ M)|
blk_exact = d["L01"] + d["L02"] - ref
our_blk = np.array([ks._lens_area(r[2], r[5], math.hypot(r[0]-r[3], r[1]-r[4])) +
                    ks._lens_area(r[2], r[8], math.hypot(r[0]-r[6], r[1]-r[7])) for r in X]) - ours
m = blk_exact > 0
for name, b in (("gefera", gef_blocked), ("this work", our_blk)):
    e = np.abs(b - blk_exact)[m] / blk_exact[m]
    print(f"union |S∩(P∪M)|, {name:10s}: NaN {int((~np.isfinite(b[m])).sum())}, max rel {np.nanmax(e):.2e}, "
          f"median {np.nanmedian(e):.2e}, >1e-6: {int((e > 1e-6).sum())}")

# our speed on the same sets
sp = {}
for nm in ("configs", "traj_jupiter_earth", "traj_large_moon"):
    Y = X if nm == "configs" else np.load(os.path.join(HERE, nm + ".npz"))["X"]
    Y = np.ascontiguousarray(Y); o = np.empty(len(Y)); batch_stable_shift(Y, o)
    best = min((lambda t0: (batch_stable_shift(Y, o), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(7))
    sp[nm] = 1e9 * best / len(Y)
json.dump({"accuracy": rows, "speed_ours": sp}, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print("\nthis work ns per evaluation:", {k: round(v, 1) for k, v in sp.items()})
