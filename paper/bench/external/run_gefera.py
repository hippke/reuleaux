"""gefera (Gordon & Agol 2022), Fortran photlib built from the public source
(github.com/tagordon/gefera, commit e27e832) with its own flags (-Ofast).
flux_ng returns the relative flux change F - 1 (= -0.01 for a body of
radius 0.1 on the star). Uniform source (u1 = u2 = 0): the blocked area is
|S ∩ (P ∪ M)| = -pi (F - 1), so A(S∩P∩M) = A(S∩P) + A(S∩M) + pi (F - 1). Run with the env that has the lib."""
import ctypes, time, math, json, os
import numpy as np
from ctypes import byref, c_double, c_int

import sys
LIB = sys.argv[1] if len(sys.argv) > 1 else "photlib.so"
lib = ctypes.CDLL(os.path.abspath(LIB))
lib.flux_ng.restype = None

def flux_ng(c1, c2, rp, rm, bp, bpm, cth, sth):
    j = len(bp)
    arrs = [np.ascontiguousarray(a, dtype=float) for a in (bp, bpm, cth, sth)]
    ptrs = [byref((c_double * j).from_buffer(a)) for a in arrs]
    out = np.zeros(j)
    lib.flux_ng(byref(c_double(c1)), byref(c_double(c2)), byref(c_double(rp)), byref(c_double(rm)),
                *ptrs, byref((c_double * j).from_buffer(out)), byref(c_int(j)))
    return out

def geometry(row):
    x0, y0, r0 = row[0:3]
    px, py, rp = (row[3] - x0) / r0, (row[4] - y0) / r0, row[5] / r0
    mx, my, rm = (row[6] - x0) / r0, (row[7] - y0) / r0, row[8] / r0
    bp = math.hypot(px, py)
    dx, dy = mx - px, my - py
    bpm = math.hypot(dx, dy)
    if bp == 0:
        return None                      # planet centred on the star: theta undefined
    if bpm == 0:                         # concentric planet and moon: any theta
        return rp, rm, bp, 0.0, 1.0, 0.0, r0
    cth = (-px * dx - py * dy) / (bp * bpm)
    # gefera's own impacts() returns theta in [0, pi] (Kahan's angle from the
    # three distances), i.e. sin(theta) >= 0
    sth = abs(-px * dy + py * dx) / (bp * bpm)
    return rp, rm, bp, bpm, cth, sth, r0

d = np.load("configs.npz", allow_pickle=True)
X = d["X"]
blocked = np.full(len(X), np.nan)
t0 = time.time()
for n, row in enumerate(X):
    g = geometry(row)
    if g is None:
        continue
    rp, rm, bp, bpm, cth, sth, r0 = g
    F = flux_ng(0.0, 0.0, rp, rm, np.array([bp]), np.array([bpm]), np.array([cth]), np.array([sth]))[0]
    blocked[n] = -r0 * r0 * math.pi * F
print("accuracy pass", time.time() - t0, "s")

# speed: light-curve style batches (fixed radii, many positions), like the paper mixture
rng = np.random.default_rng(5)
res = {}
for label, (rp, rm) in {"jupiter+earth": (0.1, 0.0092), "large moon": (0.12, 0.05)}.items():
    n = 200000
    ap = rng.uniform(0, 2 * np.pi, n); dp = rng.uniform(0.75, 1.05, n)
    px, py = dp * np.cos(ap), dp * np.sin(ap)
    am = rng.uniform(0, 2 * np.pi, n); dm = rng.uniform(0.0, rp + rm, n)
    mx, my = px + dm * np.cos(am), py + dm * np.sin(am)
    bp = np.hypot(px, py); dx, dy = mx - px, my - py; bpm = np.hypot(dx, dy)
    cth = (-px * dx - py * dy) / (bp * bpm); sth = np.abs(-px * dy + py * dx) / (bp * bpm)
    flux_ng(0.0, 0.0, rp, rm, bp[:100], bpm[:100], cth[:100], sth[:100])
    best = 1e9
    for _ in range(5):
        t = time.perf_counter(); F = flux_ng(0.0, 0.0, rp, rm, bp, bpm, cth, sth); best = min(best, time.perf_counter() - t)
    res[label] = 1e9 * best / n
    np.savez(f"traj_{label.replace(' ', '_').replace('+', '_')}.npz",
             X=np.column_stack([np.zeros(n), np.zeros(n), np.ones(n), px, py, np.full(n, rp), mx, my, np.full(n, rm)]),
             gefera_F=F)
print("gefera ns per evaluation (uniform LD, one call per batch):", res)
np.savez("out_gefera.npz" if LIB == "photlib.so" else "out_gefera_O2.npz", blocked=blocked)
json.dump(res, open("speed_gefera.json" if LIB == "photlib.so" else "speed_gefera_O2.json", "w"), indent=1)
