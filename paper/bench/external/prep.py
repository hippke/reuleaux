"""Exact lenses (50 digits) of the pairs (0,1) and (0,2); empties get ref = 0."""
import sys, os
import numpy as np
import mpmath as mp
mp.mp.dps = 50
d = dict(np.load("configs.npz", allow_pickle=True))
X = d["X"]

def lens(c1, c2):
    (x1, y1, r1), (x2, y2, r2) = [[mp.mpf(float(v)) for v in c] for c in (c1, c2)]
    dd = mp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if dd >= r1 + r2:
        return mp.mpf(0)
    if dd <= abs(r1 - r2):
        return mp.pi * min(r1, r2) ** 2
    return (r1 ** 2 * mp.acos((dd * dd + r1 * r1 - r2 * r2) / (2 * dd * r1))
            + r2 ** 2 * mp.acos((dd * dd + r2 * r2 - r1 * r1) / (2 * dd * r2))
            - mp.sqrt((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2)) / 2)

L01 = np.array([float(lens(r[0:3], r[3:6])) for r in X])
L02 = np.array([float(lens(r[0:3], r[6:9])) for r in X])
ref = d["ref"].copy()
ref[d["case"] == "empty"] = 0.0
d.update(L01=L01, L02=L02, ref=ref)
np.savez("configs.npz", **d)
np.savetxt("configs.csv", X, delimiter=",", fmt="%.17g")
print("ok", np.isfinite(ref).sum())
