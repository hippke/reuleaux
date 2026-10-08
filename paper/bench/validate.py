import numpy as np, math
from common import *
rng = np.random.default_rng(11)
for name, X in [("general", fam_general(rng, 20000)), ("similar", fam_similar(rng, 20000)),
                ("hier", fam_hierarchical(rng, 20000)), ("corner", fam_corner(rng, 5000)[0]),
                ("sliver", fam_sliver_lens(rng, 5000))]:
    Y = shift(X)
    a_s = run("stable", Y); a_g = run("generic", X); a_v = run("vector", Y)
    rmin = np.min(X[:, [2, 5, 8]], axis=1); u = np.pi * rmin**2
    ok = np.isfinite(a_s) & np.isfinite(a_g)
    print(f"{name:8s} nan stable {np.sum(~np.isfinite(a_s))} generic {np.sum(~np.isfinite(a_g))}  "
          f"|stable-generic|/disk max {np.max(np.abs(a_s-a_g)[ok]/u[ok]):.1e}  rel {np.max((np.abs(a_s-a_g)/np.maximum(a_g,1e-300))[ok & (a_g>0)]):.1e}"
          f"  |vector-generic|/disk {np.nanmax(np.abs(a_v-a_g)/u):.1e}")
