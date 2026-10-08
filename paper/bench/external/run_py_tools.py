"""matplotlib-venn 1.1.2 (exact arcgon region algebra) and Shapely/GEOS
(polygonal circles) on the shared configurations."""
import time, json, warnings
import numpy as np
import shapely
from matplotlib_venn._region import VennCircleRegion

warnings.filterwarnings("ignore")
d = np.load("configs.npz", allow_pickle=True)
X = d["X"]

def mvenn(row):
    r = VennCircleRegion(np.array(row[0:2], float), float(row[2]))
    _, i1 = r.subtract_and_intersect_circle(np.array(row[3:5], float), float(row[5]))
    _, i2 = i1.subtract_and_intersect_circle(np.array(row[6:8], float), float(row[8]))
    return i2.size()

out = np.full(len(X), np.nan)
fails = 0
for n, row in enumerate(X):
    try:
        out[n] = mvenn(row)
    except Exception:
        fails += 1
print("matplotlib-venn exceptions:", fails)
np.save("out_mvenn.npy", out)

def shp(Xs, q):
    c = [shapely.buffer(shapely.points(Xs[:, 3 * k], Xs[:, 3 * k + 1]), Xs[:, 3 * k + 2], quad_segs=q)
         for k in range(3)]
    return shapely.area(shapely.intersection(shapely.intersection(c[0], c[1]), c[2]))

res = {}
for q in (8, 64, 512):
    o = shp(X, q)
    np.save(f"out_shapely_q{q}.npy", o)
for name in ("configs", "traj_jupiter_earth", "traj_large_moon"):
    Y = X if name == "configs" else np.load(name + ".npz")["X"][:40000]
    res[name] = {}
    for q in (8, 64, 512):
        best = 1e9
        for _ in range(3):
            t = time.perf_counter(); shp(Y, q); best = min(best, time.perf_counter() - t)
        res[name][f"shapely_q{q}"] = 1e9 * best / len(Y)
    Z = Y[:5000]
    t = time.perf_counter()
    for row in Z:
        try:
            mvenn(row)
        except Exception:
            pass
    res[name]["mvenn"] = 1e9 * (time.perf_counter() - t) / len(Z)
print(json.dumps(res, indent=1))
json.dump(res, open("speed_py_tools.json", "w"), indent=1)
print("shapely", shapely.__version__, shapely.geos_version_string)
