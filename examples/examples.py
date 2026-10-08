#!/usr/bin/env python3
"""Usage examples for reuleaux (run: python examples.py)."""

import math

import numpy as np

import reuleaux as tc


def header(text):
    print("\n" + text + "\n" + "-" * len(text))


# 1. The simplest call: area only -------------------------------------------
header("1. Area of the common overlap of three circles")
c1 = (0.0, 0.0, 1.0)
c2 = (0.8, 0.3, 0.6)
c3 = (0.5, -0.4, 0.7)
print("overlap_area(c1, c2, c3) =", tc.overlap_area(c1, c2, c3))

# 2. Full analysis: case, arcs, vertices, centroid ---------------------------
header("2. Full analysis")
ov = tc.overlap(c1, c2, c3)
print(ov)
print("vertices:\n", ov.vertices)
print("centroid:", ov.centroid)
print("independent chord quadrature (area, x_c, y_c):", tc.reference_area(c1, c2, c3))

# 3. Arbitrary units and positions: the result scales as length² -----------
header("3. Scale invariance (km, far from the origin)")
s, x0, y0 = 6.957e5, 1.5e8, -2.0e7           # solar radius in km, offset by 1 au
cs = [(x0 + s * x, y0 + s * y, s * r) for x, y, r in (c1, c2, c3)]
a_km = tc.overlap_area(*cs)
print(f"area = {a_km:.12e} km^2, area / s^2 = {a_km / s**2:.15f} (unit circles: {ov.area:.15f})")

# 4. The historical reflex error ---------------------------------------------
header("4. Fewell (2006) Eq. 16 / Kipping (2011) Eq. 37 versus the exact area")
ex = next(e for e in tc.EXAMPLES if e["name"] == "reflex-deep-1")
cs = tc.example_circles(ex)
ov = tc.overlap(*cs)
moon = math.pi * ex["rm"] ** 2
a_ok = tc.fewell_area(*cs, corrected=True)
a_bad = tc.fewell_area(*cs, corrected=False)
print(f"{ex['desc']}")
print(f"  arc walk (this code)            : {ov.area:.12f}")
print(f"  Fewell Eq. 1 + Gordon & Agol fix: {a_ok:.12f}")
print(f"  Fewell Eq. 16 / Kipping Eq. 37  : {a_bad:.12f}  "
      f"(error {100 * (a_bad - ov.area) / moon:+.1f} % of the moon disk)")

# 5. Pandora: eclipse ratio of a moon behind its planet on the stellar limb -
header("5. Pandora convention: star (0, 0, 1), planet, moon")
xp, yp, rp, xm, ym, rm = 0.96, 0.03, 0.10, 1.00, -0.02, 0.05
print(f"exact er             = {tc.pandora_er(xp, yp, rp, xm, ym, rm):.10f}")
for n in (25, 99, 399):
    print(f"pixel raster n = {n:3d} = {tc.pixelart_er(xp, yp, xm, ym, rp, rm, n):.10f}")

# 6. Batch evaluation along a trajectory -------------------------------------
header("6. Batch: moon crossing the planet on the stellar limb")
x = np.linspace(0.80, 1.15, 8)
areas = tc.overlap_area_batch(0.0, 0.0, 1.0, 0.98, 0.0, 0.10, x, 0.02, 0.05)
for xi, ai in zip(x, areas):
    print(f"  x_moon = {xi:.3f}:  A / (pi r_m^2) = {ai / (math.pi * 0.05**2):.6f}")

# 7. Degenerate input -------------------------------------------------------
header("7. Degenerate input (third circle through both corners of a lens)")
ov = tc.overlap((-0.5, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.0, math.sqrt(0.75)))
print(ov)
print("exact lens area:", tc.lens_area(1.0, 1.0, 1.0))

# 8. All curated examples -----------------------------------------------------
header("8. The 20 curated examples (python reuleaux examples)")
tc.main(["examples"])
