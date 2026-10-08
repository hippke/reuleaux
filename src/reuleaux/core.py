#!/usr/bin/env python3
"""Exact area of the common overlap of three circles (standalone).

    A = area(D1 ∩ D2 ∩ D3),   D_i = disk with centre (x_i, y_i) and radius r_i

Method (see docs/notes.md for the full story):

1. Containment reduction: if disk i lies inside disk j, D_i ∩ D_j = D_i, so
   disk j is dropped. One disk left -> pi r^2; two left -> lens.
2. A disjoint (or externally tangent) pair -> 0.
3. Vertices = pairwise circle crossings strictly inside the third disk.
   Non-degenerate configurations have V in {0, 1, 2, 3, 4}:
   V = 0, 1 -> empty; V = 2 -> lens of one pair (or a tangential contact);
   V = 3 -> circular triangle; V = 4 -> circular quadrilateral.
4. Directed arc walk: from every vertex, the unique counter-clockwise arc
   that keeps the region on its left (no other crossing inside its open
   span, midpoint strictly inside the two other disks) -> one closed cycle.
5. Green's theorem per arc, in vector form (all position trigonometry
   cancels; only the arc angle needs one atan2):

       A = 1/2 * sum_arcs [ r^2 * dtheta + ox*(v_y - u_y) - oy*(v_x - u_x) ]

   for an arc of the circle with centre (ox, oy) and radius r, from point u
   to point v, CCW around its own centre, central angle dtheta in (0, 2 pi].

Because dtheta comes from the geometry (atan2 of cross and dot product), arcs
longer than pi ("reflex" arcs) need no special case. Chord-based closed
forms (Fewell 2006 Eq. 16, Kipping 2011 Eq. 37) got exactly that case wrong
(corrected by Gordon & Agol 2022, App. B); `fewell_area()` reproduces both
the historical and the corrected formula for comparison.

Interface (all lengths in arbitrary but consistent units):

    overlap_area(c1, c2, c3)       -> float                (fast path)
    overlap(c1, c2, c3)            -> Overlap              (area, case, arcs,
                                                            vertices, centroid,
                                                            boundary polyline)
    overlap_area_batch(x1, y1, r1, x2, y2, r2, x3, y3, r3) -> ndarray
    reference_area(c1, c2, c3)     -> (area, x_centroid, y_centroid)
                                      independent chord quadrature
    fewell_area(c1, c2, c3, corrected=True)  classical circular-triangle form
    pandora_er(xp, yp, rp, xm, ym, rm)      exact eclipse ratio (Pandora)
    pixelart_er(xp, yp, xm, ym, rp, rm, n)  Pandora's legacy pixel raster

with c = (x, y, r). numba is used if installed (set THREE_CIRCLES_NO_NUMBA=1
to force pure Python); numpy is required.

Command line:

    python -m reuleaux area  x1 y1 r1  x2 y2 r2  x3 y3 r3
    python -m reuleaux pandora xp yp rp xm ym rm [--grid 25]
    python -m reuleaux examples
    python -m reuleaux selftest [--n 5000]
    python -m reuleaux bench
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Arc", "Overlap", "overlap", "overlap_area", "overlap_area_batch",
    "reference_area", "fewell_area", "lens_area", "pandora_er",
    "pixelart_er", "EXAMPLES", "HAVE_NUMBA",
]

# ---------------------------------------------------------------------------
# optional numba
# ---------------------------------------------------------------------------

HAVE_NUMBA = False
if os.environ.get("THREE_CIRCLES_NO_NUMBA", "0") != "1":
    try:
        from numba import njit as _njit
        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover
        HAVE_NUMBA = False


def _jit(func):
    """njit if numba is available, otherwise the plain Python function."""
    if HAVE_NUMBA:
        return _njit(cache=True)(func)
    return func


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TWO_PI = 2.0 * math.pi
EPS_TAN = 1e-12   # tangency / disjointness / containment window (relative)
EPS_IN = 1e-9     # strict-inside margin of vertex and midpoint tests (relative)
EPS_COIN = 1e-9   # coincident-vertex window (relative)

CASE_EMPTY = 0
CASE_DISK = 1
CASE_LENS = 2
CASE_WALK = 3
CASE_DEGENERATE = 4

# workspace layout (one float64 array, so the jitted core never allocates)
_WS_ARCS = 0      # 4 arcs x 6: circle, ux, uy, vx, vy, dtheta
_WS_VERT = 24     # 6 vertices x 4: x, y, circle i, circle j
_WS_CP = 48       # 3 circles x 4 crossing points x 2
_WS_CPN = 72      # crossing-point counts per circle
_WS_NXT = 75      # out-arc of vertex u: next vertex
_WS_CIR = 81      # out-arc of vertex u: circle
_WS_VIS = 87      # visited flags
_WS_ALIVE = 93    # alive flags of the containment reduction
WS_SIZE = 96


# ---------------------------------------------------------------------------
# jitted kernel
# ---------------------------------------------------------------------------

@_jit
def _seg(r, delta):
    """Area between an arc of central angle delta and its chord,
    r²/2 (delta - sin delta); Taylor series below 0.5 rad (no cancellation
    for thin slivers)."""
    if delta < 0.5:
        x = delta * delta
        p = (1.0 / 6.0 - x * (1.0 / 120.0 - x * (1.0 / 5040.0 - x * (1.0 / 362880.0
             - x * (1.0 / 39916800.0 - x * (1.0 / 6227020800.0 - x * (1.0 / 1307674368000.0
             - x * (1.0 / 355687428096000.0 - x / 121645100408832000.0))))))))
        return 0.5 * r * r * delta * x * p
    return 0.5 * r * r * (delta - math.sin(delta))


@_jit
def _half_chord2(r1, r2, d):
    """Squared half chord h² of two crossing circles: 16 x (area of the
    triangle with sides r1, r2, d)² / (4 d²), with Kahan's stable Heron
    product (sides sorted a >= b >= c, brackets as written). Avoids the
    cancellation of the textbook r1² - a² for thin slivers."""
    a = r1
    b = r2
    c = d
    if a < b:
        a, b = b, a
    if b < c:
        b, c = c, b
    if a < b:
        a, b = b, a
    p = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return p / (4.0 * d * d)


@_jit
def lens_area(r1, r2, d):
    """Area of the intersection of two disks (radii r1, r2, centre distance d).
    Written as two circular segments with angles from atan2, so thin slivers
    keep full relative precision (the textbook acos form does not)."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    h = math.sqrt(max(_half_chord2(r1, r2, d), 0.0))
    a1 = ((d - r2) * (d + r2) + r1 * r1) / (2.0 * d)   # centre 1 -> radical line
    a2 = ((d - r1) * (d + r1) + r2 * r2) / (2.0 * d)   # centre 2 -> radical line
    return _seg(r1, 2.0 * math.atan2(h, a1)) + _seg(r2, 2.0 * math.atan2(h, a2))


@_jit
def _crossings(x1, y1, r1, x2, y2, r2, scale):
    """Crossing points of two circle boundaries.

    Returns (n, px, py, qx, qy), n in {0, 1, 2} (1 = tangent). With two
    crossings, p lies to the LEFT of the direction centre1 -> centre2."""
    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx * dx + dy * dy)
    if d <= EPS_TAN * scale:
        return 0, 0.0, 0.0, 0.0, 0.0
    a = ((d - r2) * (d + r2) + r1 * r1) / (2.0 * d)
    h2 = _half_chord2(r1, r2, d)
    ux = dx / d
    uy = dy / d
    bx = x1 + a * ux
    by = y1 + a * uy
    if h2 < -EPS_TAN * scale * scale:
        return 0, 0.0, 0.0, 0.0, 0.0
    if h2 <= EPS_TAN * scale * scale:
        return 1, bx, by, 0.0, 0.0
    h = math.sqrt(h2)
    return 2, bx - h * uy, by + h * ux, bx + h * uy, by - h * ux


@_jit
def _arc_angle(ux, uy, vx, vy):
    """CCW central angle from local vector u to local vector v, in (0, 2 pi]."""
    t = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if t <= 0.0:
        t += TWO_PI
    return t


@_jit
def _put_arc(ws, k, s, ux, uy, vx, vy, delta):
    b = _WS_ARCS + 6 * k
    ws[b] = s
    ws[b + 1] = ux
    ws[b + 2] = uy
    ws[b + 3] = vx
    ws[b + 4] = vy
    ws[b + 5] = delta


@_jit
def _lens_arcs(ws, a, b, cx, cy, cr, px, py, qx, qy):
    """Boundary of the lens D_a ∩ D_b (CCW): on circle a the arc inside D_b,
    on circle b the arc inside D_a. p, q = the two crossing points."""
    # make p the crossing to the left of a -> b
    if (cx[b] - cx[a]) * (py - cy[a]) - (cy[b] - cy[a]) * (px - cx[a]) < 0.0:
        px, qx = qx, px
        py, qy = qy, py
    # circle a: q -> p (through the side facing b)
    _put_arc(ws, 0, float(a), qx, qy, px, py,
             _arc_angle(qx - cx[a], qy - cy[a], px - cx[a], py - cy[a]))
    # circle b: p -> q (through the side facing a)
    _put_arc(ws, 1, float(b), px, py, qx, qy,
             _arc_angle(px - cx[b], py - cy[b], qx - cx[b], qy - cy[b]))


@_jit
def _inside(px, py, ox, oy, mm):
    return (px - ox) ** 2 + (py - oy) ** 2 < mm


@_jit
def _core(cx, cy, cr, ws):
    """Area of D0 ∩ D1 ∩ D2 (cx, cy, cr: length-3 arrays; ws: workspace).

    Returns (area, case, n_vertices, n_arcs, reflex). The boundary arcs are
    left in ws (layout _WS_ARCS), the vertices in ws (_WS_VERT). Degenerate
    (measure-zero) configurations return area = NaN, case CASE_DEGENERATE."""
    scale = max(cr[0], max(cr[1], cr[2]))
    if not (scale > 0.0):
        return 0.0, CASE_EMPTY, 0, 0, False
    margin = EPS_IN * scale

    # ---- containment reduction: disk i inside disk j -> drop j ------------
    for i in range(3):
        ws[_WS_ALIVE + i] = 1.0
    changed = True
    while changed:
        changed = False
        for i in range(3):
            if ws[_WS_ALIVE + i] == 0.0:
                continue
            for j in range(3):
                if j == i or ws[_WS_ALIVE + j] == 0.0:
                    continue
                d = math.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2)
                if d + cr[i] <= cr[j] + EPS_TAN * scale:
                    ws[_WS_ALIVE + j] = 0.0
                    changed = True
                    break
            if changed:
                break
    n_alive = 0
    a = -1
    b = -1
    for i in range(3):
        if ws[_WS_ALIVE + i] != 0.0:
            if a < 0:
                a = i
            elif b < 0:
                b = i
            n_alive += 1

    # ---- one disk left: full disk ------------------------------------------
    if n_alive == 1:
        r = cr[a]
        _put_arc(ws, 0, float(a), cx[a] + r, cy[a], cx[a] + r, cy[a], TWO_PI)
        return math.pi * r * r, CASE_DISK, 0, 1, False

    # ---- two disks left: lens or empty -------------------------------------
    if n_alive == 2:
        d = math.sqrt((cx[a] - cx[b]) ** 2 + (cy[a] - cy[b]) ** 2)
        if d >= cr[a] + cr[b] - EPS_TAN * scale:
            return 0.0, CASE_EMPTY, 0, 0, False
        area = lens_area(cr[a], cr[b], d)
        n, px, py, qx, qy = _crossings(cx[a], cy[a], cr[a], cx[b], cy[b], cr[b], scale)
        if n == 2:
            _lens_arcs(ws, a, b, cx, cy, cr, px, py, qx, qy)
            return area, CASE_LENS, 0, 2, False
        return area, CASE_LENS, 0, 0, False

    # ---- three disks: any disjoint pair -> empty ---------------------------
    for i in range(3):
        for j in range(i + 1, 3):
            d = math.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2)
            if d >= cr[i] + cr[j] - EPS_TAN * scale:
                return 0.0, CASE_EMPTY, 0, 0, False

    # ---- crossings per circle and vertices ----------------------------------
    for s in range(3):
        ws[_WS_CPN + s] = 0.0
    nv = 0
    for pair in range(3):
        if pair == 0:
            i = 0
            j = 1
            k = 2
        elif pair == 1:
            i = 0
            j = 2
            k = 1
        else:
            i = 1
            j = 2
            k = 0
        n, px, py, qx, qy = _crossings(cx[i], cy[i], cr[i], cx[j], cy[j], cr[j], scale)
        mmk = (cr[k] - margin) ** 2
        mpk = (cr[k] + margin) ** 2
        for t in range(n):
            if t == 0:
                x = px
                y = py
            else:
                x = qx
                y = qy
            for s in (i, j):
                c = int(ws[_WS_CPN + s])
                ws[_WS_CP + 8 * s + 2 * c] = x
                ws[_WS_CP + 8 * s + 2 * c + 1] = y
                ws[_WS_CPN + s] = c + 1
            dd = (x - cx[k]) ** 2 + (y - cy[k]) ** 2
            if dd < mmk:
                if nv >= 6:
                    return math.nan, CASE_DEGENERATE, nv, 0, False
                vb = _WS_VERT + 4 * nv
                ws[vb] = x
                ws[vb + 1] = y
                ws[vb + 2] = i
                ws[vb + 3] = j
                nv += 1
            elif dd <= mpk:
                # crossing ON the third circle (triple point): inside/outside
                # is undecidable, the caller re-evaluates nudged radii
                return math.nan, CASE_DEGENERATE, nv, 0, False

    # ---- dispatch on the vertex count ---------------------------------------
    if nv <= 1:
        return 0.0, CASE_EMPTY, nv, 0, False
    if nv == 2:
        i0 = int(ws[_WS_VERT + 2])
        j0 = int(ws[_WS_VERT + 3])
        if i0 == int(ws[_WS_VERT + 6]) and j0 == int(ws[_WS_VERT + 7]):
            # lens of one pair, fully inside the third disk
            d = math.sqrt((cx[i0] - cx[j0]) ** 2 + (cy[i0] - cy[j0]) ** 2)
            _lens_arcs(ws, i0, j0, cx, cy, cr, ws[_WS_VERT], ws[_WS_VERT + 1],
                       ws[_WS_VERT + 4], ws[_WS_VERT + 5])
            return lens_area(cr[i0], cr[j0], d), CASE_LENS, 2, 2, False
        return 0.0, CASE_EMPTY, 2, 0, False   # tangential contact
    if nv > 4:
        return math.nan, CASE_DEGENERATE, nv, 0, False

    # ---- coincident vertices (triple points) -> degenerate -----------------
    vsq = (EPS_COIN * scale) ** 2
    for u in range(nv):
        for v in range(u + 1, nv):
            if ((ws[_WS_VERT + 4 * u] - ws[_WS_VERT + 4 * v]) ** 2
                    + (ws[_WS_VERT + 4 * u + 1] - ws[_WS_VERT + 4 * v + 1]) ** 2) < vsq:
                return math.nan, CASE_DEGENERATE, nv, 0, False

    # ---- directed arc walk ----------------------------------------------------
    for u in range(nv):
        ux0 = ws[_WS_VERT + 4 * u]
        uy0 = ws[_WS_VERT + 4 * u + 1]
        ui = int(ws[_WS_VERT + 4 * u + 2])
        uj = int(ws[_WS_VERT + 4 * u + 3])
        n_out = 0
        for v in range(nv):
            if v == u:
                continue
            vx0 = ws[_WS_VERT + 4 * v]
            vy0 = ws[_WS_VERT + 4 * v + 1]
            vi = int(ws[_WS_VERT + 4 * v + 2])
            vj = int(ws[_WS_VERT + 4 * v + 3])
            for cand in range(2):
                s = ui if cand == 0 else uj
                if s != vi and s != vj:
                    continue
                ox = cx[s]
                oy = cy[s]
                r = cr[s]
                ux_ = ux0 - ox
                uy_ = uy0 - oy
                vx_ = vx0 - ox
                vy_ = vy0 - oy
                cr_ = ux_ * vy_ - uy_ * vx_
                dt_ = ux_ * vx_ + uy_ * vy_
                if dt_ > 0.0 and abs(cr_) < 1e-12 * r * r:
                    continue                      # (near-)zero span
                short = cr_ > 0.0                 # CCW span u -> v below pi
                # test 1: no other crossing of circle s inside the open span
                bad = False
                cnt = int(ws[_WS_CPN + s])
                for t in range(cnt):
                    qx0 = ws[_WS_CP + 8 * s + 2 * t]
                    qy0 = ws[_WS_CP + 8 * s + 2 * t + 1]
                    if (qx0 == ux0 and qy0 == uy0) or (qx0 == vx0 and qy0 == vy0):
                        continue
                    qx = qx0 - ox
                    qy = qy0 - oy
                    if short:
                        if (ux_ * qy - uy_ * qx) > 0.0 and (qx * vy_ - qy * vx_) > 0.0:
                            bad = True
                            break
                    else:
                        if not ((vx_ * qy - vy_ * qx) > 0.0 and (qx * uy_ - qy * ux_) > 0.0):
                            bad = True
                            break
                if bad:
                    continue
                # test 2: arc midpoint strictly inside the two other disks
                wx = ux_ + vx_
                wy = uy_ + vy_
                ww = wx * wx + wy * wy
                if ww < 1e-16 * r * r:            # antipodal end points
                    mx = ox - uy_
                    my = oy + ux_
                else:
                    f = r / math.sqrt(ww)
                    if not short:
                        f = -f
                    mx = ox + wx * f
                    my = oy + wy * f
                ok = True
                for o in range(3):
                    if o != s and not _inside(mx, my, cx[o], cy[o], (cr[o] - margin) ** 2):
                        ok = False
                        break
                if not ok:
                    continue
                if n_out > 0:
                    return math.nan, CASE_DEGENERATE, nv, 0, False
                ws[_WS_NXT + u] = v
                ws[_WS_CIR + u] = s
                n_out += 1
                break
        if n_out != 1:
            return math.nan, CASE_DEGENERATE, nv, 0, False

    # ---- follow the cycle --------------------------------------------------
    # Green's theorem per arc: 1/2 [r² dtheta + ox (v_y - u_y) - oy (v_x - u_x)]
    # (the position trigonometry cancels). Summed over the cycle this equals
    #     sum_arcs r²/2 (dtheta - sin dtheta)  +  shoelace(vertices),
    # i.e. circular segments on the chords plus the polygon of the vertices
    # (Fewell's "Heron triangle + segments", generalised). That form is
    # evaluated: it keeps full relative precision for tiny regions.
    for u in range(nv):
        ws[_WS_VIS + u] = 0.0
    p0x = ws[_WS_VERT]
    p0y = ws[_WS_VERT + 1]
    cur = 0
    steps = 0
    area = 0.0
    poly = 0.0
    reflex = False
    while ws[_WS_VIS + cur] == 0.0:
        ws[_WS_VIS + cur] = 1.0
        v = int(ws[_WS_NXT + cur])
        s = int(ws[_WS_CIR + cur])
        ox = cx[s]
        oy = cy[s]
        r = cr[s]
        ux0 = ws[_WS_VERT + 4 * cur]
        uy0 = ws[_WS_VERT + 4 * cur + 1]
        vx0 = ws[_WS_VERT + 4 * v]
        vy0 = ws[_WS_VERT + 4 * v + 1]
        crs = (ux0 - ox) * (vy0 - oy) - (uy0 - oy) * (vx0 - ox)     # r² sin(dtheta)
        delta = math.atan2(crs, (ux0 - ox) * (vx0 - ox) + (uy0 - oy) * (vy0 - oy))
        if delta <= 0.0:
            delta += TWO_PI
        if delta > math.pi:
            reflex = True
        if delta < 0.5:
            area += _seg(r, delta)
        else:
            area += 0.5 * (r * r * delta - crs)                      # no trig call
        poly += (ux0 - p0x) * (vy0 - p0y) - (uy0 - p0y) * (vx0 - p0x)
        _put_arc(ws, steps, float(s), ux0, uy0, vx0, vy0, delta)
        steps += 1
        cur = v
    if steps != nv or cur != 0:
        return math.nan, CASE_DEGENERATE, nv, 0, False
    area += 0.5 * poly

    # ---- plausibility: 0 <= A <= area of the smallest disk ------------------
    a_min = math.pi * min(cr[0], min(cr[1], cr[2])) ** 2
    if area < -1e-9 * scale * scale or area > a_min * (1.0 + 1e-9) + 1e-15 * scale * scale:
        return math.nan, CASE_DEGENERATE, nv, steps, reflex
    return max(area, 0.0), CASE_WALK, nv, steps, reflex


@_jit
def _area_shifted(x1, y1, r1, x2, y2, r2, x3, y3, r3, cx, cy, cr, ws):
    """Translate the origin to the centre of the smallest circle (reduces
    cancellation in the Green's sum), then call the core."""
    if r1 <= r2 and r1 <= r3:
        x0 = x1
        y0 = y1
    elif r2 <= r3:
        x0 = x2
        y0 = y2
    else:
        x0 = x3
        y0 = y3
    cx[0] = x1 - x0
    cy[0] = y1 - y0
    cr[0] = r1
    cx[1] = x2 - x0
    cy[1] = y2 - y0
    cr[1] = r2
    cx[2] = x3 - x0
    cy[2] = y3 - y0
    cr[2] = r3
    res = _core(cx, cy, cr, ws)
    return res[0], res[1], x0, y0


@_jit
def _area_batch(x1, y1, r1, x2, y2, r2, x3, y3, r3, out, case):
    cx = np.empty(3)
    cy = np.empty(3)
    cr = np.empty(3)
    ws = np.empty(WS_SIZE)
    for n in range(x1.shape[0]):
        a, c, _, _ = _area_shifted(x1[n], y1[n], r1[n], x2[n], y2[n], r2[n],
                                   x3[n], y3[n], r3[n], cx, cy, cr, ws)
        out[n] = a
        case[n] = c


@_jit
def _area_scalar(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    cx = np.empty(3)
    cy = np.empty(3)
    cr = np.empty(3)
    ws = np.empty(WS_SIZE)
    return _area_shifted(x1, y1, r1, x2, y2, r2, x3, y3, r3, cx, cy, cr, ws)[0]


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

CASE_NAMES = {
    CASE_EMPTY: "empty", CASE_DISK: "disk", CASE_LENS: "lens",
    CASE_WALK: "walk", CASE_DEGENERATE: "degenerate",
}


# Taylor coefficients of E(phi) = 4 sin³phi / (3 (2 phi - sin 2 phi)) - cos phi
# (offset of a circular segment's centroid from its chord, in units of r;
# phi = half the central angle), even powers phi², phi⁴, ..., phi²⁰.
_E_SERIES = (0.2, -0.012380952380952381, 0.00046031746031746033,
             -1.4821686250257679e-05, -5.286556477032667e-07, -3.414098708883289e-08,
             -7.466053601174316e-10, 5.972218811334104e-11, 8.629190801682645e-12,
             5.871044878379732e-13)


def _segment_offset(r, dtheta):
    """Distance of the centroid of a circular segment (central angle dtheta)
    from the midpoint of its chord, measured towards the arc."""
    phi = 0.5 * dtheta
    if phi < 0.5:
        x = phi * phi
        acc = 0.0
        for c in reversed(_E_SERIES):
            acc = acc * x + c
        return r * x * acc
    return r * (4.0 * math.sin(phi) ** 3 / (3.0 * (dtheta - math.sin(dtheta))) - math.cos(phi))


@dataclass
class Arc:
    """One boundary arc: CCW around its own centre (cx, cy) from the point
    (ux, uy) at angle theta0 to the point (vx, vy) at theta0 + dtheta."""
    circle: int          # index of the input circle (0, 1, 2)
    cx: float
    cy: float
    r: float
    theta0: float
    dtheta: float
    ux: float = 0.0
    uy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def reflex(self) -> bool:
        return self.dtheta > math.pi

    @property
    def segment_area(self) -> float:
        """Area between the arc and its chord."""
        return _seg(self.r, self.dtheta)

    def points(self, n: int = 200) -> np.ndarray:
        t = self.theta0 + np.linspace(0.0, self.dtheta, n)
        return np.column_stack((self.cx + self.r * np.cos(t), self.cy + self.r * np.sin(t)))


@dataclass
class Overlap:
    """Result of `overlap()`."""
    area: float
    case: str                      # empty | disk | lens | triangle | quadrilateral
    n_vertices: int
    reflex: bool                   # any boundary arc longer than pi
    arcs: list = field(default_factory=list)
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    vertex_pairs: list = field(default_factory=list)
    circles: tuple = ()
    perturbed: bool = False        # degenerate input, evaluated with nudged radii

    @property
    def centroid(self):
        """Centroid of the overlap region (None if the region is empty).

        Region = polygon of the arc end points + one circular segment per
        arc; segment centroids from the chord midpoint (series for thin
        segments), all relative to the first vertex: precision relative to
        the size of the region, not to the coordinates."""
        if not self.arcs or not (self.area > 0.0):
            return None
        if len(self.arcs) == 1 and self.arcs[0].dtheta == TWO_PI:
            return (float(self.arcs[0].cx), float(self.arcs[0].cy))
        px, py = self.arcs[0].ux, self.arcs[0].uy
        pts = [(a.ux - px, a.uy - py) for a in self.arcs]
        area = mx = my = 0.0
        for i in range(1, len(pts) - 1):            # polygon as a fan from p0
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            t = 0.5 * (x1 * y2 - y1 * x2)
            area += t
            mx += t * (x1 + x2) / 3.0
            my += t * (y1 + y2) / 3.0
        for a in self.arcs:                         # circular segments
            ux, uy, vx, vy = a.ux - px, a.uy - py, a.vx - px, a.vy - py
            dx, dy = vx - ux, vy - uy
            c = math.hypot(dx, dy)
            sa = a.segment_area
            e = _segment_offset(a.r, a.dtheta)
            # the segment lies to the right of the chord u -> v
            gx = 0.5 * (ux + vx) + e * dy / c
            gy = 0.5 * (uy + vy) - e * dx / c
            area += sa
            mx += sa * gx
            my += sa * gy
        return (float(px + mx / area), float(py + my / area))

    def boundary(self, n_per_arc: int = 200) -> np.ndarray:
        """Closed polyline of the region boundary (CCW), for plotting."""
        if not self.arcs:
            return np.zeros((0, 2))
        return np.vstack([arc.points(n_per_arc) for arc in self.arcs])

    def __str__(self):
        s = f"area = {self.area:.15g}  ({self.case}, V = {self.n_vertices}"
        if self.reflex:
            s += ", reflex arc"
        if self.perturbed:
            s += ", degenerate input: nudged radii"
        s += ")"
        for arc in self.arcs:
            s += (f"\n  arc on circle {arc.circle}: theta0 = {arc.theta0:+.6f}, "
                  f"dtheta = {arc.dtheta:.6f} rad{'  (reflex)' if arc.reflex else ''}")
        return s


def _as_circle(c):
    x, y, r = (float(v) for v in c)
    if not (r >= 0.0) or not all(map(math.isfinite, (x, y, r))):
        raise ValueError(f"invalid circle {c!r}: need finite x, y and r >= 0")
    return x, y, r


_PERTURB = ((1.0, -1.0, 0.5), (-1.0, 0.5, 1.0), (0.5, 1.0, -1.0))
PERTURB_REL = 1e-8   # radius nudge for degenerate input (10x the EPS_IN band)


def _analyse(cs):
    x0, y0 = min(cs, key=lambda c: c[2])[:2]
    cx = np.array([c[0] - x0 for c in cs])
    cy = np.array([c[1] - y0 for c in cs])
    cr = np.array([c[2] for c in cs])
    ws = np.zeros(WS_SIZE)
    area, case, nv, narcs, reflex = _core(cx, cy, cr, ws)
    return area, case, nv, narcs, reflex, ws, cx, cy, cr, x0, y0


def overlap(c1, c2, c3) -> Overlap:
    """Full analysis of D1 ∩ D2 ∩ D3; c = (x, y, r).

    Exactly degenerate input (a pairwise crossing lying on the third circle,
    i.e. a triple point) is evaluated with the radii nudged by
    ±PERTURB_REL * max(r) and the two areas averaged; `perturbed` is then
    True and the arcs/vertices belong to the + nudge."""
    circles = (_as_circle(c1), _as_circle(c2), _as_circle(c3))
    rs = [c[2] for c in circles]
    if min(rs) == 0.0:
        return Overlap(0.0, "empty", 0, False, circles=circles)
    res = _analyse(circles)
    perturbed = False
    if res[1] == CASE_DEGENERATE:
        perturbed = True
        scale = max(rs)
        for f in _PERTURB:
            plus = _analyse(tuple((x, y, r + PERTURB_REL * scale * f[i])
                                  for i, (x, y, r) in enumerate(circles)))
            minus = _analyse(tuple((x, y, r - PERTURB_REL * scale * f[i])
                                   for i, (x, y, r) in enumerate(circles)))
            if plus[1] != CASE_DEGENERATE and minus[1] != CASE_DEGENERATE:
                res = (0.5 * (plus[0] + minus[0]),) + plus[1:]
                break
        else:
            raise RuntimeError("degenerate geometry could not be resolved")
    area, case, nv, narcs, reflex, ws, cx, cy, cr, x0, y0 = res
    arcs = []
    for k in range(narcs):
        b = _WS_ARCS + 6 * k
        s = int(ws[b])
        ox, oy, r = cx[s], cy[s], cr[s]
        th0 = math.atan2(ws[b + 2] - oy, ws[b + 1] - ox)
        arcs.append(Arc(s, ox + x0, oy + y0, r, th0, ws[b + 5],
                        ws[b + 1] + x0, ws[b + 2] + y0, ws[b + 3] + x0, ws[b + 4] + y0))
    verts = np.array([[ws[_WS_VERT + 4 * u] + x0, ws[_WS_VERT + 4 * u + 1] + y0]
                      for u in range(nv)]).reshape(-1, 2)
    pairs = [(int(ws[_WS_VERT + 4 * u + 2]), int(ws[_WS_VERT + 4 * u + 3])) for u in range(nv)]
    name = CASE_NAMES[case]
    if case == CASE_WALK:
        name = "triangle" if nv == 3 else "quadrilateral"
    return Overlap(float(area), name, int(nv), bool(reflex), arcs, verts, pairs,
                   circles, perturbed)


def overlap_area(c1, c2, c3) -> float:
    """Area of D1 ∩ D2 ∩ D3 (fast path; degenerate input via `overlap()`)."""
    (x1, y1, r1), (x2, y2, r2), (x3, y3, r3) = c1, c2, c3
    a = _area_scalar(float(x1), float(y1), float(r1), float(x2), float(y2), float(r2),
                     float(x3), float(y3), float(r3))
    if a != a:
        a = overlap(c1, c2, c3).area
    return a


def overlap_area_batch(x1, y1, r1, x2, y2, r2, x3, y3, r3) -> np.ndarray:
    """Vectorised area for broadcastable arrays of circle parameters."""
    arrs = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in
                                 (x1, y1, r1, x2, y2, r2, x3, y3, r3)])
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a).ravel() for a in arrs]
    out = np.empty(flat[0].size)
    case = np.empty(flat[0].size, dtype=np.int64)
    _area_batch(*flat, out, case)
    for n in np.nonzero(case == CASE_DEGENERATE)[0]:
        out[n] = overlap(*[(flat[3 * c][n], flat[3 * c + 1][n], flat[3 * c + 2][n])
                           for c in range(3)]).area
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# independent reference: chord quadrature
# ---------------------------------------------------------------------------

_GL_CACHE = {}


def reference_area(c1, c2, c3, nodes: int = 48):
    """Independent reference (no arcs, no case logic).

    A = ∫ max(0, min_i(y_i + s_i(x)) - max_i(y_i - s_i(x))) dx with
    s_i = sqrt(r_i² - (x - x_i)²). The x axis is split at every circle's
    x-extent and at every pairwise crossing, so the integrand is smooth on
    each piece; x = a + (b - a)(1 - cos t)/2 removes the square-root end
    points, Gauss-Legendre in t does the rest (~1e-14 relative).
    Returns (area, x_centroid, y_centroid); centroid None if area == 0."""
    cs = [_as_circle(c) for c in (c1, c2, c3)]
    lo = max(x - r for x, y, r in cs)
    hi = min(x + r for x, y, r in cs)
    if not lo < hi:
        return 0.0, None, None
    brk = [lo, hi]
    for x, y, r in cs:
        brk += [x - r, x + r]
    for i in range(3):
        for j in range(i + 1, 3):
            (x1, y1, r1), (x2, y2, r2) = cs[i], cs[j]
            d = math.hypot(x2 - x1, y2 - y1)
            if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
                continue
            a = (d * d + r1 * r1 - r2 * r2) / (2 * d)
            h = math.sqrt(max(r1 * r1 - a * a, 0.0))
            ux, uy = (x2 - x1) / d, (y2 - y1) / d
            brk += [x1 + a * ux - h * uy, x1 + a * ux + h * uy]
    brk = np.unique(np.clip(np.array(brk), lo, hi))
    scale = max(c[2] for c in cs)
    rules = []
    for m in (nodes, 2 * nodes):
        if m not in _GL_CACHE:
            g, w = np.polynomial.legendre.leggauss(m)
            _GL_CACHE[m] = (0.5 * math.pi * (g + 1.0), 0.5 * math.pi * w)   # t in (0, pi)
        rules.append(_GL_CACHE[m])

    def piece(a, b, rule):
        t, wt = rule
        x = a + 0.5 * (b - a) * (1.0 - np.cos(t))
        wx = wt * 0.5 * (b - a) * np.sin(t)
        top = np.full_like(x, np.inf)
        bot = np.full_like(x, -np.inf)
        for xc, yc, r in cs:
            s = np.sqrt(np.maximum(r * r - (x - xc) ** 2, 0.0))
            top = np.minimum(top, yc + s)
            bot = np.maximum(bot, yc - s)
        f = np.maximum(top - bot, 0.0)
        mid = np.where(f > 0.0, 0.5 * (top + bot), 0.0)
        return np.array([np.sum(wx * f), np.sum(wx * f * x), np.sum(wx * f * mid)])

    def adaptive(a, b, depth):
        # two rule orders; bisect where they disagree (near-tangent kinks)
        lo_, hi_ = piece(a, b, rules[0]), piece(a, b, rules[1])
        if depth >= 40 or abs(hi_[0] - lo_[0]) <= 1e-16 * scale * scale + 1e-15 * abs(hi_[0]):
            return hi_
        m = 0.5 * (a + b)
        return adaptive(a, m, depth + 1) + adaptive(m, b, depth + 1)

    tot = np.zeros(3)
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a > 0.0:
            tot += adaptive(a, b, 0)
    area, mx, my = tot
    if area <= 0.0:
        return 0.0, None, None
    return float(area), float(mx / area), float(my / area)


# ---------------------------------------------------------------------------
# classical closed form (Fewell 2006; Gordon & Agol 2022 correction)
# ---------------------------------------------------------------------------

def fewell_area(c1, c2, c3, corrected: bool = True) -> float:
    """Circular-triangle area as Heron triangle on the three chords plus
    three circular segments (Fewell 2006 Eq. 1).

    corrected=True : reflex segments as r²(pi - asin(c/2r)) + (c/4)sqrt(4r² - c²)
                     (Gordon & Agol 2022, App. B)
    corrected=False: reflex segments as r² asin(c/2r) + (c/4)sqrt(4r² - c²)
                     (Fewell 2006 Eq. 16, Kipping 2011 Eq. 37: the historical error)

    Only defined for a circular triangle (V = 3); the vertices come from
    `overlap()`. A segment is reflex when the circle's centre lies on the
    other side of the chord than the third vertex."""
    ov = overlap(c1, c2, c3)
    if ov.case != "triangle":
        raise ValueError(f"fewell_area needs a circular triangle, got {ov.case}")
    circ = ov.circles
    area = 0.0
    chords = []
    for k in range(3):
        on = [ov.vertices[u] for u in range(3) if k in ov.vertex_pairs[u]]
        off = [ov.vertices[u] for u in range(3) if k not in ov.vertex_pairs[u]][0]
        p0, p1 = on
        c = float(np.hypot(*(p1 - p0)))
        chords.append(c)
        r = circ[k][2]
        nvec = np.array([p1[1] - p0[1], p0[0] - p1[0]])
        o = np.array(circ[k][:2])
        reflex = np.dot(o - p0, nvec) * np.dot(off - p0, nvec) < 0.0
        asn = math.asin(min(1.0, c / (2.0 * r)))
        root = 0.25 * c * math.sqrt(max(4.0 * r * r - c * c, 0.0))
        if not reflex:
            area += r * r * asn - root
        elif corrected:
            area += r * r * (math.pi - asn) + root
        else:
            area += r * r * asn + root
    c1_, c2_, c3_ = chords
    heron = 0.25 * math.sqrt(max((c1_ + c2_ + c3_) * (c2_ + c3_ - c1_)
                                 * (c1_ + c3_ - c2_) * (c1_ + c2_ - c3_), 0.0))
    return heron + area


# ---------------------------------------------------------------------------
# Pandora conventions (star at (0, 0) with radius 1)
# ---------------------------------------------------------------------------

def pandora_er(xp, yp, rp, xm, ym, rm) -> float:
    """Exact eclipse ratio er = A(star ∩ planet ∩ moon) / A(star ∩ moon),
    with er = 1 when the moon is off the star (Pandora's convention)."""
    a_sm = lens_area(1.0, rm, math.hypot(xm, ym))
    if a_sm <= 0.0:
        return 1.0
    return min(1.0, overlap_area((0.0, 0.0, 1.0), (xp, yp, rp), (xm, ym, rm)) / a_sm)


def pixelart_counts(xp, yp, xm, ym, rp, rm, numerical_grid=25):
    """numpy re-implementation of Pandora's legacy `pixelart()` raster
    (Hippke & Heller 2022, pandoramoon/eclipse.py), same formulas and grid:
    moon-centred (n+1) x (n+1) pixel centres, moon radius n/2 pixels, the
    'anti_aliasing' dilations included. Returns (image, n_triple, frac, cci)
    with frac = n_triple / (pi (n/2)²) (Pandora's estimate of
    A(star ∩ planet ∩ moon) / (pi rm²)) and cci = A(star ∩ moon) / (pi rm²)."""
    n = int(numerical_grid)
    if n % 2 == 0:
        n += 1
    r_star = (1.0 / rm) * n
    idx = np.arange(n + 1, dtype=float)
    X, Y = np.meshgrid(idx, idx, indexing="ij")
    image = np.zeros((n + 1, n + 1), dtype=np.int8)
    mid = int(math.ceil(n / 2))
    aa = math.sqrt((n + 1) ** 2 - n ** 2) / 2.0
    q = (n - 2 * X[:mid, :mid]) ** 2 + (n - 2 * Y[:mid, :mid]) ** 2 < n ** 2 + aa
    image[:mid, :mid][q] = 3
    image[mid:, :mid] = np.flipud(image[:mid, :mid])
    image[:, mid:] = np.fliplr(image[:, :mid])
    aa = -0.5 / n
    d_star = np.sqrt((xm * r_star + 2 * X - n) ** 2 + (ym * r_star + 2 * Y - n) ** 2)
    image = image + 5 * (d_star < r_star - aa)
    d_pl = np.sqrt((-(xp - xm) * r_star + 2 * X - n) ** 2 + (-(yp - ym) * r_star + 2 * Y - n) ** 2)
    image = image + 2 * (d_pl < (rp / rm) * n - aa)
    n_triple = int(np.sum(image == 10))
    frac = n_triple / (math.pi * (n / 2.0) ** 2)
    zm = math.hypot(xm, ym)
    if zm < 1.0 + rm and 1.0 - rm > zm:
        cci = 1.0
    elif zm < 1.0 + rm:
        cci = lens_area(1.0, rm, zm) / (math.pi * rm * rm)
    else:
        cci = 0.0
    return image, n_triple, frac, cci


def pixelart_er(xp, yp, xm, ym, rp, rm, numerical_grid=25) -> float:
    """Pandora's legacy raster estimate of er (argument order as in Pandora!)."""
    _, _, frac, cci = pixelart_counts(xp, yp, xm, ym, rp, rm, numerical_grid)
    if cci > 0:
        return min(1.0, 1.0 - (cci - frac) / cci)
    return 1.0


# ---------------------------------------------------------------------------
# curated examples (Pandora convention: star (0, 0, 1), planet, moon)
# ---------------------------------------------------------------------------

EXAMPLES = [
    dict(name="ingress-lens", expect="lens",
         desc="Moon fully on the star, partial planet-moon overlap: triple = planet-moon lens",
         xp=0.95, yp=0.15, rp=0.10, xm=0.90, ym=0.25, rm=0.035),
    dict(name="realistic-tri-1", expect="triangle",
         desc="Planet and moon straddle the limb, lens cut by the stellar rim",
         xp=0.96, yp=0.03, rp=0.10, xm=1.00, ym=-0.02, rm=0.05),
    dict(name="realistic-tri-2", expect="triangle",
         desc="As realistic-tri-1, thinner crescent",
         xp=0.94, yp=0.06, rp=0.11, xm=1.00, ym=-0.03, rm=0.045),
    dict(name="realistic-quad", expect="quadrilateral",
         desc="Moon straddles the star-planet lens (Fewell case d), collinear",
         xp=0.95, yp=0.00, rp=0.12, xm=0.92, ym=0.00, rm=0.11),
    dict(name="realistic-quad-b", expect="quadrilateral",
         desc="Fewell case d, slightly tilted",
         xp=0.96, yp=0.02, rp=0.13, xm=0.92, ym=-0.01, rm=0.10),
    dict(name="moon-in-planet-limb", expect="lens",
         desc="Moon inside the planet, moon on the limb: triple = star-moon lens",
         xp=0.94, yp=0.02, rp=0.15, xm=0.99, ym=-0.01, rm=0.05),
    dict(name="overlap-fully-on-star", expect="lens",
         desc="Both bodies fully on the star, partial overlap",
         xp=0.50, yp=0.30, rp=0.12, xm=0.62, ym=0.35, rm=0.05),
    dict(name="overlap-off-star", expect="empty",
         desc="Planet-moon overlap entirely off the star",
         xp=1.15, yp=0.05, rp=0.12, xm=1.20, ym=0.10, rm=0.05),
    dict(name="external-tangency", expect="empty",
         desc="Planet and moon externally tangent on the star (d = rp + rm)",
         xp=0.70, yp=0.30, rp=0.10, xm=0.85, ym=0.30, rm=0.05),
    dict(name="internal-tangency", expect="lens",
         desc="Moon internally tangent inside the planet (d = rp - rm), on the limb",
         xp=0.90, yp=0.00, rp=0.14, xm=0.90 + 0.09 * 0.8660254037844387,
         ym=-0.09 * 0.5, rm=0.05),
    dict(name="full-disk", expect="disk",
         desc="Moon fully inside planet and star: triple = pi rm^2",
         xp=0.40, yp=0.20, rp=0.20, xm=0.45, ym=0.18, rm=0.04),
    dict(name="empty-case-i", expect="empty",
         desc="All three pairs intersect, no common overlap (Fewell case i)",
         xp=0.98, yp=0.05, rp=0.09, xm=1.06, ym=-0.04, rm=0.05),
    dict(name="moon-tangent-star-in", expect="disk",
         desc="Moon tangent to the stellar rim from inside and tangent inside the planet",
         xp=0.92, yp=0.03, rp=0.09, xm=0.96, ym=0.00, rm=0.04),
    dict(name="taxonomy-tri", expect="triangle",
         desc="Textbook circular triangle, moon on the limb",
         xp=0.7458, yp=0.3111, rp=0.3329, xm=0.9803, ym=0.2877, rm=0.2634),
    dict(name="taxonomy-tri-moonbig", expect="triangle",
         desc="Circular triangle with the moon larger than the planet",
         xp=0.0830, yp=-0.5095, rp=0.5666, xm=-0.0311, ym=-1.4987, rm=0.5871),
    dict(name="reflex-deep-1", expect="triangle", reflex=True,
         desc="Reflex circular triangle (moon arc 3.78 rad > pi)",
         xp=0.1998, yp=-0.5836, rp=0.5443, xm=0.1995, ym=-0.6157, rm=0.5280),
    dict(name="reflex-deep-2", expect="triangle", reflex=True,
         desc="Deep reflex circular triangle (moon arc 3.91 rad)",
         xp=-0.1829, yp=0.2847, rp=0.7721, xm=-0.2457, ym=0.6425, rm=0.5114),
    dict(name="reflex-nearpi", expect="triangle", reflex=True,
         desc="Reflex arc barely above pi (3.147 rad): branch-selection stress",
         xp=0.4497, yp=-0.4356, rp=0.3751, xm=0.7654, ym=-0.5950, rm=0.1224),
    dict(name="taxonomy-quad", expect="quadrilateral",
         desc="Fewell case d quadrilateral, moon on the limb",
         xp=-0.5404, yp=-0.0954, rp=0.4707, xm=-1.0743, ym=-0.2841, rm=0.3673),
    dict(name="taxonomy-quad-2", expect="quadrilateral",
         desc="Fewell case d quadrilateral, nearly equal radii",
         xp=0.5039, yp=-0.2216, rp=0.5644, xm=0.8401, ym=-0.3363, rm=0.5392),
]


def example_circles(ex):
    """(star, planet, moon) circles of an EXAMPLES entry."""
    return (0.0, 0.0, 1.0), (ex["xp"], ex["yp"], ex["rp"]), (ex["xm"], ex["ym"], ex["rm"])


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _random_configs(rng, n):
    """Mixed families: general, Pandora-like (small bodies near the limb),
    nearly equal radii (reflex-rich), wildly different scales."""
    out = []
    for k in range(n):
        fam = k % 4
        if fam == 0:
            cs = [(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0.05, 1.2))
                  for _ in range(3)]
        elif fam == 1:
            rp = rng.uniform(0.02, 0.2)
            rm = rng.uniform(0.005, min(0.5 * rp, 0.1))
            ap = rng.uniform(0, TWO_PI)
            dp = rng.uniform(0.75, 1.05)
            xp, yp = dp * math.cos(ap), dp * math.sin(ap)
            am = rng.uniform(0, TWO_PI)
            dm = rng.uniform(0.0, rp + rm)
            cs = [(0.0, 0.0, 1.0), (xp, yp, rp),
                  (xp + dm * math.cos(am), yp + dm * math.sin(am), rm)]
        elif fam == 2:
            r = rng.uniform(0.5, 1.0)
            cs = [(rng.normal(0, 0.35 * r), rng.normal(0, 0.35 * r), r * rng.uniform(0.85, 1.15))
                  for _ in range(3)]
        else:
            s = 10.0 ** rng.uniform(-6, 6)
            x0, y0 = rng.uniform(-1e3, 1e3, 2) * s
            cs = [(x0 + s * rng.uniform(-1, 1), y0 + s * rng.uniform(-1, 1), s * rng.uniform(0.2, 1.2))
                  for _ in range(3)]
        out.append(cs)
    return out


def selftest(n_fuzz=20000, seed=1, verbose=True):
    """Validate against the independent chord quadrature and other identities.
    Returns True if every check passes."""
    ok = True

    def report(name, passed, info=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}{(': ' + info) if info else ''}")

    if verbose:
        print(f"reuleaux self test (numba: {HAVE_NUMBA})")

    # 1. curated examples: case labels, reflex flags, area vs reference
    worst = 0.0
    labels_ok = True
    for ex in EXAMPLES:
        cs = example_circles(ex)
        ov = overlap(*cs)
        ref = reference_area(*cs)[0]
        worst = max(worst, abs(ov.area - ref) / (math.pi * ex["rm"] ** 2))
        labels_ok &= ov.case == ex["expect"] and (ov.reflex or not ex.get("reflex", False))
    report("20 curated examples: case labels and reflex flags", labels_ok)
    report("20 curated examples: area vs chord quadrature", worst < 1e-12,
           f"max |dA| = {worst:.1e} of the moon area")

    # 2. fuzz vs the independent reference
    rng = np.random.default_rng(seed)
    cfgs = _random_configs(rng, n_fuzz)
    counts = {}
    worst = worst_c = 0.0
    n_pert = 0
    for cs in cfgs:
        ov = overlap(*cs)
        counts[ov.case] = counts.get(ov.case, 0) + 1
        n_pert += ov.perturbed
        ref, xr, yr = reference_area(*cs)
        rmin = min(c[2] for c in cs)
        unit = math.pi * rmin ** 2
        worst = max(worst, abs(ov.area - ref) / unit)
        if ref > 1e-6 * unit:
            xc, yc = ov.centroid
            worst_c = max(worst_c, math.hypot(xc - xr, yc - yr) / rmin)
    info = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    report(f"{n_fuzz} random configurations vs chord quadrature", worst < 1e-10,
           f"max |dA| = {worst:.1e} of the smallest disk; {info}; {n_pert} nudged")
    report("centroid vs chord quadrature", worst_c < 1e-9,
           f"max offset {worst_c:.1e} of the smallest radius")

    # 3. fast scalar and batch paths equal the full analysis
    sub = cfgs[:2000]
    a_full = np.array([overlap(*cs).area for cs in sub])
    a_fast = np.array([overlap_area(*cs) for cs in sub])
    cols = [np.array([cs[c][q] for cs in sub]) for c in range(3) for q in range(3)]
    a_batch = overlap_area_batch(*cols)
    report("overlap_area / overlap_area_batch == overlap().area",
           np.array_equal(a_full, a_fast) and np.array_equal(a_full, a_batch))

    # 4. the classical formula: corrected == walk, historical != walk on reflex
    worst = 0.0
    worst_bad = 0.0
    n_tri = n_refl = 0
    for cs in cfgs:
        ov = overlap(*cs)
        if ov.case != "triangle" or ov.perturbed:
            continue
        n_tri += 1
        unit = math.pi * min(c[2] for c in cs) ** 2
        worst = max(worst, abs(fewell_area(*cs) - ov.area) / unit)
        if ov.reflex:
            n_refl += 1
            worst_bad = max(worst_bad, abs(fewell_area(*cs, corrected=False) - ov.area) / unit)
    report(f"Fewell Eq. 1 / Gordon & Agol corrected == walk ({n_tri} triangles)",
           worst < 1e-9, f"max {worst:.1e}")
    report(f"historical reflex formula (Fewell Eq. 16) is wrong ({n_refl} reflex)",
           worst_bad > 1e-3, f"max error {worst_bad:.3f} of the smallest disk")

    # 5. degenerate and limiting geometries
    s3 = math.sqrt(3.0) / 2.0
    tri_point = overlap((1.0, 0.0, 1.0), (-0.5, s3, 1.0), (-0.5, -s3, 1.0))
    report("triple point (three circles through one point): area 0",
           abs(tri_point.area) < 1e-8, f"{tri_point.area:.1e}, nudged={tri_point.perturbed}")
    same = overlap((0.3, 0.2, 0.5), (0.3, 0.2, 0.5), (0.3, 0.2, 0.5))
    report("three identical circles: pi r^2", abs(same.area - math.pi * 0.25) < 1e-15)
    conc = overlap((0, 0, 1.0), (0, 0, 0.5), (0, 0, 0.7))
    report("concentric circles: smallest disk", abs(conc.area - math.pi * 0.25) < 1e-15)
    # third circle through both crossing points of the other two (V = 4 corners coincide)
    cp = overlap((-0.5, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.0, math.sqrt(0.75)))
    ref = reference_area((-0.5, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.0, math.sqrt(0.75)))[0]
    report("third circle through both lens corners (triple points)",
           abs(cp.area - ref) < 1e-6 * ref and cp.perturbed,
           f"{cp.area:.12f} vs {ref:.12f} ({cp.case}, nudged={cp.perturbed})")

    # 6. Pandora's legacy raster converges to the exact er (and is coarse at n = 25)
    err = {}
    for grid in (25, 399):
        worst = 0.0
        for ex in EXAMPLES:
            e_exact = pandora_er(ex["xp"], ex["yp"], ex["rp"], ex["xm"], ex["ym"], ex["rm"])
            e_pix = pixelart_er(ex["xp"], ex["yp"], ex["xm"], ex["ym"], ex["rp"], ex["rm"], grid)
            worst = max(worst, abs(e_exact - e_pix))
        err[grid] = worst
    report("legacy pixel raster -> exact er as the grid is refined",
           err[399] < 0.1 * err[25],
           f"max |er error| n=25: {err[25]:.4f}, n=399: {err[399]:.5f}")
    if verbose:
        print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def bench(repeat=5):
    cs_list = [example_circles(ex) for ex in EXAMPLES]
    cols = [np.array([cs[c][q] for cs in cs_list]) for c in range(3) for q in range(3)]
    big = [np.tile(c, 5000) for c in cols]          # 100,000 evaluations
    overlap_area_batch(*cols)                        # compile
    overlap_area(*cs_list[0])
    print(f"numba: {HAVE_NUMBA}; 20 curated examples (Pandora geometries)")
    best = min(_timeit(lambda: overlap_area_batch(*big)) for _ in range(repeat))
    print(f"  overlap_area_batch : {best / big[0].size * 1e9:8.0f} ns per configuration")
    best = min(_timeit(lambda: [overlap_area(*cs) for cs in cs_list]) for _ in range(repeat))
    print(f"  overlap_area       : {best / len(cs_list) * 1e9:8.0f} ns per call (from Python)")
    best = min(_timeit(lambda: [overlap(*cs) for cs in cs_list]) for _ in range(repeat))
    print(f"  overlap (full)     : {best / len(cs_list) * 1e6:8.1f} us per call")
    best = min(_timeit(lambda: [pixelart_er(ex["xp"], ex["yp"], ex["xm"], ex["ym"], ex["rp"],
                                            ex["rm"], 25) for ex in EXAMPLES])
               for _ in range(repeat))
    print(f"  pixelart_er n=25   : {best / len(EXAMPLES) * 1e6:8.1f} us per call (numpy port)")
    best = min(_timeit(lambda: [reference_area(*cs) for cs in cs_list]) for _ in range(repeat))
    print(f"  reference_area     : {best / len(cs_list) * 1e6:8.1f} us per call")


def _timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("area", help="overlap of three circles: x1 y1 r1 x2 y2 r2 x3 y3 r3")
    pa.add_argument("v", type=float, nargs=9)
    pp = sub.add_parser("pandora", help="eclipse ratio: xp yp rp xm ym rm (star = (0, 0, 1))")
    pp.add_argument("v", type=float, nargs=6)
    pp.add_argument("--grid", type=int, default=25, help="pixel grid of the legacy raster")
    sub.add_parser("examples", help="table of the 20 curated examples")
    ps = sub.add_parser("selftest", help="validate against an independent quadrature")
    ps.add_argument("--n", type=int, default=5000)
    sub.add_parser("bench", help="timings")
    args = p.parse_args(argv)

    if args.cmd == "area":
        v = args.v
        cs = (v[0:3], v[3:6], v[6:9])
        ov = overlap(*cs)
        print(ov)
        if ov.centroid is not None:
            print(f"  centroid = ({ov.centroid[0]:.12g}, {ov.centroid[1]:.12g})")
        print(f"  reference (chord quadrature) = {reference_area(*cs)[0]:.15g}")
        if ov.case == "triangle":
            print(f"  Fewell Eq. 1 / Gordon & Agol = {fewell_area(*cs):.15g}")
            if ov.reflex:
                print(f"  Fewell Eq. 16 (historical)   = {fewell_area(*cs, corrected=False):.15g}")
    elif args.cmd == "pandora":
        xp, yp, rp, xm, ym, rm = args.v
        ov = overlap((0, 0, 1), (xp, yp, rp), (xm, ym, rm))
        print(f"A(star ∩ planet ∩ moon) = {ov.area:.15g}  ({ov.case})")
        print(f"er exact                = {pandora_er(xp, yp, rp, xm, ym, rm):.15g}")
        print(f"er pixel raster n={args.grid:<4d}  = "
              f"{pixelart_er(xp, yp, xm, ym, rp, rm, args.grid):.15g}")
    elif args.cmd == "examples":
        print(f"{'name':24s} {'case':14s} {'V':>2s} {'reflex':>6s} {'A / (pi rm^2)':>15s} "
              f"{'er exact':>10s} {'er n=25':>10s}")
        for ex in EXAMPLES:
            cs = example_circles(ex)
            ov = overlap(*cs)
            e = pandora_er(ex["xp"], ex["yp"], ex["rp"], ex["xm"], ex["ym"], ex["rm"])
            e25 = pixelart_er(ex["xp"], ex["yp"], ex["xm"], ex["ym"], ex["rp"], ex["rm"], 25)
            print(f"{ex['name']:24s} {ov.case:14s} {ov.n_vertices:2d} {str(ov.reflex):>6s} "
                  f"{ov.area / (math.pi * ex['rm'] ** 2):15.10f} {e:10.6f} {e25:10.6f}")
    elif args.cmd == "selftest":
        return 0 if selftest(args.n) else 1
    elif args.cmd == "bench":
        bench()
    return 0


if __name__ == "__main__":
    sys.exit(main())
