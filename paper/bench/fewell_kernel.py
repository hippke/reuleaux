"""Fewell (2006) closed form for the circular triangle, as published.

Vertices from Fewell's Eq. (6) (expanded half-chord), chords, Heron's
formula on the chords (his Delta/4), three circular segments with arcsin;
reflex segments either with Fewell's Eq. (16) (= Kipping 2011, Eq. 37) or
with the correction of Gordon & Agol (2022, App. B). Implemented in the
global frame (no rotations), which is cheaper than the literal 7-step
recipe, so its timing is a lower bound. Only valid for a circular
triangle (one vertex per pair); returns NaN otherwise.
"""

import math

from numba import njit


@njit(cache=True)
def _pair(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    """Crossing of circles 1, 2 that lies inside circle 3 (Fewell Eqs. 6, 14)."""
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    d = math.sqrt(d2)
    xa = (r1 * r1 - r2 * r2 + d2) / (2.0 * d)
    disc = 2.0 * d2 * (r1 * r1 + r2 * r2) - (r1 * r1 - r2 * r2) ** 2 - d2 * d2
    if disc <= 0.0:
        return False, 0.0, 0.0
    ya = math.sqrt(disc) / (2.0 * d)
    ux = dx / d
    uy = dy / d
    px = x1 + xa * ux - ya * uy
    py = y1 + xa * uy + ya * ux
    qx = x1 + xa * ux + ya * uy
    qy = y1 + xa * uy - ya * ux
    ip = (px - x3) ** 2 + (py - y3) ** 2 < r3 * r3
    iq = (qx - x3) ** 2 + (qy - y3) ** 2 < r3 * r3
    if ip == iq:
        return False, 0.0, 0.0
    if ip:
        return True, px, py
    return True, qx, qy


@njit(cache=True)
def _segment(r, c, ox, oy, ax, ay, bx, by, tx, ty, corrected):
    """Segment on chord a-b of circle (o, r); t = opposite vertex."""
    nx = by - ay
    ny = ax - bx
    s_o = (ox - ax) * nx + (oy - ay) * ny
    s_t = (tx - ax) * nx + (ty - ay) * ny
    asn = math.asin(min(1.0, c / (2.0 * r)))
    root = 0.25 * c * math.sqrt(max(4.0 * r * r - c * c, 0.0))
    if s_o * s_t >= 0.0:                       # minor segment
        return r * r * asn - root
    if corrected:                              # Gordon & Agol (2022), App. B
        return r * r * (math.pi - asn) + root
    return r * r * asn + root                  # Fewell Eq. 16 / Kipping Eq. 37


@njit(cache=True)
def fewell_triangle(x1, y1, r1, x2, y2, r2, x3, y3, r3, corrected):
    ok, ax, ay = _pair(x1, y1, r1, x2, y2, r2, x3, y3, r3)      # vertex 12
    if not ok:
        return math.nan
    ok, bx, by = _pair(x1, y1, r1, x3, y3, r3, x2, y2, r2)      # vertex 13
    if not ok:
        return math.nan
    ok, cx, cy = _pair(x2, y2, r2, x3, y3, r3, x1, y1, r1)      # vertex 23
    if not ok:
        return math.nan
    c1 = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)   # chord on circle 1 (vertices 12, 13)
    c2 = math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)   # chord on circle 2 (12, 23)
    c3 = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)   # chord on circle 3 (13, 23)
    heron = 0.25 * math.sqrt(max((c1 + c2 + c3) * (c2 + c3 - c1) * (c1 + c3 - c2)
                                 * (c1 + c2 - c3), 0.0))
    return (heron
            + _segment(r1, c1, x1, y1, ax, ay, bx, by, cx, cy, corrected)
            + _segment(r2, c2, x2, y2, ax, ay, cx, cy, bx, by, corrected)
            + _segment(r3, c3, x3, y3, bx, by, cx, cy, ax, ay, corrected))


@njit(cache=True)
def fewell_batch(X, corrected, out):
    for n in range(X.shape[0]):
        out[n] = fewell_triangle(X[n, 0], X[n, 1], X[n, 2], X[n, 3], X[n, 4], X[n, 5],
                                 X[n, 6], X[n, 7], X[n, 8], corrected)
