"""Shared tools for the paper benchmarks: configuration families, batch
drivers for every method, and a 50-digit reference."""

import math
import os
import sys

import mpmath as mp
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import reuleaux as tc                       # noqa: E402
import kernel_stable as ks                       # noqa: E402
import kernel_vector as kv                       # noqa: E402
from fewell_kernel import fewell_batch           # noqa: E402

mp.mp.dps = 50


# ---------------------------------------------------------------------------
# configuration families; every row: x1 y1 r1 x2 y2 r2 x3 y3 r3
# ---------------------------------------------------------------------------

def fam_general(rng, n):
    X = np.empty((n, 9))
    for k in range(n):
        for c in range(3):
            X[k, 3 * c:3 * c + 3] = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                                     rng.uniform(0.05, 1.2))
    return X


def fam_similar(rng, n):
    """Comparable radii, strong overlap: rich in triangles with reflex arcs."""
    X = np.empty((n, 9))
    for k in range(n):
        r = rng.uniform(0.5, 1.0)
        for c in range(3):
            X[k, 3 * c:3 * c + 3] = (rng.normal(0, 0.35 * r), rng.normal(0, 0.35 * r),
                                     r * rng.uniform(0.85, 1.15))
    return X


def fam_hierarchical(rng, n):
    """Large disk (r = 1), medium disk on its rim, small disk near the medium one
    (star, planet, moon)."""
    X = np.empty((n, 9))
    for k in range(n):
        rp = rng.uniform(0.02, 0.2)
        rm = rng.uniform(0.005, min(0.5 * rp, 0.1))
        ap = rng.uniform(0, 2 * math.pi)
        dp = rng.uniform(0.75, 1.05)
        xp, yp = dp * math.cos(ap), dp * math.sin(ap)
        am = rng.uniform(0, 2 * math.pi)
        dm = rng.uniform(0.0, rp + rm)
        X[k] = (0.0, 0.0, 1.0, xp, yp, rp, xp + dm * math.cos(am), yp + dm * math.sin(am), rm)
    return X


def fam_corner(rng, n, depth_lo=-9.0, depth_hi=-0.3):
    """Tiny circular triangles: the third disk clips a corner of the lens of
    the first two to a depth delta * r_min, delta log-uniform."""
    X = np.empty((n, 9))
    deltas = np.empty(n)
    k = 0
    while k < n:
        r1, r2 = rng.uniform(0.3, 1.0, 2)
        d = rng.uniform(abs(r1 - r2) + 0.1 * min(r1, r2), r1 + r2 - 0.1 * min(r1, r2))
        th = rng.uniform(0, 2 * math.pi)
        x1, y1 = rng.uniform(-1, 1, 2)
        x2, y2 = x1 + d * math.cos(th), y1 + d * math.sin(th)
        a = (d * d + r1 * r1 - r2 * r2) / (2 * d)
        h = math.sqrt(r1 * r1 - a * a)
        ux, uy = math.cos(th), math.sin(th)
        mx, my = x1 + a * ux, y1 + a * uy
        px, py = mx - h * uy, my + h * ux                  # one lens corner
        nx, ny = (mx - px) / h, (my - py) / h              # into the lens
        r3 = rng.uniform(0.3, 1.0)
        delta = 10.0 ** rng.uniform(depth_lo, depth_hi) * min(r1, r2, r3)
        x3, y3 = px - nx * (r3 - delta), py - ny * (r3 - delta)
        X[k] = (x1, y1, r1, x2, y2, r2, x3, y3, r3)
        deltas[k] = delta / min(r1, r2, r3)
        k += 1
    return X, deltas


def fam_sliver_lens(rng, n, depth_lo=-9.0, depth_hi=-0.3):
    """Thin lenses: small disk barely inside the rim of a large one; third
    disk contains the small disk's neighbourhood (result = lens)."""
    X = np.empty((n, 9))
    for k in range(n):
        r1 = 1.0
        r2 = 10.0 ** rng.uniform(-3, -1)
        t = 10.0 ** rng.uniform(depth_lo, depth_hi)
        d = r1 + r2 - t * r2
        th = rng.uniform(0, 2 * math.pi)
        x2, y2 = d * math.cos(th), d * math.sin(th)
        X[k] = (0.0, 0.0, r1, x2, y2, r2, x2, y2, 3.0 * r2)
    return X


# ---------------------------------------------------------------------------
# batch drivers (all called from compiled loops: no interpreter overhead)
# ---------------------------------------------------------------------------

@njit(cache=True)
def shift(X):
    """Translate each row to the centre of its smallest circle."""
    Y = X.copy()
    for n in range(X.shape[0]):
        k = 0
        if X[n, 5] < X[n, 3 * k + 2]:
            k = 1
        if X[n, 8] < X[n, 3 * k + 2]:
            k = 2
        x0 = X[n, 3 * k]
        y0 = X[n, 3 * k + 1]
        for c in range(3):
            Y[n, 3 * c] -= x0
            Y[n, 3 * c + 1] -= y0
    return Y


@njit(cache=True)
def batch_stable(X, out):
    for n in range(X.shape[0]):
        out[n] = ks.area3_fast_core(X[n, 0], X[n, 1], X[n, 2], X[n, 3], X[n, 4], X[n, 5],
                                    X[n, 6], X[n, 7], X[n, 8])[0]


@njit(cache=True)
def batch_vector(X, out):
    for n in range(X.shape[0]):
        out[n] = kv.area3_fast_core(X[n, 0], X[n, 1], X[n, 2], X[n, 3], X[n, 4], X[n, 5],
                                    X[n, 6], X[n, 7], X[n, 8])[0]


@njit(cache=True)
def batch_case(X, case, nv):
    for n in range(X.shape[0]):
        r = ks.area3_fast_core(X[n, 0], X[n, 1], X[n, 2], X[n, 3], X[n, 4], X[n, 5],
                               X[n, 6], X[n, 7], X[n, 8])
        case[n] = r[1]
        nv[n] = r[2]


def batch_generic(X, out):
    case = np.empty(X.shape[0], dtype=np.int64)
    tc._area_batch(*[np.ascontiguousarray(X[:, j]) for j in range(9)], out, case)


@njit(cache=True)
def _lens_acos(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    t1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    t2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)
    tri = (-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1)
    return (r1 * r1 * math.acos(min(1.0, max(-1.0, t1)))
            + r2 * r2 * math.acos(min(1.0, max(-1.0, t2))) - 0.5 * math.sqrt(max(tri, 0.0)))


def run(method, X):
    out = np.empty(X.shape[0])
    if method == "stable":
        batch_stable(X, out)
    elif method == "vector":
        batch_vector(X, out)
    elif method == "generic":
        batch_generic(X, out)
    elif method == "fewell":
        fewell_batch(X, True, out)
    elif method == "fewell_eq16":
        fewell_batch(X, False, out)
    else:
        raise ValueError(method)
    return out


# ---------------------------------------------------------------------------
# 50-digit reference: topology from the double-precision analysis (checked
# independently by the chord quadrature), every quantity recomputed in mpmath
# ---------------------------------------------------------------------------

def _mp_crossings(c1, c2):
    (x1, y1, r1), (x2, y2, r2) = [[mp.mpf(v) for v in c] for c in (c1, c2)]
    dx, dy = x2 - x1, y2 - y1
    d = mp.sqrt(dx * dx + dy * dy)
    a = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    h = mp.sqrt(r1 * r1 - a * a)
    ux, uy = dx / d, dy / d
    return [(x1 + a * ux - h * uy, y1 + a * uy + h * ux),
            (x1 + a * ux + h * uy, y1 + a * uy - h * ux)]


def _mp_lens(r1, r2, d):
    r1, r2, d = mp.mpf(r1), mp.mpf(r2), mp.mpf(d)
    return (r1 ** 2 * mp.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
            + r2 ** 2 * mp.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
            - mp.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)) / 2)


def _mp_eval(cs, ov):
    """Area in mpmath for circles cs (mpf triples) with the topology of ov."""
    if ov.case == "empty":
        return mp.mpf(0)
    if ov.case == "disk":
        return mp.pi * cs[ov.arcs[0].circle][2] ** 2
    if ov.case == "lens":
        if len(ov.arcs) >= 2:
            pairs = [(ov.arcs[0].circle, ov.arcs[1].circle)]
        else:                      # near-tangent lens: the pair that matches
            pairs = [(0, 1), (0, 2), (1, 2)]
        best = None
        for a, b in pairs:
            (xa, ya, ra), (xb, yb, rb) = cs[a], cs[b]
            d = mp.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            if d >= ra + rb or d <= abs(ra - rb):
                continue
            v = _mp_lens(ra, rb, d)
            if best is None or abs(v - ov.area) < abs(best - ov.area):
                best = v
        return best
    verts = {}
    for u, (i, j) in enumerate(ov.vertex_pairs):
        cand = _mp_crossings(cs[i], cs[j])
        fx, fy = ov.vertices[u]
        verts[(float(fx), float(fy))] = min(cand, key=lambda p: (p[0] - fx) ** 2 + (p[1] - fy) ** 2)
    area = mp.mpf(0)
    pts = []
    for arc in ov.arcs:
        U = verts[(float(arc.ux), float(arc.uy))]
        V = verts[(float(arc.vx), float(arc.vy))]
        x, y, r = cs[arc.circle]
        ux, uy, vx, vy = U[0] - x, U[1] - y, V[0] - x, V[1] - y
        dth = mp.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        if dth <= 0:
            dth += 2 * mp.pi
        area += r * r / 2 * (dth - mp.sin(dth))
        pts.append(U)
    p0 = pts[0]
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        area += ((a[0] - p0[0]) * (b[1] - p0[1]) - (a[1] - p0[1]) * (b[0] - p0[0])) / 2
    return area


def mp_area(row, condition=False):
    """50-digit area of the configuration `row` (None if degenerate). With
    condition=True also the componentwise relative condition number
    kappa = sum_i |p_i dA/dp_i| / A over the nine inputs."""
    cs_f = [tuple(row[3 * c:3 * c + 3]) for c in range(3)]
    ov = tc.overlap(*cs_f)
    if ov.perturbed:
        return (None, None) if condition else None
    cs = [[mp.mpf(float(v)) for v in c] for c in cs_f]
    A = _mp_eval(cs, ov)
    if not condition:
        return A
    if A is None or A == 0:
        return A, None
    kap = mp.mpf(0)
    for c in range(3):
        for q in range(3):
            p = cs[c][q]
            if p == 0:
                continue
            h = abs(p) * mp.mpf(10) ** -30
            cp = [list(v) for v in cs]
            cm = [list(v) for v in cs]
            cp[c][q] = p + h
            cm[c][q] = p - h
            dA = (_mp_eval(cp, ov) - _mp_eval(cm, ov)) / (2 * h)
            kap += abs(p * dA)
    return A, kap / A


def textbook_lens_for(row):
    """Textbook acos lens of the surviving pair for lens results (else NaN)."""
    cs = [tuple(row[3 * c:3 * c + 3]) for c in range(3)]
    ov = tc.overlap(*cs)
    if ov.case != "lens":
        return math.nan
    if len(ov.arcs) < 2:
        best = math.nan
        for a, b in ((0, 1), (0, 2), (1, 2)):
            (xa, ya, ra), (xb, yb, rb) = cs[a], cs[b]
            v = _lens_acos(ra, rb, math.hypot(xa - xb, ya - yb))
            if 0 < v < math.pi * min(ra, rb) ** 2 and (best != best or abs(v - ov.area) < abs(best - ov.area)):
                best = v
        return best
    a, b = ov.arcs[0].circle, ov.arcs[1].circle
    (xa, ya, ra), (xb, yb, rb) = cs[a], cs[b]
    return _lens_acos(ra, rb, math.hypot(xa - xb, ya - yb))


@njit(cache=True)
def batch_stable_shift(X, out):
    """Stable specialised kernel including the translation to the smallest centre."""
    for n in range(X.shape[0]):
        k = 0
        if X[n, 5] < X[n, 2]:
            k = 1
        if X[n, 8] < X[n, 3 * k + 2]:
            k = 2
        x0 = X[n, 3 * k]
        y0 = X[n, 3 * k + 1]
        out[n] = ks.area3_fast_core(X[n, 0] - x0, X[n, 1] - y0, X[n, 2], X[n, 3] - x0,
                                    X[n, 4] - y0, X[n, 5], X[n, 6] - x0, X[n, 7] - y0,
                                    X[n, 8])[0]


@njit(cache=True)
def batch_vector_shift(X, out):
    for n in range(X.shape[0]):
        k = 0
        if X[n, 5] < X[n, 2]:
            k = 1
        if X[n, 8] < X[n, 3 * k + 2]:
            k = 2
        x0 = X[n, 3 * k]
        y0 = X[n, 3 * k + 1]
        out[n] = kv.area3_fast_core(X[n, 0] - x0, X[n, 1] - y0, X[n, 2], X[n, 3] - x0,
                                    X[n, 4] - y0, X[n, 5], X[n, 6] - x0, X[n, 7] - y0,
                                    X[n, 8])[0]
