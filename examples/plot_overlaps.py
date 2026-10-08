#!/usr/bin/env python3
"""Vector-PDF figures of three-circle overlaps (uses reuleaux).

    python plot_overlaps.py                       # all figures into ./figures/
    python plot_overlaps.py --out DIR             # other output directory
    python plot_overlaps.py --only walk reflex    # a subset
    python plot_overlaps.py --circles x1 y1 r1 x2 y2 r2 x3 y3 r3 [--pdf my.pdf]

Figures (all vector PDF; nothing is rasterised):

    fig_examples.pdf    the 20 curated Pandora geometries, 6 per page
    fig_taxonomy.pdf    every topological case: empty, disk, lens, triangle,
                        reflex triangle, quadrilateral
    fig_walk.pdf        the method: vertices, directed arcs, polygon + segments
    fig_reflex.pdf      the Fewell/Kipping reflex error, geometry and sweep
    fig_raster.pdf      Pandora's legacy pixel raster against the exact area
    fig_precision.pdf   thin slivers: textbook acos lens vs. the stable form
"""

from __future__ import annotations

import argparse
import math
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages    # noqa: E402
from matplotlib.collections import PatchCollection      # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

import reuleaux as tc                          # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType: text stays text
    "font.size": 9,
    "axes.titlesize": 9.5,
    "savefig.bbox": "tight",
})

CIRCLE_STYLE = [
    dict(fc="#fff3c4", ec="#c99a00"),    # circle 0 (star in Pandora convention)
    dict(fc="#d9d9d9", ec="#555555"),    # circle 1 (planet)
    dict(fc="#cfe3ff", ec="#1f5fbf"),    # circle 2 (moon)
]
ARC_COLOURS = ["#c99a00", "#333333", "#1f5fbf"]
REGION = dict(fc="#e4572e", ec="none", alpha=0.55)


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------

def _arrow_on_arc(ax, arc, colour, frac=0.5, size=9):
    t = arc.theta0 + frac * arc.dtheta
    x = arc.cx + arc.r * math.cos(t)
    y = arc.cy + arc.r * math.sin(t)
    tx, ty = -math.sin(t), math.cos(t)               # CCW tangent
    L = 1e-3 * arc.r
    ax.add_patch(FancyArrowPatch((x - L * tx, y - L * ty), (x + L * tx, y + L * ty),
                                 arrowstyle="-|>", mutation_scale=size, color=colour,
                                 lw=0, zorder=6))


def draw_overlap(ax, c1, c2, c3, names=("circle 0", "circle 1", "circle 2"),
                 zoom="all", arrows=True, vertices=True, legend=False, text=True,
                 margin=0.25):
    """Draw three circles, their common overlap and its boundary arcs.

    zoom: "all" (all circles), "smallest" (the smallest circle) or a tuple
    (xmin, xmax, ymin, ymax). Returns the Overlap."""
    cs = (c1, c2, c3)
    ov = tc.overlap(*cs)
    for k, (x, y, r) in enumerate(cs):
        st = CIRCLE_STYLE[k]
        ax.add_patch(Circle((x, y), r, fc=st["fc"], ec="none", alpha=0.55, zorder=1))
    for k, (x, y, r) in enumerate(cs):
        st = CIRCLE_STYLE[k]
        ax.add_patch(Circle((x, y), r, fc="none", ec=st["ec"], lw=0.9, zorder=3,
                            label=names[k]))
    if ov.area > 0 and ov.arcs:
        ax.add_patch(Polygon(ov.boundary(400), closed=True, zorder=4, **REGION,
                             label="common overlap"))
        for arc in ov.arcs:
            pts = arc.points(400)
            long_arc = arc.reflex and arc.dtheta < tc.TWO_PI
            ax.plot(pts[:, 0], pts[:, 1], color=ARC_COLOURS[arc.circle],
                    lw=2.4 if not long_arc else 3.0,
                    ls="-" if not long_arc else (0, (4, 1.5)), zorder=5,
                    solid_capstyle="butt")
            if arrows and arc.dtheta < tc.TWO_PI:
                _arrow_on_arc(ax, arc, ARC_COLOURS[arc.circle])
    if vertices and len(ov.vertices):
        ax.plot(ov.vertices[:, 0], ov.vertices[:, 1], "o", ms=3.5, mfc="k", mec="w",
                mew=0.6, zorder=7)
    if zoom == "all":
        xs = [x - r for x, y, r in cs] + [x + r for x, y, r in cs]
        ys = [y - r for x, y, r in cs] + [y + r for x, y, r in cs]
        box = (min(xs), max(xs), min(ys), max(ys))
    elif zoom == "smallest":
        x, y, r = min(cs, key=lambda c: c[2])
        box = (x - r, x + r, y - r, y + r)
    else:
        box = zoom
    w = box[1] - box[0]
    h = box[3] - box[2]
    m = margin * max(w, h)
    cxm, cym = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    half = 0.5 * max(w, h) + m
    ax.set_xlim(cxm - half, cxm + half)
    ax.set_ylim(cym - half, cym + half)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if text:
        rmin = min(c[2] for c in cs)
        s = f"{ov.case}, V = {ov.n_vertices}"
        if ov.reflex:
            s += ", reflex arc"
        s += f"\nA = {ov.area:.6g} = {ov.area / (math.pi * rmin ** 2):.4f} " + r"$\pi r_\mathrm{min}^2$"
        ax.text(0.02, 0.02, s, transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.9), zorder=10)
    if legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    return ov


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_examples(path):
    exs = tc.EXAMPLES
    with PdfPages(path) as pdf:
        for p in range(0, len(exs), 6):
            fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.8))
            for ax, ex in zip(axes.flat, exs[p:p + 6]):
                cs = tc.example_circles(ex)
                zoom = "smallest" if ex["rm"] < 0.2 else "all"
                draw_overlap(ax, *cs, names=("star", "planet", "moon"), zoom=zoom,
                             margin=0.35 if zoom == "smallest" else 0.05)
                er = tc.pandora_er(ex["xp"], ex["yp"], ex["rp"], ex["xm"], ex["ym"], ex["rm"])
                e25 = tc.pixelart_er(ex["xp"], ex["yp"], ex["xm"], ex["ym"], ex["rp"], ex["rm"], 25)
                ax.set_title(ex["name"] + "\n" + "\n".join(textwrap.wrap(ex["desc"], 52)),
                             fontsize=7)
                ax.text(0.98, 0.98, f"er exact {er:.5f}\ner raster(25) {e25:.5f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                        bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.9), zorder=10)
            for ax in list(axes.flat)[len(exs[p:p + 6]):]:
                ax.axis("off")
            fig.suptitle("Curated examples (Pandora convention: star (0, 0, 1), planet, moon; "
                         "zoomed on the moon). Red: star ∩ planet ∩ moon; dashed arc: reflex (> π)",
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)


def fig_taxonomy(path):
    s3 = math.sqrt(3) / 2
    cases = [
        ("empty: all pairs intersect, no common point (Fewell case i)",
         ((0, 0, 1.0), (1.9, 0, 1.0), (0.95, 1.9 * s3, 1.0))),
        ("disk: smallest disk inside the other two", ((0, 0, 1.0), (0.5, 0.2, 0.9), (0.35, 0.1, 0.35))),
        ("lens: two-circle lens inside the third disk", ((0, 0, 1.2), (-0.35, 0, 0.6), (0.35, 0, 0.6))),
        ("circular triangle (V = 3)", ((0, 0, 1.0), (1.0, 0, 1.0), (0.5, s3, 1.0))),
        ("reflex circular triangle: one arc > π", tc.example_circles(
            next(e for e in tc.EXAMPLES if e["name"] == "reflex-deep-2"))),
        ("circular quadrilateral (V = 4, Fewell case d)", ((0, 0, 1.0), (1.3, 0, 1.0), (0.65, 0, 0.75))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for ax, (title, cs) in zip(axes.flat, cases):
        draw_overlap(ax, *cs, margin=0.05)
        ax.set_title(title, fontsize=8)
    fig.suptitle("Topological cases of D₀ ∩ D₁ ∩ D₂ (V = number of vertices = corners of the region)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def fig_walk(path):
    cs = ((0.0, 0.0, 1.0), (0.8, 0.3, 0.6), (0.5, -0.4, 0.7))
    ov = tc.overlap(*cs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))

    # (a) vertices, crossings outside the third disk, directed arcs
    ax = axes[0]
    draw_overlap(ax, *cs, text=False, margin=0.05)
    for (i, j, k) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        n, px, py, qx, qy = tc._crossings(cs[i][0], cs[i][1], cs[i][2],
                                          cs[j][0], cs[j][1], cs[j][2], 1.0)
        for t, (x, y) in enumerate(((px, py), (qx, qy))[:n]):
            inside = (x - cs[k][0]) ** 2 + (y - cs[k][1]) ** 2 < cs[k][2] ** 2
            if not inside:
                ax.plot(x, y, "x", color="0.35", ms=5, mew=1.0, zorder=7)
                ax.annotate(f"{i}∩{j}, outside D{k}", (x, y), textcoords="offset points",
                            xytext=(4, -10), fontsize=6.5, color="0.35")
    for u, (x, y) in enumerate(ov.vertices):
        i, j = ov.vertex_pairs[u]
        ax.annotate(f"v{u} = {i}∩{j}", (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, weight="bold")
    for n_arc, arc in enumerate(ov.arcs):
        t = arc.theta0 + 0.5 * arc.dtheta
        ax.annotate(f"arc {n_arc} on circle {arc.circle}\nΔθ = {arc.dtheta:.3f}",
                    (arc.cx + arc.r * math.cos(t), arc.cy + arc.r * math.sin(t)),
                    textcoords="offset points", xytext=(30 * math.cos(t), 30 * math.sin(t)),
                    fontsize=7, color=ARC_COLOURS[arc.circle], va="center",
                    arrowprops=dict(arrowstyle="-", color=ARC_COLOURS[arc.circle], lw=0.5),
                    ha="left" if math.cos(t) > 0 else "right")
    ax.set_title("(a) vertices = crossings strictly inside the third disk (dots);\n"
                 "crosses: crossings outside it. Each vertex has exactly one CCW out-arc.",
                 fontsize=8.5)

    # (b) polygon + segments decomposition
    ax = axes[1]
    for k, (x, y, r) in enumerate(cs):
        ax.add_patch(Circle((x, y), r, fc="none", ec=CIRCLE_STYLE[k]["ec"], lw=0.7, ls=":"))
    starts = np.array([[a.ux, a.uy] for a in ov.arcs])
    ax.add_patch(Polygon(starts, closed=True, fc="#9bc53d", ec="k", lw=0.8, alpha=0.7,
                         label="polygon of the vertices (shoelace)"))
    for n_arc, arc in enumerate(ov.arcs):
        pts = arc.points(300)
        ax.add_patch(Polygon(pts, closed=True, fc=ARC_COLOURS[arc.circle], ec="none", alpha=0.35,
                             label=f"segment on circle {arc.circle}: "
                                   + r"$\frac{r^2}{2}(\Delta\theta-\sin\Delta\theta)$"
                                   + f" = {arc.segment_area:.4f}"))
        ax.plot(pts[:, 0], pts[:, 1], color=ARC_COLOURS[arc.circle], lw=2)
        _arrow_on_arc(ax, arc, ARC_COLOURS[arc.circle])
    xc, yc = ov.centroid
    ax.plot(xc, yc, "+", color="k", ms=9, mew=1.5, label="centroid")
    ax.set_aspect("equal")
    lo = ov.vertices.min(axis=0) - 0.25
    hi = ov.vertices.max(axis=0) + 0.25
    half = 0.5 * max(hi - lo)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.95)
    ax.set_title("(b) Green's theorem per arc = segment + triangle; summed:\n"
                 f"A = Σ segments + polygon = {ov.area:.12f}", fontsize=8.5)
    fig.text(0.5, 0.005,
             r"Per arc (centre o, from u to v, CCW): $\frac{1}{2}\left[r^2\Delta\theta"
             r" + o_x (v_y-u_y) - o_y (v_x-u_x)\right]$, $\Delta\theta$ = atan2(u'×v', u'·v') ∈ (0, 2π]: "
             "no case distinction for arcs longer than π.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _fewell_parts(ov):
    """Chords, reflex flags and the opposite vertex for every circle of a triangle."""
    out = []
    for k in range(3):
        on = [ov.vertices[u] for u in range(3) if k in ov.vertex_pairs[u]]
        off = [ov.vertices[u] for u in range(3) if k not in ov.vertex_pairs[u]][0]
        p0, p1 = on
        o = np.array(ov.circles[k][:2])
        nvec = np.array([p1[1] - p0[1], p0[0] - p1[0]])
        reflex = np.dot(o - p0, nvec) * np.dot(off - p0, nvec) < 0
        out.append((p0, p1, reflex))
    return out


def fig_reflex(path):
    ex = next(e for e in tc.EXAMPLES if e["name"] == "reflex-deep-1")
    cs = tc.example_circles(ex)
    ov = tc.overlap(*cs)
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25])

    # (a) correct decomposition
    for col, corrected in enumerate((True, False)):
        ax = fig.add_subplot(gs[0, col])
        for k, (x, y, r) in enumerate(cs):
            ax.add_patch(Circle((x, y), r, fc="none", ec=CIRCLE_STYLE[k]["ec"], lw=0.8))
        ax.add_patch(Polygon(ov.vertices, closed=True, fc="#9bc53d", ec="k", lw=0.6, alpha=0.6,
                             label="Heron triangle on the chords"))
        for k, (p0, p1, reflex) in enumerate(_fewell_parts(ov)):
            x, y, r = cs[k]
            a0 = math.atan2(p0[1] - y, p0[0] - x)
            a1 = math.atan2(p1[1] - y, p1[0] - x)
            d = (a1 - a0) % (2 * math.pi)
            # the segment's arc is the one on the far side from the triangle
            arc_major = reflex
            if (d > math.pi) != arc_major:
                a0, a1 = a1, a0
                d = (a1 - a0) % (2 * math.pi)
            if reflex and not corrected:
                # Fewell Eq. 16 effectively uses the MINOR segment's angle
                a0, a1 = a1, a0
                d = (a1 - a0) % (2 * math.pi)
            t = np.linspace(a0, a0 + d, 300)
            pts = np.column_stack((x + r * np.cos(t), y + r * np.sin(t)))
            fc = ARC_COLOURS[k] if not reflex else ("#e4572e" if corrected else "#7b2cbf")
            lab = f"segment on circle {k}"
            if reflex:
                lab += " (reflex: major segment)" if corrected else " (Eq. 16: minor angle used)"
            ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", alpha=0.45, label=lab))
        ax.set_aspect("equal")
        ax.set_xlim(-0.45, 0.8)
        ax.set_ylim(-1.2, 0.05)
        ax.legend(loc="lower left", fontsize=6.3, framealpha=0.95)
        a = tc.fewell_area(*cs, corrected=corrected)
        moon = math.pi * ex["rm"] ** 2
        if corrected:
            ax.set_title("(a) Gordon & Agol (2022) App. B: reflex segment\n"
                         rf"$r^2(\pi-\arcsin\frac{{c}}{{2r}}) + \frac{{c}}{{4}}\sqrt{{4r^2-c^2}}$"
                         f" → A = {a:.6f} (exact)", fontsize=8)
        else:
            ax.set_title("(b) Fewell (2006) Eq. 16 / Kipping (2011) Eq. 37:\n"
                         rf"$r^2\arcsin\frac{{c}}{{2r}} + \frac{{c}}{{4}}\sqrt{{4r^2-c^2}}$"
                         f" → A = {a:.6f} ({100 * (a - ov.area) / moon:+.1f} % of the moon)",
                         fontsize=8)

    # (c) sweep: move the moon along y, the moon arc passes pi
    ax = fig.add_subplot(gs[0, 2])
    ys = np.linspace(-0.80, -0.40, 801)
    ex_a, fw_ok, fw_bad, refl = [], [], [], []
    for y in ys:
        c = (cs[0], cs[1], (ex["xm"], y, ex["rm"]))
        o = tc.overlap(*c)
        ex_a.append(o.area)
        if o.case == "triangle":
            fw_ok.append(tc.fewell_area(*c))
            fw_bad.append(tc.fewell_area(*c, corrected=False))
        else:
            fw_ok.append(np.nan)
            fw_bad.append(np.nan)
        refl.append(o.reflex)
    ex_a, fw_ok, fw_bad, refl = map(np.array, (ex_a, fw_ok, fw_bad, refl))
    moon = math.pi * ex["rm"] ** 2
    ax.fill_between(ys, 0, 1.05, where=refl, color="#e4572e", alpha=0.12, lw=0,
                    transform=ax.get_xaxis_transform(), label="reflex arc present")
    ax.plot(ys, ex_a / moon, color="k", lw=1.8, label="exact (arc walk)")
    ax.plot(ys, fw_ok / moon, color="#9bc53d", lw=1.0, ls=(0, (5, 3)),
            label="Fewell Eq. 1 + G&A fix")
    ax.plot(ys, fw_bad / moon, color="#7b2cbf", lw=1.4, label="Fewell Eq. 16 / Kipping Eq. 37")
    ax.axvline(ex["ym"], color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("moon centre y (x, radii fixed as in panels a, b)")
    ax.set_ylabel(r"$A\,/\,(\pi r_\mathrm{moon}^2)$")
    ax.set_title("(c) sweep: the historical formula fails exactly\nwhen an arc exceeds π",
                 fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_raster(path):
    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    # (a, b) the raster of one limb case at n = 25
    xp, yp, rp, xm, ym, rm = 0.96, 0.03, 0.10, 1.00, -0.02, 0.05
    for col, n in enumerate((25, 99)):
        ax = fig.add_subplot(gs[0, col])
        img, ntrip, frac, cci = tc.pixelart_counts(xp, yp, xm, ym, rp, rm, n)
        ng = n if n % 2 else n + 1
        size = 2.0 * rm / ng
        rects, cols = [], []
        # Pandora's colour codes (star 5, moon 3, planet 2) are ambiguous
        # (5 = star alone = moon + planet); only the sum 10 is counted, so the
        # moon mask is recomputed with pixelart's own moon test for drawing
        aa = math.sqrt((ng + 1) ** 2 - ng ** 2) / 2
        for i in range(ng + 1):
            for j in range(ng + 1):
                ii = i if i < (ng + 1) // 2 else ng - i
                jj = j if j < (ng + 1) // 2 else ng - j
                if not (ng - 2 * ii) ** 2 + (ng - 2 * jj) ** 2 < ng ** 2 + aa:
                    continue                      # outside the moon raster
                code = int(img[i, j])
                x = xm + (2 * i - ng) * rm / ng
                y = ym + (2 * j - ng) * rm / ng
                rects.append(Rectangle((x - size / 2, y - size / 2), size, size))
                cols.append("#e4572e" if code == 10 else ("#ffd166" if code == 8 else "#cfe3ff"))
        ax.add_collection(PatchCollection(rects, facecolors=cols, edgecolors="white",
                                          linewidths=0.15 if n > 50 else 0.4))
        ov = tc.overlap((0, 0, 1), (xp, yp, rp), (xm, ym, rm))
        for k, (x, y, r) in enumerate(((0, 0, 1), (xp, yp, rp), (xm, ym, rm))):
            ax.add_patch(Circle((x, y), r, fc="none", ec=CIRCLE_STYLE[k]["ec"], lw=1.0, zorder=4))
        b = ov.boundary(400)
        ax.plot(np.append(b[:, 0], b[0, 0]), np.append(b[:, 1], b[0, 1]), color="k", lw=1.4,
                zorder=5, label="exact star ∩ planet ∩ moon")
        ax.set_xlim(xm - 1.25 * rm, xm + 1.25 * rm)
        ax.set_ylim(ym - 1.25 * rm, ym + 1.25 * rm)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        exact = ov.area / (math.pi * rm ** 2)
        ax.set_title(f"({'ab'[col]}) Pandora pixelart, numerical_grid = {n}: "
                     f"{ntrip} triple pixels\nA/(π r_m²): raster {frac:.4f}, exact {exact:.4f}; "
                     f"er: raster {tc.pixelart_er(xp, yp, xm, ym, rp, rm, n):.4f}, "
                     f"exact {tc.pandora_er(xp, yp, rp, xm, ym, rm):.4f}", fontsize=7.5)
        if col == 0:
            ax.legend(loc="lower left", fontsize=6.5)

    # (c) convergence with the grid
    ax = fig.add_subplot(gs[0, 2])
    grids = [9, 15, 25, 35, 51, 75, 99, 151, 199, 299, 399]
    for ex in tc.EXAMPLES:
        if ex["name"] not in ("realistic-tri-1", "realistic-tri-2", "realistic-quad",
                              "taxonomy-tri", "reflex-nearpi", "ingress-lens"):
            continue
        e = tc.pandora_er(ex["xp"], ex["yp"], ex["rp"], ex["xm"], ex["ym"], ex["rm"])
        err = [abs(tc.pixelart_er(ex["xp"], ex["yp"], ex["xm"], ex["ym"], ex["rp"], ex["rm"], g) - e)
               for g in grids]
        ax.loglog(grids, np.maximum(err, 1e-7), "o-", ms=3, lw=0.9, label=ex["name"])
    ax.loglog(grids, 1.0 / np.array(grids, float), "k:", lw=0.8, label="1 / n")
    ax.axvline(25, color="0.5", lw=0.6)
    ax.set_xlabel("numerical_grid n")
    ax.set_ylabel("|er raster − er exact|")
    ax.set_title("(c) raster error vs. grid (default n = 25)", fontsize=8.5)
    ax.legend(fontsize=6.3)

    # (d, e) sweep: the moon passes behind the planet while crossing the limb
    xp, yp, rp, rm, ymoon = 1.0, 0.0, 0.10, 0.04, 0.075
    xs = np.linspace(0.84, 1.16, 1601)
    e_ex = np.array([tc.pandora_er(xp, yp, rp, x, ymoon, rm) for x in xs])
    e25 = np.array([tc.pixelart_er(xp, yp, x, ymoon, rp, rm, 25) for x in xs])
    e99 = np.array([tc.pixelart_er(xp, yp, x, ymoon, rp, rm, 99) for x in xs])
    zm = np.hypot(xs, ymoon)
    limb = np.abs(zm - 1.0) < rm
    off = zm >= 1.0 + rm
    for col, (title, curves) in enumerate((
            ("(d) eclipse ratio along the track", ((e25, "#7b2cbf", "raster n = 25 (old Pandora default)"),
                                                  (e99, "#1f5fbf", "raster n = 99"),
                                                  (e_ex, "k", "exact"))),
            ("(e) raster − exact", ((e25 - e_ex, "#7b2cbf", "n = 25"), (e99 - e_ex, "#1f5fbf", "n = 99"))))):
        ax = fig.add_subplot(gs[1, 0:2] if col == 0 else gs[1, 2])
        ax.fill_between(xs, 0, 1, where=limb, color="#ffd166", alpha=0.25, lw=0,
                        transform=ax.get_xaxis_transform(), label="moon on the stellar limb")
        ax.fill_between(xs, 0, 1, where=off, color="0.85", alpha=0.4, lw=0,
                        transform=ax.get_xaxis_transform(), label="moon off the star (er := 1)")
        for y, c, lab in curves:
            ax.plot(xs, y, color=c, lw=1.5 if c == "k" else 0.8, label=lab)
        ax.set_xlabel(f"moon centre x  (y = {ymoon}, r_m = {rm}; planet ({xp}, {yp}), r_p = {rp}; star radius 1)"
                      if col == 0 else "moon centre x")
        ax.set_ylabel("er = A(S∩P∩M) / A(S∩M)" if col == 0 else "er difference")
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _lens_acos(r1, r2, d):
    """The textbook lens formula (MathWorld; Pandora's cci())."""
    t1 = (d * d + r1 * r1 - r2 * r2) / (2 * d * r1)
    t2 = (d * d + r2 * r2 - r1 * r1) / (2 * d * r2)
    tri = (-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1)
    return (r1 * r1 * math.acos(min(1, max(-1, t1))) + r2 * r2 * math.acos(min(1, max(-1, t2)))
            - 0.5 * math.sqrt(max(tri, 0.0)))


def fig_precision(path):
    try:
        import mpmath as mp
    except ImportError:
        print("  (fig_precision skipped: mpmath not installed)")
        return
    mp.mp.dps = 60

    def lens_mp(r1, r2, d):
        r1, r2, d = mp.mpf(r1), mp.mpf(r2), mp.mpf(d)
        return (r1 ** 2 * mp.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
                + r2 ** 2 * mp.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
                - mp.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(3)
    for ax, (title, r1) in zip(axes, (("star (r = 1) ∩ moon, moon just inside the limb", 1.0),
                                      ("planet (r = 0.1) ∩ moon, grazing overlap", 0.1))):
        depth = 10.0 ** rng.uniform(-9, 0, 600)
        r2 = 0.01
        e_acos, e_new = [], []
        for t in depth:
            d = r1 + r2 - t * r2
            ex = lens_mp(r1, r2, d)
            if ex <= 0:
                e_acos.append(np.nan)
                e_new.append(np.nan)
                continue
            e_acos.append(abs(_lens_acos(r1, r2, d) - float(ex)) / float(ex))
            e_new.append(abs(tc.lens_area(r1, r2, d) - float(ex)) / float(ex))
        ax.loglog(depth, np.maximum(e_acos, 1e-17), ".", ms=2.5, color="#7b2cbf",
                  label="textbook acos form (MathWorld, Pandora cci)")
        ax.loglog(depth, np.maximum(e_new, 1e-17), ".", ms=2.5, color="#2a9d8f",
                  label="segments with atan2 + Kahan Heron (this code)")
        ax.axhline(2.2e-16, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel("overlap depth (r₁ + r₂ − d) / r₂")
        ax.set_ylabel("relative error of the lens area (vs. mpmath, 60 digits)")
        ax.set_title(title + f", r_moon = {r2}", fontsize=8.5)
        ax.set_ylim(1e-17, 1e3)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


FIGURES = {
    "examples": ("fig_examples.pdf", fig_examples),
    "taxonomy": ("fig_taxonomy.pdf", fig_taxonomy),
    "walk": ("fig_walk.pdf", fig_walk),
    "reflex": ("fig_reflex.pdf", fig_reflex),
    "raster": ("fig_raster.pdf", fig_raster),
    "precision": ("fig_precision.pdf", fig_precision),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="figures", help="output directory")
    p.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="subset of figures")
    p.add_argument("--circles", type=float, nargs=9, metavar="V",
                   help="plot one configuration: x1 y1 r1 x2 y2 r2 x3 y3 r3")
    p.add_argument("--pdf", default="overlap.pdf", help="file name for --circles")
    args = p.parse_args(argv)
    if args.circles:
        v = args.circles
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ov = draw_overlap(ax, v[0:3], v[3:6], v[6:9], legend=True, margin=0.05)
        ax.set_title("common overlap of three circles")
        fig.savefig(args.pdf)
        plt.close(fig)
        print(ov)
        print("wrote", args.pdf)
        return 0
    os.makedirs(args.out, exist_ok=True)
    for key in (args.only or FIGURES):
        name, fn = FIGURES[key]
        out = os.path.join(args.out, name)
        fn(out)
        if os.path.exists(out):
            print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
