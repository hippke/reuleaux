"""Figures of the paper -> ../figures/*.pdf (vector).

All text is typeset by LaTeX in Computer Modern at 10 pt, the font and size
of the paper body (aastex701, twocolumn). Figure widths equal the
LaTeX \\textwidth (513.11743 pt) for figure* and \\columnwidth
(242.26653 pt) for figure, so the PDFs are included at scale 1.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
import reuleaux as tc                          # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "examples"))
import plot_overlaps as po                          # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)

PT = 1.0 / 72.27                       # TeX point in inches
TEXTWIDTH = 513.11743 * PT             # 7.100 in
COLUMNWIDTH = 242.26653 * PT           # 3.352 in
FS = 10                                # body font size of the paper

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "savefig.bbox": "standard",        # keep the exact figure size
    "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
U = 2.0 ** -53
COL = {"stable": "#1b7f5b", "vector": "#d1495b", "fewell": "#edae49", "fewell_eq16": "#7b2cbf",
       "lens_textbook": "#00798c"}
LAB = {"stable": "this work", "vector": "Eq.~(2), textbook crossings",
       "fewell": r"Fewell + Gordon \& Agol", "fewell_eq16": "Fewell Eq.~16 / Kipping Eq.~37",
       "lens_textbook": "textbook lens, Eq.~(1)"}


def new_fig(width, height):
    fig = plt.figure(figsize=(width, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 * PT, h_pad=2 * PT, wspace=0.02, hspace=0.02)
    return fig


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def taxonomy():
    s3 = math.sqrt(3) / 2
    cases = [
        ("(a) empty", ((0, 0, 1.0), (1.9, 0, 1.0), (0.95, 1.9 * s3, 1.0))),
        ("(b) disk", ((0, 0, 1.0), (0.5, 0.2, 0.9), (0.35, 0.1, 0.35))),
        ("(c) lens", ((0, 0, 1.2), (-0.35, 0, 0.6), (0.35, 0, 0.6))),
        ("(d) triangle", ((0, 0, 1.0), (1.0, 0, 1.0), (0.5, s3, 1.0))),
        ("(e) reflex arc", ((0, 0, 1.0), (0.1998, -0.5836, 0.5443), (0.1995, -0.6157, 0.5280))),
        ("(f) quadrilateral", ((0, 0, 1.0), (1.3, 0, 1.0), (0.65, 0, 0.75))),
    ]
    fig = new_fig(TEXTWIDTH, 1.42)
    axes = fig.subplots(1, 6)
    for ax, (title, cs) in zip(axes, cases):
        po.draw_overlap(ax, *cs, margin=0.04, text=False, arrows=False)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "taxonomy.pdf")


def method():
    cs = ((0.0, 0.0, 1.0), (0.8, 0.3, 0.6), (0.5, -0.4, 0.7))
    ov = tc.overlap(*cs)
    fig = new_fig(TEXTWIDTH, 2.75)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62])
    ax = fig.add_subplot(gs[0, 0])
    po.draw_overlap(ax, *cs, text=False, margin=0.03)
    for (i, j, k) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        n, px, py, qx, qy = tc._crossings(*cs[i], *cs[j], 1.0)
        for x, y in ((px, py), (qx, qy))[:n]:
            if (x - cs[k][0]) ** 2 + (y - cs[k][1]) ** 2 >= cs[k][2] ** 2:
                ax.plot(x, y, "x", color="0.3", ms=5, mew=1.0, zorder=7)
    offs = [(5, -12), (5, 2), (-16, 4)]
    for u, (x, y) in enumerate(ov.vertices):
        ax.annotate(f"$v_{u}$", (x, y), textcoords="offset points", xytext=offs[u])
    for k, (x, y, r) in enumerate(cs):
        ax.annotate(f"$C_{k}$", (x, y), ha="center", va="center", color=po.CIRCLE_STYLE[k]["ec"])
    ax.set_title("(a) vertices and boundary arcs")
    ax.tick_params(labelsize=FS)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax = fig.add_subplot(gs[0, 1])
    for k, (x, y, r) in enumerate(cs):
        ax.add_patch(Circle((x, y), r, fc="none", ec=po.CIRCLE_STYLE[k]["ec"], lw=0.6, ls=":"))
    starts = np.array([[a.ux, a.uy] for a in ov.arcs])
    handles = [Patch(fc="#9bc53d", ec="k", lw=0.6, alpha=0.7, label="polygon $P$")]
    ax.add_patch(Polygon(starts, closed=True, fc="#9bc53d", ec="k", lw=0.6, alpha=0.7))
    for arc in sorted(ov.arcs, key=lambda a: a.circle):
        pts = arc.points(300)
        c = po.ARC_COLOURS[arc.circle]
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec="none", alpha=0.35))
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.6)
        po._arrow_on_arc(ax, arc, c, size=9)
        handles.append(Patch(fc=c, alpha=0.35,
                             label=rf"segment on $C_{arc.circle}$, $\Delta\theta={arc.dtheta:.3f}$"))
    xc, yc = ov.centroid
    ax.plot(xc, yc, "+", color="k", ms=9, mew=1.2)
    handles.append(Line2D([], [], ls="none", marker="+", color="k", ms=9, mew=1.2, label="centroid"))
    ax.set_aspect("equal")
    lo = ov.vertices.min(axis=0) - 0.1
    hi = ov.vertices.max(axis=0) + 0.1
    half = 0.5 * max(hi - lo)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_title("(b) polygon and segments")
    lax = fig.add_subplot(gs[0, 2])
    lax.axis("off")
    lax.legend(handles=handles, loc="center left", frameon=False)
    save(fig, "method.pdf")


def reflex():
    cs = ((0.0, 0.0, 1.0), (0.1998, -0.5836, 0.5443), (0.1995, -0.6157, 0.5280))
    ov = tc.overlap(*cs)
    rmin = cs[2][2]
    fig = new_fig(TEXTWIDTH, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45])
    for col, corrected in enumerate((True, False)):
        ax = fig.add_subplot(gs[0, col])
        for k, (x, y, r) in enumerate(cs):
            ax.add_patch(Circle((x, y), r, fc="none", ec=po.CIRCLE_STYLE[k]["ec"], lw=0.7))
        ax.add_patch(Polygon(ov.vertices, closed=True, fc="#9bc53d", ec="k", lw=0.5, alpha=0.6))
        for k, (p0, p1, refl) in enumerate(po._fewell_parts(ov)):
            x, y, r = cs[k]
            a0 = math.atan2(p0[1] - y, p0[0] - x)
            a1 = math.atan2(p1[1] - y, p1[0] - x)
            d = (a1 - a0) % (2 * math.pi)
            if (d > math.pi) != refl:
                a0, a1 = a1, a0
                d = (a1 - a0) % (2 * math.pi)
            if refl and not corrected:
                a0, a1 = a1, a0
                d = (a1 - a0) % (2 * math.pi)
            t = np.linspace(a0, a0 + d, 300)
            pts = np.column_stack((x + r * np.cos(t), y + r * np.sin(t)))
            fc = po.ARC_COLOURS[k] if not refl else ("#d1495b" if corrected else "#7b2cbf")
            ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", alpha=0.45))
        ax.set_aspect("equal")
        ax.set_xlim(-0.42, 0.78)
        ax.set_ylim(-1.17, 0.03)
        ax.set_xticks([])
        ax.set_yticks([])
        a = tc.fewell_area(*cs, corrected=corrected)
        ax.set_title(("(a) major segment" if corrected else "(b) Fewell Eq.~16")
                     + f"\n$A/\\pi r_3^2 = {a / (math.pi * rmin ** 2):.4f}$")
    ax = fig.add_subplot(gs[0, 2])
    ys = np.linspace(-0.80, -0.40, 801)
    ex, ok, bad, refl = [], [], [], []
    for y in ys:
        c = (cs[0], cs[1], (cs[2][0], y, cs[2][2]))
        o = tc.overlap(*c)
        ex.append(o.area)
        tri = o.case == "triangle"
        ok.append(tc.fewell_area(*c) if tri else np.nan)
        bad.append(tc.fewell_area(*c, corrected=False) if tri else np.nan)
        refl.append(o.reflex)
    ex, ok, bad, refl = map(np.array, (ex, ok, bad, refl))
    n = math.pi * rmin ** 2
    ax.fill_between(ys, 0, 1, where=refl, color="#d1495b", alpha=0.12, lw=0,
                    transform=ax.get_xaxis_transform(), label="reflex arc")
    ax.plot(ys, ex / n, color="k", lw=1.5, label="this work")
    ax.plot(ys, ok / n, color=COL["fewell"], lw=1.0, ls=(0, (4, 2)), label=r"Fewell + G\&A")
    ax.plot(ys, bad / n, color=COL["fewell_eq16"], lw=1.2, label="Fewell Eq.~16")
    ax.set_xlabel("$y_3$")
    ax.set_ylabel(r"$A/\pi r_3^2$")
    ax.set_title("(c) displacement of $C_3$")
    ax.set_ylim(0.5, 0.95)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "reflex.pdf")


def accuracy():
    D = np.load(os.path.join(HERE, "accuracy.npz"), allow_pickle=True)
    fams = ["general", "similar", "hierarchical", "corner", "sliver"]
    fig = new_fig(TEXTWIDTH, 2.95)
    axes = fig.subplots(1, 3)
    for ax, fam, methods, title in (
            (axes[0], "corner", ("vector", "fewell", "stable"), "(a) small triangles"),
            (axes[1], "sliver", ("lens_textbook", "vector", "stable"), "(b) thin lenses")):
        ref = D[f"{fam}__ref"]
        rmin = D[f"{fam}__rmin"]
        kap = D[f"{fam}__kappa"]
        x = ref / (math.pi * rmin ** 2)
        for m in methods:
            e = np.abs(D[f"{fam}__{m}"] - ref) / ref
            ax.loglog(x, np.maximum(e, 1e-17), ".", ms=1.2, mew=0, color=COL[m])
        o = np.argsort(x)
        ax.loglog(x[o], (kap * U)[o], "-", color="k", lw=0.5)
        ax.set_xlabel(r"$A/\pi r_\mathrm{min}^2$")
        ax.set_ylim(1e-17, 1e12 if fam == "sliver" else 1)
        ax.set_title(title)
    axes[0].set_ylabel("relative error")
    axes[0].set_xticks([1e-15, 1e-10, 1e-5, 1])
    axes[1].set_xticks([1e-11, 1e-8, 1e-5, 1e-2])
    axes[1].set_yticks([1e-15, 1e-10, 1e-5, 1, 1e5, 1e10])
    ax = axes[2]
    for m in ("stable", "fewell", "vector", "lens_textbook"):
        vals = []
        for fam in fams:
            ref = D[f"{fam}__ref"]
            kap = D[f"{fam}__kappa"]
            e = np.abs(D[f"{fam}__{m}"] - ref) / ref / (kap * U)
            ok = np.isfinite(e) & np.isfinite(ref) & (ref > 0)
            vals.append(e[ok])
        v = np.maximum(np.sort(np.concatenate(vals)), 1e-6)
        ax.loglog(v, 1.0 - np.arange(len(v)) / len(v), color=COL[m], lw=1.2)
    ax.set_xlim(1e-3, 1e18)
    ax.set_xticks([1e-2, 1e3, 1e8, 1e13, 1e18])
    ax.set_ylim(1e-4, 1.3)
    ax.axvline(1, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel(r"error$/\kappa u$")
    ax.set_ylabel("fraction exceeding")
    ax.set_title("(c) all families")
    handles = [Line2D([], [], color=COL[m], lw=2, label=LAB[m])
               for m in ("stable", "vector", "lens_textbook", "fewell")]
    handles.append(Line2D([], [], color="k", lw=0.8, label=r"$\kappa u$"))
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    save(fig, "accuracy.pdf")


def speed():
    """Cost per configuration of this work and the public packages (Table 3)."""
    E = os.path.join(HERE, "external")
    ours = json.load(open(os.path.join(E, "summary.json")))["speed_ours"]
    gef = json.load(open(os.path.join(E, "speed_gefera.json")))
    venn = json.load(open(os.path.join(E, "speed_venn.json")))
    py = json.load(open(os.path.join(E, "speed_py_tools.json")))
    eul = {}
    for line in open(os.path.join(E, "speed_eulerr.csv")).read().splitlines()[1:]:
        name, total, _ = line.split(",")
        eul[name.strip('"')] = float(total)
    sets = ["configs", "traj_jupiter_earth", "traj_large_moon"]
    rows = [
        ("this work", [ours[k] for k in sets]),
        (r"\texttt{gefera}", [np.nan, gef["jupiter+earth"], gef["large moon"]]),
        (r"\texttt{venn.js}", [venn[k] for k in sets]),
        (r"\texttt{eulerr}", [eul[k] for k in sets]),
        (r"\texttt{matplotlib-venn}", [py[k]["mvenn"] for k in sets]),
        ("Shapely, 8", [py[k]["shapely_q8"] for k in sets]),
        ("Shapely, 64", [py[k]["shapely_q64"] for k in sets]),
        ("Shapely, 512", [py[k]["shapely_q512"] for k in sets]),
    ]
    labels = ["40,000 test configurations", r"sequence $r_p, r_m = 0.1, 0.0092$", r"sequence $r_p, r_m = 0.12, 0.05$"]
    colours = ["#1b7f5b", "#edae49", "#00798c"]
    fig = new_fig(COLUMNWIDTH, 3.6)
    ax = fig.subplots()
    h = 0.26
    y = np.arange(len(rows))[::-1]
    for k in range(3):
        vals = np.array([r[1][k] for r in rows], float)
        ax.barh(y + (1 - k) * h, vals, height=h, color=colours[k], label=labels[k], left=1.0)
    ax.set_xscale("log")
    ax.set_xlim(10, 3e6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ns per configuration")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.legend(loc="outside upper center", ncol=1, frameon=False)
    save(fig, "speed.pdf")


if __name__ == "__main__":
    taxonomy()
    method()
    reflex()
    accuracy()
    speed()
    print("figures written to", OUT)
