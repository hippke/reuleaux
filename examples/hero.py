"""Draw a three-circle overlap: python examples/hero.py -> docs/hero.png"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import reuleaux as rx

circles = [(0.0, 0.0, 1.0), (1.05, 0.15, 0.85), (0.45, -0.95, 0.9)]
ov = rx.overlap(*circles)

colors = ["#4C9BE8", "#F2A541", "#E6527C"]
fig, ax = plt.subplots(figsize=(7, 6.2), dpi=200)
for (x, y, r), c in zip(circles, colors):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", alpha=0.28))
    ax.add_patch(Circle((x, y), r, fc="none", ec=c, lw=1.6))
ax.add_patch(Polygon(ov.boundary(), closed=True, fc="#FFFFFF", ec="none", alpha=0.85, zorder=3))
for a in ov.arcs:                       # overlap boundary, coloured by circle
    p = a.points()
    ax.plot(p[:, 0], p[:, 1], color=colors[a.circle], lw=3.4, solid_capstyle="round", zorder=4)
ax.scatter(*ov.vertices.T, s=46, c="#1b2330", zorder=5)
cx, cy = ov.centroid
ax.scatter([cx], [cy], marker="+", s=140, c="#1b2330", lw=2, zorder=5)
ax.text(1.45, -1.25, f"A = {ov.area:.6f}", fontsize=14, color="#1b2330", weight="bold")
ax.text(1.45, -1.5, f"{ov.case}, V = {ov.n_vertices}\n\u25CF vertices   + centroid", fontsize=10, color="#4a5565", va="top")
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-1.15, 2.2); ax.set_ylim(-2.0, 1.2)
fig.savefig("docs/hero.png", bbox_inches="tight", pad_inches=0.15, facecolor="white")
