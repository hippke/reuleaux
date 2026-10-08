# External codes vs. this work (benchmark harness, not yet in the paper)

Shared input: `configs.npz` = the 40,000 configurations of the paper's
accuracy test (`../accuracy.npz`, five families), with 50-digit references
`ref`, condition numbers `kappa` and exact lenses `L01`, `L02` (`prep.py`).

| Script | Code under test | Run with |
|---|---|---|
| `run_gefera.py [photlib.so / photlib_O2.so]` | gefera (Gordon & Agol 2022), Fortran `photlib` built from github.com/tagordon/gefera @ e27e832 with its own flags (`-Ofast`) and with `-O2` | env with numpy (ctypes) |
| `run_venn.js` | venn.js `intersectionArea` (Frederickson), src/circleintersection.js @ 5ff0ef3 | `node run_venn.js` |
| `run_eulerr.R` | eulerr 7.1.0 `intersect_ellipses(circle = TRUE)` (Larsson) | `Rscript run_eulerr.R` |
| `run_py_tools.py` | matplotlib-venn 1.1.2 region algebra; Shapely 2.1.2 / GEOS 3.14.1 polygons | python with both installed |
| `analyse.py` | comparison, our timings (`summary.json`) | main env (numba) |

The external tools ran in a temporary conda env (python 3.10, numpy 1.23,
gfortran, nodejs 26, R 4 + eulerr, shapely, matplotlib-venn), all pinned to
CPU core 2 for timing (`taskset -c 2`).

Interface notes: gefera's `flux_ng` returns F - 1 and expects
theta in [0, pi] (its `impacts()` computes theta from the three distances),
so sin(theta) >= 0; with a uniform source the blocked area is -pi (F - 1).
