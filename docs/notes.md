# The common overlap of three circles: what we learned

This folder summarises the work on the area of the common overlap of three
disks, A = |D₁ ∩ D₂ ∩ D₃|. It was done for the exomoon transit code
**Pandora** (Hippke & Heller 2022), where the quantity decides how much of a
moon's shadow is hidden behind its planet while one of them is on the
stellar limb. The method is general: any three circles, any units.

| File | What |
|---|---|
| `THREE_CIRCLE_OVERLAP.md` | this document |
| `three_circles.py` | standalone calculator (numpy; numba optional): API, CLI, self test, benchmark |
| `examples.py` | usage examples |
| `plot_overlaps.py` | vector-PDF figures; `figures/` holds the generated PDFs |

Quick start:

```bash
python three_circles.py area 0 0 1  0.8 0.3 0.6  0.5 -0.4 0.7
python three_circles.py pandora 0.96 0.03 0.10 1.00 -0.02 0.05    # xp yp rp xm ym rm
python three_circles.py examples        # the 20 curated geometries
python three_circles.py selftest        # validation vs an independent quadrature
python three_circles.py bench
python examples.py
python plot_overlaps.py                 # -> figures/*.pdf
python plot_overlaps.py --circles 0 0 1 0.8 0.3 0.6 0.5 -0.4 0.7 --pdf my.pdf
```

```python
import three_circles as tc
tc.overlap_area((0, 0, 1), (0.8, 0.3, 0.6), (0.5, -0.4, 0.7))     # 0.357820532455879
ov = tc.overlap((0, 0, 1), (0.8, 0.3, 0.6), (0.5, -0.4, 0.7))
ov.case, ov.n_vertices, ov.reflex, ov.arcs, ov.vertices, ov.centroid, ov.boundary()
tc.overlap_area_batch(x1, y1, r1, x2, y2, r2, x3, y3, r3)          # broadcasting arrays
tc.reference_area(c1, c2, c3)          # independent chord quadrature (area, x_c, y_c)
tc.fewell_area(c1, c2, c3, corrected=False)   # the historical (wrong) reflex formula
tc.pandora_er(xp, yp, rp, xm, ym, rm)         # exact eclipse ratio
tc.pixelart_er(xp, yp, xm, ym, rp, rm, 25)    # Pandora's legacy raster (its argument order!)
```

---

## 1. The problem

### 1.1 In Pandora

A planet (radius r_p) and its moon (r_m) transit a star of radius 1. When
the two bodies overlap on the sky, the overlap must not be counted twice.
Pandora handles this in `eclipse()`:

- Both bodies fully on the star: the overlap is a two-circle lens (exact
  closed form, `cci()`).
- **At least one body on the stellar limb:** the light that is blocked twice
  is the part of the moon that lies on the star *and* behind the planet:

  ```
  A_SMP = |star ∩ planet ∩ moon|          (three-circle overlap)
  A_SM  = |star ∩ moon|                   (lens)
  er    = A_SMP / A_SM   (er = 1 when the moon is off the star)
  ```

  The moon's flux deficit is scaled by (1 − er).

Before this work, Pandora estimated A_SMP with a **pixel raster**
(`pixelart()`, default 25 × 25 pixels across the moon). The Pandora paper
(Sect. 2.5) cites Fewell's 9 and Kipping's 27 cases and calls the raster
"much more pragmatic". It runs at 0.25 million points per second per core,
about 3 orders of magnitude slower than the analytic paths.

### 1.2 In general

The same quantity appears wherever three disks overlap: mutual events of
planets and moons, planet–planet occultations, circumbinary transits,
eclipsing multiples, and outside astronomy in wireless coverage or Venn
diagrams.

---

## 2. Previous work and its errors

### 2.1 Literature map

| Who | Method | Problem found |
|---|---|---|
| **Fewell 2006** (DSTO-TN-0722) | First closed form: circular triangle = Heron triangle on the three chords + three circular segments (his Eq. 1); 7-step algorithm; 9-case taxonomy | **Reflex formula (Eq. 16) is wrong** |
| **Kipping 2011** (LUNA, MNRAS 416, 689) | Fewell transcribed to star/planet/moon; 27 cases + subcases | Inherits the error (**Eq. 37**, case 14.1b) |
| **Gordon & Agol 2022** (gefera, MNRAS 2023) | Exact limb-darkened mutual transits via Green's theorem; 16-case tree | **App. B corrects Fewell/Kipping.** gefera itself returns NaN when a planet–moon crossing lies exactly on the limb and at b + r = 1 (found while porting it; the Fortran is built with -Ofast) |
| Pál 2012; Short et al. 2018 | Green's theorem / numerical line integrals | framework, cross-check |
| **Hippke & Heller 2022** (Pandora) | Pixel raster `pixelart()` | quantisation error, slow; quirks below |
| Kálmán et al. 2023 (TLCM) | Pixel classification + Gauss–Legendre quadrature | alternative, also limb-darkening-weighted |
| Librino et al. 2012 | Trellis / inclusion–exclusion for N circles | confirms Fewell as the N = 3 reference |

### 2.2 The reflex error (Fewell Eq. 16, Kipping Eq. 37)

A circular triangle is the Heron triangle on its three chords plus one
circular segment per circle. A segment on chord c of a circle of radius r
has the area

```
minor segment (arc < π):  r² arcsin(c/2r) − (c/4) √(4r² − c²)
major segment (arc > π):  r² (π − arcsin(c/2r)) + (c/4) √(4r² − c²)   (Gordon & Agol 2022, App. B)
Fewell Eq. 16 / Kipping:  r² arcsin(c/2r) + (c/4) √(4r² − c²)          (wrong)
```

**Root cause.** The chord length does not determine the arc: arcsin(c/2r)
returns at most π/2, the half-angle of the *minor* arc. When the region
contains more than half of a circle (the circle's centre lies on the far
side of the chord from the third vertex), the arc is the *major* one. Eq. 16
flips the sign of the triangle term, but it keeps the minor arc's angle and
misses the π r² term.

**How large is the error?** (`fewell_area(..., corrected=False)`,
`figures/fig_reflex.pdf`)

- Curated example `reflex-deep-1` (moon arc 3.78 rad):
  - exact 0.751936;
  - Eq. 16 gives 0.572922, **−20.4 % of the moon's disk**.
- Over 93 reflex triangles in a 20,000-configuration fuzz, the maximum error
  is **52.7 % of the smallest disk**.
- The historical formula is continuous at the transition (arc = π) and
  diverges beyond it. That is why it can look plausible.
- Gordon & Agol saw it as LUNA–gefera light-curve discrepancies near the
  inner edge of the stellar limb (their Sect. 8). That is the Pandora
  regime: a moon of comparable size partly behind a planet on the limb.
- Reflex triangles need radii that are not too different. For an
  Earth-sized moon behind Jupiter they are rare, but they do occur for big
  moons and in general circle problems.

### 2.3 Pandora's pixel raster (`pixelart()`)

We validated it with an independent pixel counter on the same grid. The
verdict: **no counting bug**. The two counters differ by 0–24 pixels per
map, all within 0.0015 px of a circle boundary. The raster's problems are
resolution, cost and a few quirks:

- **Quantisation.** Measured against the exact area:
  - pixelart at n = 25 is off by up to 0.52 % of the moon area on the
    curated suite and 1.2 % on random geometries;
  - its er is off by up to **0.105** on the suite (`taxonomy-tri-moonbig`:
    0.893 vs 0.998);
  - realistic limb case `realistic-tri-1`: 1.0000 instead of 0.9858;
  - for slivers of moon on the limb, er errs by up to 1.0 (0.5 in the sweep
    of `figures/fig_raster.pdf` e).
  - The error falls roughly as 1/n (`fig_raster.pdf` c), but with large
    scatter: a coarse grid can be right by luck.
- **Light-curve impact (Pandora regression cases):**
  - up to 4.4 ppm in flux;
  - a spurious 0.07 ppm moon signal where the moon is fully hidden;
  - a spurious er = 0.004 for exactly externally tangent bodies.
- **Cost:** 3464 ns per call at n = 25 inside numba (0.25 M points/s, as the
  paper states); 49 µs at n = 99.
- **Quirks (not bugs):**
  - The grid has (n + 1) × (n + 1) samples, not n × n; an even n is
    silently made odd.
  - The "anti_aliasing" dilations are dimensionally inconsistent between the
    moon and the star/planet tests (both ≪ 0.05 px). They compensate the
    pixel-centre undercount.
  - The colour codes (star 5, moon 3, planet 2) are ambiguous: 5 means
    "star" or "moon + planet". This is harmless because only the unique sum
    10 is counted, but it matters if one reuses the image (found while
    drawing `fig_raster.pdf`).

### 2.4 The textbook lens formula loses precision on slivers

The MathWorld two-circle lens

```
A = r₁² acos((d² + r₁² − r₂²)/(2 d r₁)) + r₂² acos(...) − ½ √((−d+r₁+r₂)(d+r₁−r₂)(d−r₁+r₂)(d+r₁+r₂))
```

is used in Pandora's `cci()`, in our first analytic version and in many
codes. It cancels catastrophically for thin overlaps: acos(1 − ε) ≈ √(2ε)
loses digits, and the three terms nearly cancel. Measured against 60-digit
mpmath (`figures/fig_precision.pdf`), maximum relative error:

| Overlap | Textbook acos form | This code (segments + Kahan Heron) |
|---|---|---|
| star (r = 1) ∩ moon, moon just inside the limb | **1.5 × 10⁶** | 3.0 × 10⁻¹⁵ |
| planet ∩ moon, grazing | 3.9 × 10³ | 2.5 × 10⁻¹⁵ |
| random lenses | 1.5 × 10⁻⁸ | 1.6 × 10⁻¹⁵ |

For Pandora the absolute effect is negligible. The optimized kernel errs by
up to 1.3 × 10⁻⁵ of the moon's disk on limb slivers, i.e. ≲ 10⁻⁹ in stellar
flux for an Earth-sized moon, far below 1 ppm. But it makes the relative
area of thin slivers meaningless.

### 2.5 The constant-intensity assumption (physics, not geometry)

Pandora (like LUNA) gives the overlap the mean stellar intensity of the
moon's on-star disk. Against an exact limb-darkened reference (gefera
port), Jupiter-sized planet and quadratic limb darkening (0.4, 0.26):

| Moon | Before | After the first-order correction |
|---|---|---|
| Earth-sized (k = 0.0092), fully on the star | 1.36 ppm | 0.20 ppm |
| Earth-sized, on the limb band | 1.37 ppm | 0.58 ppm |
| Neptune-sized moons (regression cases) | 14.4 ppm | 2.8 ppm |

The correction (Pandora "T4", `lens_gradient_term()`):

- It adds the intensity gradient at the moon centre times the offset of the
  overlap centroid.
- It fades to zero across the limb band, because Pandora had no centroid of
  the *three*-circle region.

`Overlap.centroid` (Sect. 3.6) now provides exactly that centroid. This is
a possible extension, not used in Pandora yet. The exact limb-darkened
overlap (gefera) costs 0.6–1.1 µs per partially overlapping sample.

---

## 3. Our method

### 3.1 Algorithm

1. **Containment reduction.** If disk i lies in disk j (|o_i − o_j| + r_i ≤
   r_j, including internal tangency), then D_i ∩ D_j = D_i and disk j is
   dropped; repeat.
   - One disk left: π r².
   - Two left: their lens.
2. **Disjoint pair.** If any remaining pair is disjoint or externally
   tangent, A = 0.
3. **Vertices.** These are the pairwise boundary crossings that lie
   *strictly inside* the third disk; they are the corners of the region.
   After step 1 and step 2, the non-degenerate cases are:

   | V | Region |
   |---|---|
   | 0 or 1 | empty (Fewell case i; V = 1 is a tangency) |
   | 2, same pair | that pair's lens, lying inside the third disk |
   | 2, different pairs | empty (tangential contact) |
   | 3 | circular triangle (exactly one vertex per pair) |
   | 4 | circular quadrilateral (Fewell case d: all four vertices on one circle) |

   **V = 5 or 6 does not occur unless the configuration is degenerate**: it
   would need a pairwise crossing to fall exactly on a lens corner (measure
   zero). This was argued from the region's convexity and arc counting and
   confirmed empirically: no V > 4 in any fuzz run.
4. **Directed arc walk.** The region is convex and lies inside every disk,
   so its boundary, traversed counter-clockwise, runs CCW around each arc's
   own centre. From every vertex u there is exactly one out-arc u → v on a
   circle s shared by u and v that satisfies:
   - (a) the open span contains no other crossing of circle s (with either
     other circle), so membership in the other disks is constant along it;
   - (b) its midpoint lies strictly inside the two other disks.

   Following the out-arcs from vertex 0 must visit every vertex once and
   close. Anything else is reported as degenerate.
5. **Area by Green's theorem**, arc by arc (Sect. 3.2).

The structure was built as a correctness-first reference (pure Python,
`overlap3/analytic.py` in the Pandora project). It was then optimized
(Sect. 4) and vendored into Pandora as `pandoramoon/overlap_analytic.py`
(commit `5edf35f`), where it is the default (`eclipse_method="analytic"`).

### 3.2 Green's theorem without trigonometry

A = ½ ∮ (x dy − y dx). Take an arc of the circle with centre o and radius r,
from angle α to β = α + Δθ, x = o_x + r cos t, y = o_y + r sin t:

```
½ ∫ (x dy − y dx) = ½ [ r² Δθ + o_x r (sin β − sin α) − o_y r (cos β − cos α) ]
```

With the end points u = o + r(cos α, sin α) and v = o + r(cos β, sin β):
r sin β − r sin α = v_y − u_y and r cos β − r cos α = v_x − u_x. Hence

```
A_arc = ½ [ r² Δθ + o_x (v_y − u_y) − o_y (v_x − u_x) ],
Δθ = atan2(u' × v', u' · v') ∈ (0, 2π],   u' = u − o,  v' = v − o
```

- All position trigonometry cancels; one atan2 per arc remains. This cut
  ~24 transcendental calls per walk to 4–6.
- **Arcs longer than π need no special case**: Δθ comes from the geometry,
  not from a chord length. The Gordon & Agol correction is built in by
  construction.
- Arc selection uses only cross-product signs. The midpoint of the CCW arc
  is o ± r (u' + v')/|u' + v'| (plus sign if u' × v' > 0); antipodal end
  points are a special case.

### 3.3 Numerically stable evaluation (new in `three_circles.py`)

Summing A_arc over the cycle and regrouping (each A_arc = segment + triangle
o u v, and the triangles' o terms cancel around the cycle) gives

```
A = Σ_arcs  r²/2 (Δθ − sin Δθ)   +   ½ Σ_edges (u − p₀) × (v − p₀)
           (circular segments)         (shoelace polygon of the vertices)
```

This is Fewell's "Heron triangle + segments", generalised to any V, with
reflex arcs included automatically. The standalone code evaluates it as
follows:

- **Segments:** Δθ − sin Δθ by its Taylor series below 0.5 rad (no
  cancellation for thin segments). Above 0.5 rad it uses r² sin Δθ = u' × v'
  from the coordinates, so the walk still calls no sin.
- **Polygon:** shoelace relative to the first vertex. The precision is then
  relative to the size of the region, not to the coordinates.
- **Crossings and lenses:**
  - the half chord comes from **Kahan's stable Heron product** (sort
    a ≥ b ≥ c, then (a+(b+c))(c−(a−b))(c+(a−b))(a+(b−c))) instead of
    r² − a²;
  - the centre-to-radical-line distances use (d − r₂)(d + r₂) + r₁²;
  - the lens area is two segments with angles from atan2 (Sect. 2.4 table).
- **Origin:** shifted to the centre of the smallest circle (translation
  invariance of the area).

Effect: over 20,000 random configurations, including scales from 10⁻⁶ to
10⁶ and offsets of 10³ radii, the maximum deviation from the independent
quadrature is **3.2 × 10⁻¹⁴ of the smallest disk**. The first draft of
this file (vector form plus acos lens, as in Pandora) reached
6.9 × 10⁻¹⁰ on sliver lenses in the same test.

### 3.4 Tolerances and degenerate input

- **Windows** (relative to the largest radius):
  - tangency/containment/disjointness: 10⁻¹²;
  - strictly inside (vertex and midpoint tests): 10⁻⁹;
  - coincident vertices: 10⁻⁹.
- **Measure-zero configurations** are reported as degenerate (NaN in the
  kernel):
  - a crossing lying *on* the third circle (a triple point);
  - coincident vertices;
  - a broken cycle.
- **Fallbacks for degenerate input:**
  - Pandora falls back to the raster.
  - `three_circles.overlap()` re-evaluates with the radii nudged by
    ±10⁻⁸ max(r) and averages the two areas (`Overlap.perturbed = True`).
    Example: a circle through both corners of a lens gives 1.2283696986087x,
    equal to the exact lens to 10⁻¹⁵.
  - Three circles through one point give 7 × 10⁻¹⁸.
- The first version returned A = 0 for a circle through both lens corners
  (V = 0 because both corners are "not strictly inside"). The explicit
  on-circle check fixes that.
- Lesson: floating-point equivalence tolerances must scale with an
  intermediate magnitude (disk area), not with a possibly tiny result.

### 3.5 Fast paths of the production kernel (`overlap3/analytic_fast.py`, in Pandora)

- **V = 3:** one vertex per pair, so every circle carries exactly two
  vertices and one arc. The midpoint test alone fixes each arc's direction,
  and the adjacencies are known a priori. The kernel is fully scalar, with
  no allocation and no candidate scan.
- **V = 4 (Fewell case d):** the two arcs on the shared circle k are single
  gaps between angularly adjacent vertices with forced CCW direction. The
  arcs on circles i and j are selected by midpoint. Only the W X X W /
  X W W X arrangements admit a cycle.
- **Fallback:** any structural surprise falls back to the generic walk, so
  the fast paths cannot change results.
- **er early-out:** A(star ∩ moon) is computed first; if it is 0, the kernel
  returns er = 1 at once.

The standalone file keeps only the generic walk (clearer, and with the
stable evaluation of 3.3). It is therefore ~3× slower than the production
kernel (Sect. 4).

### 3.6 Centroid (extension)

Region = polygon + segments. The centroid of a segment lies on its axis, at

```
e = r [ 4 sin³φ / (3 (2φ − sin 2φ)) − cos φ ],   φ = Δθ/2
```

beyond the chord midpoint. Below φ = 0.5 the code uses the Taylor series
e/r = φ²/5 − 0.0123810 φ⁴ + … (coefficients derived with exact rational
arithmetic), because the closed form cancels. Against the independent
quadrature, the centroid error is ≤ 1.7 × 10⁻¹² of the smallest radius.
A naive Green's-moment version erred by 3 × 10⁻⁵ of the radius (1.5 % of
the region size) on a tiny region far from the origin.

---

## 4. Faster: measured history

Single core, Intel Core Ultra 5 226V, numba 0.68. "Internal" means called
from jitted code, which is how Pandora calls it.

| Step | Per call | Notes |
|---|---|---|
| Pandora `pixelart()` n = 25 (baseline) | 3464 ns internal | 0.25 M points/s, as in the paper |
| v1 pure Python (reference) | 65–70 µs | correctness first; 20× *slower* than the raster |
| v1 numba port, bit-identical | 841 ns | pair enumeration order must match exactly: (0,2) vs (2,0) changes results by 10⁻¹⁰ (cancellation) |
| v2 vector-form walk (3.2) | ~123 ns | transcendental calls 24 → 4–6; bug found on the way: the reflex branch must exclude the arc end points from the no-crossing test |
| **v3** V = 3 / V = 4 fast paths, er early-out | **88 ns (er), 75 ns (area)** | **39× faster than the raster**, exact; v2 → v3 interleaved 1.46× |
| this folder, generic walk + stable evaluation | 245 ns (batch), 673 ns (scalar from Python), 44 µs (pure Python) | `python three_circles.py bench` |

Per case in v3: empty 14 ns, full disk 12.5 ns, lens 30 ns, V3 walk 100 ns,
V4 walk 145 ns. The walk is within ~2× of its floor of 2–6 atan2 plus a few
sqrt. A fast atan2 approximation was rejected on accuracy.

**Inside Pandora** (commit `5edf35f`):

- `eclipse()` per eclipse draw: Kepler 40 → 8 µs, PLATO 190 → 32 µs.
- Whole likelihood: about +3 %, because only ~10 % of the sampler's draws
  have a mutual eclipse, with only ~9–43 limb-overlap samples each.
- The eclipse-to-non-eclipse cost ratio fell from 1.38–1.46 to 1.03–1.04.

Later follow-ups:

- **R6a**: test the squared separation before any square root in
  `eclipse()`'s per-sample loop (1.5 → 0.65 ns per rejected sample).
- **R6b** (share the lens geometry with the limb-darkening term) and
  **R6c** (overlap phase windows from the orbit) were measured and
  rejected. They gave ≤ 0.2 % on average for 33–100 extra lines.

**Measurement lessons:**

- The Python → numba boundary costs ~150 ns per scalar call. Benchmark hot
  kernels from a jitted driver; otherwise the speedup looks 2× smaller
  (15× instead of 39×).
- Compare implementations interleaved in one process: session and thermal
  drift exceed the code effects.
- `cache=True` staleness: clear `__pycache__/*.nb*` after editing a callee.
  A stale cache once made the V4 fast path look inert.
- A never-executed branch or extra code in a hot jitted function can cost
  several % (code layout, inlining). Split rarely used paths into separate
  functions.

---

## 5. Better: validation

**`three_circles.py selftest`** (all pass; numbers for `--n 20000`):

- 20 curated examples:
  - case labels and reflex flags correct;
  - area vs the independent chord quadrature ≤ 3.5 × 10⁻¹⁵ of the moon
    area.
- 20,000 random configurations in four families: general, Pandora-like
  limb, nearly equal radii (reflex-rich), extreme scales and offsets.
  - Area: ≤ 3.2 × 10⁻¹⁴ of the smallest disk.
  - Centroid: ≤ 1.7 × 10⁻¹².
  - Cases: 7967 empty, 2267 disk, 4897 lens, 4635 triangles,
    234 quadrilaterals.
- Fewell Eq. 1 with the Gordon & Agol correction equals the walk on all
  4635 triangles (≤ 3.2 × 10⁻¹²). The historical Eq. 16 fails on reflex
  triangles (up to 52.7 %).
- Degenerate cases: triple points, identical circles, concentric circles,
  a circle through both lens corners.
- The legacy raster converges to the exact er: max error 0.105 at n = 25,
  0.0021 at n = 399.

**Independent reference (`reference_area`).** It uses no arcs and no cases:

- A = ∫ max(0, min_i (y_i + s_i) − max_i (y_i − s_i)) dx, with s_i the
  half chord of circle i.
- The x axis is split at all circle extents and all crossings, so the
  integrand is smooth on each piece.
- The substitution x = a + (b − a)(1 − cos t)/2 removes the square-root end
  points; then Gauss–Legendre in t, with adaptive bisection where two rule
  orders disagree (near tangencies).

**Earlier project (`overlap3/`, 171 tests):**

- pixel maps from two independent counters (`out/pixelmaps.pdf`);
- lens cases identical to Pandora's `cci`;
- v1 numba bit-identical to pure Python;
- v3 within 4.5 × 10⁻¹⁴ of the disk area over 20,000 targeted fuzz
  configurations (3445 V3 and 359 V4 walks);
- inside Pandora, er within 7 × 10⁻⁸ of an independent chord integration
  over 1500 limb geometries;
- `pixelart_er` here reproduces the original pixel images exactly
  (2040 of 2040 maps compared).

---

## 6. Summary: what is better, what is faster

| | Previous | Now |
|---|---|---|
| Reflex circular triangles | Fewell Eq. 16 / Kipping Eq. 37 wrong (20–53 % of the small disk); fixed by case analysis in G&A 2022 | correct by construction (Δθ from atan2), no case distinction |
| Pandora limb eclipses | 25-px raster: area ≤ ~1 % of the moon, er up to 0.1–1.0 off for slivers, ≤ 4.4 ppm in flux | exact; ≤ 10⁻¹⁴ of the disk |
| Speed | raster 3464 ns per point | 88 ns (Pandora kernel), 39× |
| Thin slivers | textbook acos lens: relative error up to 10⁶ | segments + Kahan Heron: ≤ 3 × 10⁻¹⁵ |
| Degenerate input | gefera NaN at a crossing on the limb; raster always "works" but quantised | detected; Pandora falls back to the raster; the standalone averages two ±10⁻⁸ nudges |
| Region centroid | not available (Pandora's LD-gradient term switched off on the limb) | `Overlap.centroid`, ≤ 2 × 10⁻¹² |
| Taxonomy | 9 (Fewell) / 27 (Kipping) / 16 (G&A) cases | 5 outcomes from V ∈ {0…4}; one generic walk |

**Limits.**

- The area is exact *geometry*. Using it with one intensity for the overlap
  is a limb-darkening approximation (Sect. 2.5).
- Near-tangent input is ill-conditioned: errors in the input radii and
  distances propagate, which no formula can fix.
- The standalone file trades ~3× speed for generality and precision. For
  hot loops, use Pandora's `overlap_analytic.py`, or port the V = 3 / V = 4
  fast paths.

---

## 7. Figures (`python plot_overlaps.py`, vector PDF)

| File | Content |
|---|---|
| `figures/fig_taxonomy.pdf` | every outcome: empty (Fewell case i), disk, lens, triangle, reflex triangle, quadrilateral |
| `figures/fig_walk.pdf` | (a) vertices vs crossings outside the third disk, directed arcs with Δθ; (b) polygon + segments decomposition and centroid |
| `figures/fig_reflex.pdf` | correct vs historical segment for `reflex-deep-1`; sweep showing the historical formula diverging once an arc exceeds π |
| `figures/fig_raster.pdf` | Pandora's raster at n = 25 / 99 against the exact boundary; error vs grid; er along a track crossing the limb behind the planet |
| `figures/fig_precision.pdf` | relative lens error vs overlap depth: textbook acos vs stable form (needs mpmath) |
| `figures/fig_examples.pdf` | the 20 curated Pandora geometries (4 pages) with exact and raster er |
| `figures/custom_example.pdf` | output of `--circles` for one configuration |

---

## 8. References

- Fewell, M. P. 2006, *Area of Common Overlap of Three Circles*, DSTO-TN-0722.
- Kipping, D. M. 2011, MNRAS 416, 689 (LUNA).
- Gordon, T. A. & Agol, E. 2022, arXiv:2207.06024; MNRAS 522, 2439 (2023) (gefera; App. B).
- Hippke, M. & Heller, R. 2022, A&A, arXiv:2205.09410 (Pandora).
- Pál, A. 2012, MNRAS 420, 1630; Short, D. R., Orosz, J. A. & Jenkins, J. M. 2018, AJ 156, 297.
- Kálmán, Sz. et al. 2023, arXiv:2311.04647 (TLCM moon module).
- Librino, F., Levorato, M. & Zorzi, M. 2012, arXiv:1204.3569.
- Kahan, W. 2014, *Miscalculating Area and Angles of a Needle-like Triangle* (stable Heron).

Source material: the Pandora project at
`/home/michael/Downloads/pandora/`. It includes `overlap3/` (development
project, 171 tests), `papers/` (PDFs and notes), `PERFORMANCE_LOG.md` §1 and
§44–45, and `Pandora/BUGS.md` (known limitations).
