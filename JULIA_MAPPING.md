# JULIA_MAPPING.md

**Porting findings: the Z7/IGEO7 Xarray + Zarr + focal-kernel stack to Julia**

Target Julia stack: `Zarr.jl` + `DiskArrays.jl` + `DimensionalData.jl` (/ `YAXArrays.jl`)
+ `KernelAbstractions.jl`, with **bidirectional read/write Zarr compatibility** against
the Python (`xarray` / `xdggs` / `zarr-python` v2) archives produced by this repository.

This is a *findings and porting-contract* document, not an implementation. It records
the facts a Julia implementation has to honour, which of them were verified against the
artefacts in this repo, and which still need to be checked on the Julia side.

Legend:

- ✅ **verified** — checked directly against source and/or the shipped archives in
  `data/working/` during this analysis.
- 📄 **documented** — asserted by code/docstrings/tests in this repo, taken at face value.
- ⚠️ **to verify in Julia** — expected behaviour of a Julia package; no Julia toolchain
  was available in the analysis environment, so this must be confirmed before relying on it.

---

## 0. Executive summary

1. **The hard part is already Julia.** `z7py` is a *port from* Julia, not the origin:
   `z7py/tests/test_z7.py` says it "Mirrors the reference test suite in `z7jl/tests_z7.jl`",
   `z7py/latitudes.py` says "Translated from `z7jl/AuthalicLatitude.jl`", and the lookup
   tables in `z7py/z7.py` carry comments like "(Julia 1-indexed, Python 0-indexed)" and
   "Equivalent to Julia's `leading_zeros`-based implementation" (✅, see §2.6).
   Upstream is <https://github.com/wrenoud/Z7/>. So the GBT index arithmetic should be
   *adopted*, not re-derived; the genuinely new Julia work is the **array/storage/index
   layer** (§3) and the **focal-kernel layer** (§4).
2. **The Zarr archives are plain Zarr v2 + `_ARRAY_DIMENSIONS`** (the xarray convention)
   with no exotic codecs and no consolidated metadata — the easiest possible interop
   target for `Zarr.jl` (§3.2, §6).
3. **The one genuinely new index concept** — the monotonic base-7 RangeIndex — maps
   cleanly onto a custom `DimensionalData.Lookups.Lookup` wrapping a lazy
   `AbstractVector{UInt64}`; no xarray-style "index plugin" protocol is needed (§5.3).
4. **The slope kernel is a gather + elementwise expression** over an `(N,6)` neighbour
   table. It is a natural `KernelAbstractions.@kernel`, with two caveats: use
   `(6, N)` column-major layout, and Metal has no `Float64` (§4.4, §5.5).
5. **Three concrete traps** for a Julia port: signed/unsigned (`Int64` breaks base cells
   8–11), dimension-order reversal in `Zarr.jl` for the 2-D ranges coordinate, and
   1-based vs 0-based indexing in tables that are *value*-indexed (digits 0…6) rather
   than *position*-indexed (§6.4, §2.5).

---

## 1. Source inventory

What exists in this repo, and what each part means for a Julia port.

| Path | Role | Julia porting relevance |
|---|---|---|
| `z7py/z7.py` (704 ln) | Bit-packed Z7 index arithmetic: digits, parent, resolution, monotonic int, GBT neighbour tables & carry cascade, pentagon/rotation handling | **Already exists upstream in Julia (`z7jl`).** Reuse; do not re-port from Python |
| `z7py/latitudes.py` | Authalic ↔ geodetic latitude (Karney 2022 series, WGS84) | Translated *from* `z7jl/AuthalicLatitude.jl`; reuse upstream |
| `z7py/tests/test_z7.py` (496 ln) | Test suite mirroring `z7jl/tests_z7.jl` | **Golden-vector source** for cross-language conformance (§8.1) |
| `src/z7_xarray_paper/z7_zarr.py` (363 ln) | Zarr archive writer/reader, dggs-convention v1 metadata, `find_monotonic_ranges`, `compression={"none","ranges"}` dispatch | **Primary port target** → §3 |
| `src/z7_xarray_paper/config.py` | DGGRID/IGEO7 metafile constants, convention registration block, default compressor, `DEFAULT_CHUNK_CELLS = 7^5` | Port as plain constants |
| `src/z7_xarray_paper/kernels/neighbours.py` | `(N,6)` neighbour table + `searchsorted` position projection, `INVALID` sentinel | **Port target** → §4.2 |
| `src/z7_xarray_paper/kernels/halo.py` | `HaloChunk` (core+halo, sorted, `owned_mask`) and its builder | **Port target** → §4.5 |
| `src/z7_xarray_paper/kernels/slope.py` (386 ln) | FDA slope on Z7 axes; eager `slope()` and lazy chunked `slope_blocked()` | **Port target** → §4 |
| `src/z7_xarray_paper/distance_measures.py` | `HexGridDistortionModel`: per-axis anisotropy weights from a level-4 parent table | Port target (trivial arithmetic, §4.3) |
| `scripts/phase3a_slope_math.py` | Settles the axis labelling + validates FDA against a tilted plane and a least-squares gold standard | Reference for the Julia validation suite |
| `scripts/phase3a_global_band_sweep.py` | Builds the global level-4 anisotropy table (`d_k, d_j, d_i` per parent) | Produces `dist_lookup_level4.parquet`; Julia only needs to *read* it |
| `scripts/phase3b_slope_pori_r10.py` | End-to-end in-memory slope on Pori r10 | Numeric reference run |
| `scripts/demo_regrid_pori_to_z7.py` | DEM → IGEO7 regrid → sort → ranges → Zarr write → round-trip assertions | The write path the Julia writer must match |
| `scripts/z7_slope_blocked_halo.ipynb` | Blocked/halo slope at r12 with `Z7MonotonicIndex` | Source of the chunk-size recommendation (7^6) |
| `data/working/*.zarr` | 4 shipped archives (2 dense, 2 ranges; levels 10 and 12) | **Interop fixtures** — a Julia reader must open these unchanged (§8.2) |
| `data/working/dist_lookup_level4.parquet` | 24 012 rows × (`n`, `mean_d_k`, `mean_d_j`, `mean_d_i`, `mean_d_geom`), `cell_ids` UInt64 index | Read with `Parquet2.jl`/`Arrow.jl` (§4.3) |

External Python dependencies *not* in this repo but on the critical path:
`xdggs`, `xdggs-dggrid4py` (provides `IGEO7Index`, `Z7MonotonicIndex`, `mapblocks_regridding`,
`cell_ids2geographic`), `dggrid4py` + the DGGRID binary, `pyproj`, `numba`, `dask`.
See §7.3 for what each implies for a Julia port.

---

## 2. Layer 1 — Z7 index arithmetic

### 2.1 Bit layout (✅ `z7py/z7.py:7-17`, confirmed against the shipped archives)

A Z7 index is one `UInt64`:

```
bits 63..60   base cell id, 0..11              (4 bits)
bits 59..57   resolution digit 1               (3 bits, values 0..6, or 7 = "beyond resolution")
bits 56..54   resolution digit 2
...
bits  2..0    resolution digit 20
```

Digit accessor (1-based digit position `i ∈ 1..20`):

```
shift  = 57 - 3*(i-1)
digit  = (raw >> shift) & 0x07
```

`resolution(raw)` = position of the first digit equal to 7, minus 1 (20 if none).
`parent_at(raw, L)` = keep base cell + digits 1..L, set digits L+1..20 to 7.

✅ Verified on `data/working/pori_z7_r10.zarr`: every stored id has digit `L+1 == 7`
and digits `1..L ∈ 0..6`; all 3 101 ids are in base cell 0.

### 2.2 Monotonic (base-7) integer (✅ `z7py/z7.py:246-276`)

```
mono(raw, L) = base_cell
for i in 1..L:  mono = mono*7 + digit(raw, i)
```

with the exact inverse `monotonic_int_to_z7`.

**Range of values.** ✅ The monotonic space at level `L` has `12·7^L` slots; the largest
possible value is `12·7^20 = 957 507 195 571 344 012`, comfortably below
`typemax(Int64) = 9 223 372 036 854 775 807`. So monotonic ids **fit in `Int64`**, which
makes signed gap arithmetic (`diff(mono) != 1`) safe — the Python code relies on exactly
this (`z7_zarr.py:97`, `.astype(np.int64)`). Packed Z7 ids do **not** fit in `Int64` (§6.4).

**Density.** ✅ The monotonic space is *not* fully dense. Cell counts follow
`cells(L) = 10·7^L + 2` (verified against all 21 rows of `RESOLUTION_STATS`), while the
monotonic space is `12·7^L`, so only **5/6 ≈ 83.3 %** of monotonic ids correspond to a real
cell. The gaps come from the 12 pentagonal base cells having 6 children instead of 7.
This does **not** break the range index — ranges are derived from data that actually exists,
so a phantom id can never fall *inside* a range (it would split the range instead). But a
Julia port must not assume "a parent at level `L−k` covers exactly `7^k` consecutive
monotonic ids" near pentagons, and must not feed arbitrary monotonic ids to
`monotonic_int_to_z7` and expect a valid cell back.

### 2.3 Sort-order invariant (✅ verified empirically — important simplification)

At a **fixed** resolution `L`, the padding digits are a constant suffix and every digit
occupies its own 3-bit field, so lexicographic order on `(base_cell, d₁…d_L)` equals both
the numeric order of the packed `UInt64` **and** the numeric order of the monotonic int.

Verified on both `pori_z7_r10.zarr` (N = 3 101) and `pori_z7_r12.zarr` (N = 158 430):
`argsort(packed) == argsort(monotonic)`, and both are already strictly increasing.

Consequences for Julia:

- Sorting can be done directly on the raw `Vector{UInt64}` — no base-7 conversion needed.
- `searchsortedfirst` for cell lookup works on raw packed ids (this is what
  `neighbours.py:42` does), so the *lookup* path never needs the monotonic transform.
- The monotonic int is needed only for **contiguity**: gap detection when building ranges,
  and `O(1)` offset arithmetic inside a range. Do not conflate the two roles.
- ⚠️ The invariant holds **only** for a single-resolution dataset. Mixed-resolution
  collections break it (and monotonic values collide across resolutions).

### 2.4 GBT neighbour computation (📄 `z7py/z7.py:384-558`)

- 6 directions, `d = 1..6`, digit-wise GBT addition starting at the **finest** digit
  (position = resolution) and cascading carries towards coarser digits.
- Rotation parity: **odd resolution → CW tables, even resolution → CCW tables**
  (`is_even = (resolution % 2 == 0)`), matching the archive attribute
  `rotation_pattern = "alternating_cw_odd_ccw_even"`.
- Four 7×7 `UInt8` tables: `_GBT_CW_0/_GBT_CW_1` and `_GBT_CCW_0/_GBT_CCW_1`
  (`*_0` = result digit, `*_1` = carry digit). Total 4 × 49 = 196 bytes.
- A carry surviving past digit 1 means the neighbour crosses into another **base cell**:
  `_BASE_CELL_NEIGHBOURS_RAW[12,6]`, then rotation correction via `_ROTATIONS[12]` and,
  for polar base cells 0 and 11, `_POLE_0_ROTATIONS[6,6]`.
- Rotations are applied as digit-wise `(digit * mult) % 7`, where `mult` is `5^rots mod 7`.
- Pentagon handling: `_EXCLUDE_NEIGHBOURS[12]` (1-based slot index, value 2 for base
  cells 0–5 and 5 for base cells 6–11); the excluded slot is set to
  `INVALID = 0xFFFF_FFFF_FFFF_FFFF = typemax(UInt64)`.
- Complexity `O(r)` per neighbour, `O(6r)` per cell.

### 2.5 Indexing conventions — porting trap

The tables mix two different indexing bases, and the Python code already carries the
Julia-side comments:

| Table | Indexed by | Python | Julia |
|---|---|---|---|
| `_GBT_CW_0/1`, `_GBT_CCW_0/1` | digit **values** 0…6 | `tbl[a, b]` | `tbl[a+1, b+1]` (or `OffsetArray`) |
| `_BASE_CELL_NEIGHBOURS_RAW` | base cell 0…11, slot 0…5 | `bcn[bc][i]` | `bcn[bc+1, i]` — note the code's own comment "(Julia 1-indexed, Python 0-indexed)" |
| `_EXCLUDE_NEIGHBOURS` | base cell 0…11 → **1-based** slot | `excl[bc]`, compared as `i+1 == exclusion` | already 1-based; comment "1-based (Julia style)" |
| `_POLE_0_ROTATIONS` | digit values 1…6 | `pole0[row-1, col-1]` | `pole0[row, col]` |
| digit position `i` | 1…20 | already 1-based | unchanged |
| direction `d` | 1…6 | already 1-based | unchanged |

In Julia the 7×7 tables are best expressed as `SMatrix{7,7,UInt8}` constants (compiled
into the kernel, GPU-friendly, no global-array capture).

### 2.6 Evidence that a Julia reference already exists (✅)

```
z7py/tests/test_z7.py:3   "Mirrors the reference test suite in z7jl/tests_z7.jl."
z7py/latitudes.py:3       "Translated from z7jl/AuthalicLatitude.jl."
z7py/latitudes.py:18      "These are the same as in the Julia code, 6x6 matrix."
z7py/latitudes.py:100     "etc[0] in Julia corresponds to normalized_meridian_arc_unit"
z7py/z7.py:130            "(Julia 1-indexed, Python 0-indexed)"
z7py/z7.py:146            "Exclusion index (1-based in Julia → 0-based here)"
z7py/z7.py:368            "Equivalent to Julia's leading_zeros-based implementation"
z7py/z7.py:454            "exclusion = int(excl[base_cell])  # 1-based (Julia style)"
```

**Action:** before writing any Julia GBT code, pull `z7jl` (`github.com/wrenoud/Z7`),
check which of `z7_to_monotonic_int` / `monotonic_int_to_z7` / `get_parent_at` already
exist there, and treat the Python module as the *specification of the delta* rather than
the source. Note `first_non_zero` in Python is a linear digit scan explicitly documented
as equivalent to a `leading_zeros`-based Julia one-liner — prefer the Julia form.

---

## 3. Layer 2 — Monotonic index and the Zarr storage layout

### 3.1 Two archive flavours (✅ `z7_zarr.py`, confirmed on disk)

| | `compression = "none"` | `compression = "ranges"` |
|---|---|---|
| Coordinate variable | `cell_ids`, shape `(N,)`, `<u8` | `cell_id_ranges`, shape `(R, 2)`, `<u8` |
| Coordinate dims | `["cell_ids"]` | `["ranges", "bounds"]` |
| Data variables | on dim `cell_ids`, length `N` | on dim `cell_ids`, length `N` (dim has **no** coord array on disk) |
| Stored values | every cell id | packed Z7 **start/end pairs**, inclusive |
| Python index | `xdggs` → `IGEO7Index` (eager `PandasIndex`) | `Z7MonotonicIndex` (lazy) |
| Group attr | `dggs.coordinate = "cell_ids"` | `dggs.coordinate = "cell_id_ranges"`, plus root `coordinates: "cell_id_ranges"` |

Both keep `dggs.spatial_dimension = "cell_ids"` — i.e. **the data dimension is always
named `cell_ids`**, even in the ranges flavour where no `cell_ids` array exists on disk.
This is deliberate (`z7_zarr.py:36-48`): the dim name equals the coord name so that
`xdggs.decode` yields a single indexed dim coordinate.

> ⚠️ Note the range table stores **packed Z7 ids**, not monotonic ints. A reader must
> convert `start`/`end` to monotonic ints itself to compute lengths and offsets.

### 3.2 Exact on-disk metadata (✅ read from `data/working/`)

Group `.zgroup`: `{"zarr_format": 2}`. No `.zmetadata` — archives are written with
`consolidated=False` (`z7_zarr.py:250, 312`).

Group `.zattrs` (r12 ranges archive, verbatim structure):

```json
{
  "clipper_scale_factor": 100000000,
  "coordinates": "cell_id_ranges",
  "dggs": {
    "compression": "ranges",
    "coordinate": "cell_id_ranges",
    "dggs_vert0_azimuth": 0.0,
    "dggs_vert0_lat": 58.28252559,
    "dggs_vert0_lon": 11.2,
    "ellipsoid": {"inverse_flattening": 298.257223563,
                  "name": "wgs84", "semimajor_axis": 6378137.0},
    "name": "igeo7",
    "refinement_level": 12,
    "rotation_pattern": "alternating_cw_odd_ccw_even",
    "spatial_dimension": "cell_ids"
  },
  "regridder": "xdggs_dggrid4py.mapblocks_nearestcentroid",
  "source_crs": "EPSG:3301",
  "source_path": "...",
  "zarr_conventions": [{
    "description": "Discrete Global Grid Systems convention for zarr",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38"
  }]
}
```

Per-array `.zattrs` for the coordinate additionally carry the **xdggs decode block**
(`z7_zarr.py:138-155`) — these are separate from the group-level `dggs` block and both
must be written:

```json
{"_ARRAY_DIMENSIONS": ["ranges", "bounds"],
 "grid_name": "igeo7", "level": 12,
 "igeo7_dggs_vert0_lon": 11.2, "igeo7_wgs84_geodetic_conversion": true}
```

Array `.zarray` (r12):

| array | dtype | shape | chunks | fill_value | compressor |
|---|---|---|---|---|---|
| `cell_ids` (dense archive) | `<u8` | `[158430]` | `[39608]` | `null` | blosc/zstd, clevel 3, shuffle 1, blocksize 0 |
| `cell_id_ranges` (ranges archive) | `<u8` | `[1073, 2]` | `[1073, 2]` | `null` | blosc/zstd, clevel 3, shuffle 1 |
| `elevation` | `<f4` | `[158430]` | `[16807]` | `"NaN"` | blosc/zstd, clevel 3, shuffle 1 |
| `slope_lookup`, `slope_geodesic` (r10) | `<f4` | `[3101]` | `[3101]` | `0.0` | blosc/lz4, clevel 5, shuffle 1 |

All arrays: `"order": "C"`, `"filters": null`, `"zarr_format": 2`.

Notes:
- The two r10 archives were written with the **older default** (blosc/**lz4** clevel 5);
  r12 uses the project default `numcodecs.Blosc(cname="zstd", clevel=3, shuffle=SHUFFLE)`
  (`config.py:80-94`). A Julia reader must handle both cnames.
- `fill_value` is `"NaN"` (a JSON *string*) for `<f4` elevation, `0.0` for the slope
  variables, and `null` for `<u8`. ⚠️ Confirm `Zarr.jl` decodes the `"NaN"` string form.
- ✅ **Coordinate and data chunkings differ** in the dense r12 archive: `cell_ids` is
  chunked `39608` (4 chunks) while `elevation` is chunked `16807` (10 chunks). This is the
  exact mismatch that forced `slope_blocked()` away from `xr.map_blocks`
  (`slope.py:283-286`). A Julia chunk loop must derive its blocking from the **data**
  array's chunks and never assume the coordinate agrees.

### 3.3 Range construction semantics (✅ `z7_zarr.py:74-103`, verified by reconstruction)

```
mono          = monotonic(cell_ids, level)           # requires ascending order
assert all(diff(mono) >= 0)                          # raises otherwise
gap_idx       = findall(diff(mono) != 1)
starts_idx    = [1; gap_idx .+ 1]
ends_idx      = [gap_idx; N]
range_start   = cell_ids[starts_idx]                 # packed Z7, not monotonic
range_end     = cell_ids[ends_idx]                   # inclusive
```

✅ Verified on `pori_z7_r10_ranges.zarr`: R = 136, `Σ (mono_end − mono_start + 1) = 3101`,
and expanding every range reproduces the dense `cell_ids` of `pori_z7_r10.zarr` exactly.

Measured compression:

| archive | level | N | R | R/N | range len min/mean/max |
|---|---|---|---|---|---|
| `pori_z7_r10*` | 10 | 3 101 | 136 | 4.39 % | 1 / 22.8 / 739 |
| `pori_z7_r12*` | 12 | 158 430 | 1 073 | 0.68 % | — |

Coordinate bytes on disk (chunk payload only), r12:
dense `cell_ids` **7 750 B** (uncompressed 1 267 440 B) → ranges `cell_id_ranges`
**1 996 B** (uncompressed 17 168 B). ✅

**Reader algorithm** a Julia `Z7MonotonicIndex` equivalent must implement:

```
ms[k] = mono(range_start[k]); me[k] = mono(range_end[k])
len[k] = me[k] - ms[k] + 1
off    = [0; cumsum(len)[1:end-1]]          # 0-based position of each range's first cell
N      = sum(len)                            # must equal the data variable length

# label -> position  (the "sel" path), O(log R)
k = searchsortedlast(ms, mono(target))
pos = (ms[k] <= mono(target) <= me[k]) ? off[k] + (mono(target) - ms[k]) : missing

# position -> label  (the "isel" / materialise path), O(log R) per element
k = searchsortedlast(off, p)
label = monotonic_to_z7(ms[k] + (p - off[k]), level)
```

Both directions are exercised by the round-trip assertions in
`scripts/demo_regrid_pori_to_z7.py:226-240` (`sel` positions and full `values()`
reconstruction) — reuse those as the Julia acceptance test.

### 3.4 Chunking policy (📄 `config.py:75-77`, `z7_slope_blocked_halo.ipynb` cell 17)

- `DEFAULT_CHUNK_CELLS = 16 807 = 7^5`.
- Chunk sizes should be **powers of 7** so a chunk covers a whole GBT subtree, which is
  spatially compact and therefore minimises the halo/interior ratio.
- Documented halo estimates (∝ √N): 7^5 → ~490 cells (~2.9 %); 7^6 = 117 649 → ~1 300
  (~1.1 %); 7^7 = 823 543 → ~3 400 (~0.4 %).
- **Recommended: 7^6 = 117 649 cells** (~461 KB at `Float32`, fits L2/L3 per worker;
  ~235 K chunks for a global r12 grid). The r12 notebook run uses exactly this.
- The abstract's stated target is 10–100 MB per chunk with boundaries aligned to parent
  zones at level `N−k`, `k ∈ {2,3,4,5}` — note `7^k` cells == the descendants of one
  parent at level `L−k`, which is the same statement.

---

## 4. Layer 3 — the slope kernel (GBT focal operation)

### 4.1 Axis labelling and the FDA formula (📄 `kernels/slope.py:1-23`, settled by `phase3a_slope_math.py`)

z7py canonical directions → GBT axes (this is the labelling that *works*; the paper's
naive `(1,4)` pairing is explicitly shown to be wrong in `phase3a_slope_math.py` Variant 1):

```
d=1 (col 1): k+     d=6 (col 6): k-
d=2 (col 2): j+     d=5 (col 5): j-
d=4 (col 4): i+     d=3 (col 3): i-
```

(columns given 1-based for Julia; Python uses 0…5.)

```
Δh[c]  = h[neighbour_c] - h[self]                       for c = 1..6
d_y    = (d_j + d_i) / 2
∂h/∂x  = (Δh[1] - Δh[6]) / (2 d_k)
∂h/∂y  = (Δh[4] + Δh[5] - Δh[2] - Δh[3]) / (2 √3 d_y)
slope  = hypot(∂h/∂x, ∂h/∂y)                            [m/m]
```

The k axis is treated as local *x*; j and i together span *y*. Antipodality of the three
axis pairs is parity-invariant even though the absolute bearings rotate by ~19.1° between
odd and even resolutions.

Masking: any cell with **any** neighbour outside the AOI (`pos < 0`), including pentagon
`INVALID` slots, is set to `NaN` (`slope.py:141-149`). Output attrs: `units = "m/m"`,
`long_name = "slope magnitude (FDA)"`.

### 4.2 Neighbour table and position projection (📄 `kernels/neighbours.py`)

```
nbrs[i, :] = get_neighbours(cell_ids[i])                # (N,6) UInt64, INVALID in pentagon slots
pos        = searchsorted(cell_ids, nbrs)               # exact-match check, else -1
boundary   = any(pos .< 0, dims=2)
```

Because `cell_ids` is sorted (§2.3) and `INVALID = typemax(UInt64)` sorts last, the
sentinel needs no special case — it simply fails the exact-match test.

Julia notes:
- Store the table as **`(6, N)`** (`Matrix{UInt64}`), not `(N, 6)`: column-major makes a
  cell's six neighbours contiguous, which is what both the CPU cache and a GPU thread want.
- `searchsortedfirst(cell_ids, x)` per element is `O(6N log N)`. Since the neighbour ids
  per cell are *not* sorted, a merge-scan is not directly available; but see below.
- **Better:** with a range table in hand (§3.3) the position lookup is `O(log R)` with
  `R ≪ N` — replace the `searchsorted` over N ids by `searchsortedlast` over R range
  starts plus a subtraction. This is the main algorithmic payoff of the monotonic index
  in the kernel path, and the Python code does **not** yet exploit it inside the kernel
  (it still calls `np.searchsorted` on the materialised ids, `neighbours.py:42`).

### 4.3 Distances (📄 `slope.py:50-117`, `distance_measures.py`)

Two modes:

**`lookup`** (production). Per cell, take its **level-4** parent (`_PARENT_LEVEL = 4`),
look it up in the global anisotropy table, and scale:

```
base_d = cls_m(level) * sqrt(π / (2√3))          # ≈ cls_m * 0.95231
d_k    = base_d * mean_d_k / mean_d_geom          # per level-4 parent
d_j    = base_d * mean_d_j / mean_d_geom
d_i    = base_d * mean_d_i / mean_d_geom
```

✅ `data/working/dist_lookup_level4.parquet`: 24 012 rows (= all level-4 cells), columns
`n, mean_d_k, mean_d_j, mean_d_i, mean_d_geom`, keyed by a `UInt64` `cell_ids` column
(the packed level-4 parent id). Julia: read with `Parquet2.jl`, sort by key once, and
resolve with `searchsortedfirst` rather than a `Dict` (branch-free, GPU-transferable).
Cost is `O(unique parents)`, effectively `O(1)` per cell for any sub-face AOI.

**`geodesic`** (validation). Per cell × direction, `pyproj.Geod(ellps="WGS84").inv`
between centroids, then collapse to per-axis means of the two antipodal ends. Requires
DGGRID centroids for all unique neighbour ids. Julia equivalent: `Geodesy.jl` /
`Proj.jl`; keep on CPU. Documented divergence between the two modes is ≈ **1.86 %**
over the AOI (README abstract).

`cls_m` per level comes from `RESOLUTION_STATS` (`z7py/z7.py:29-51`), e.g.
level 10 → 479.4882 m, level 12 → 68.4983 m.

### 4.4 Execution shape

`slope()` is eager and index-agnostic once `pos`, `boundary` and `d_*` exist — pure
gather + arithmetic over N. That is the KernelAbstractions target. Stage split for Julia:

| stage | shape | character | backend |
|---|---|---|---|
| 1. neighbour table | N → (6,N) `UInt64` | integer, table-driven, data-dependent carry loop (≤ 20 iters), rare pentagon/rotation branches | KA, any |
| 2. positions | (6,N) → (6,N) `Int32` | binary search, `O(log R)` with range table | KA, any |
| 3. axis distances | N → 3×N `Float64` | gather from 24 012-row table via binary search on parent id | KA, any |
| 4. FDA | (6,N)+N → N | elementwise, ~10 flops/cell | KA, any |
| 5. geodesic distances *(optional)* | N×6 | `Geod.inv`, iterative | CPU only |

Stage 1 dominates; stages 2–4 are memory-bound. Stages 1–4 are fusible into one kernel
(no intermediate `(6,N)` materialisation) if memory traffic matters, at the cost of
testability — recommend keeping them separate first and fusing only if measured.

### 4.5 Halo strategy for chunked execution (📄 `kernels/halo.py`, `slope.py:275-386`)

Per chunk, computed **eagerly and cheaply** (only `UInt64` ids are touched — never the
data):

1. `nbrs = neighbours(chunk_ids)`; `pos_local = searchsorted(chunk_ids, nbrs)`.
2. `outside = (pos_local < 0) & (nbrs != INVALID)` → `halo_candidates = unique(nbrs[outside])`.
3. Keep only candidates that exist in the full AOI (`searchsorted` against all ids)
   → `halo_indices` (integer positions into the data array).
4. At compute time: read `core = data[start:end]` and `halo = data[halo_indices]`,
   concatenate, **sort by cell id**, carry an `owned_mask`, run the kernel on the combined
   array, return `result[owned_mask]`.

Key property: AOI-boundary cells stay `NaN`; chunk-interior boundary "NaN bleed" is
eliminated. `HaloChunk` keeps the combined array sorted precisely so that the same
`searchsorted` position projection works unchanged.

Julia mapping:
- The halo index set is **irregular in index space**, so `YAXArrays`' `MovingWindow` /
  a rectangular-window abstraction does **not** apply. Write the chunk loop manually over
  `DiskArrays.eachchunk(data)`.
- Step 4's `data[halo_indices]` is scattered disk reads. ⚠️ Check the `DiskArrays.jl`
  batching strategy in use (it groups scattered indices into chunk-aligned reads); if
  scattered reads degrade, coalesce `halo_indices` into contiguous runs first — with
  monotonic chunking the halo clusters near the chunk edges, so runs are few.
- Python's `slope_blocked` deliberately avoids `xr.map_blocks` and hand-rolls
  `dask.delayed` because the coordinate and data chunkings disagree (§3.2). Julia has the
  same hazard with `mapCube`; the manual loop sidesteps it.

---

## 5. Julia package mapping

### 5.1 Overview

| Python | Julia | Confidence |
|---|---|---|
| `numpy` arrays of `uint64/float32` | `Vector{UInt64}` / `Vector{Float32}` | ✅ direct |
| `numba @njit` scalar kernels | plain Julia functions (already fast) | ✅ direct |
| `numba @njit(parallel=True)` batches | `Threads.@threads` or `KernelAbstractions.@kernel` on `CPU()` | ⚠️ |
| `zarr-python` v2 store | `Zarr.jl` (`zopen`, `zgroup`, `zcreate`) | ⚠️ v2 native; check v3 status |
| lazy chunked array | `DiskArrays.jl` (`Zarr.ZArray <: AbstractDiskArray`) | ⚠️ verify subtype |
| `xarray.Dataset`/`DataArray` | `YAXArrays.jl` `Dataset`/`YAXArray` over `DimensionalData.jl` | ⚠️ |
| labelled dims + `.sel` | `DimensionalData.Dim{:cell_ids}` + `At`/`Near`/`..` selectors | ⚠️ |
| custom `xarray.indexes.Index` | custom `DimensionalData.Lookups.Lookup` subtype | ⚠️ §5.3 |
| `dask.array` / `dask.delayed` | `Dagger.jl`, or a plain threaded chunk loop | ⚠️ |
| `pyproj.Geod.inv` | `Geodesy.jl` / `Proj.jl` | ⚠️ |
| `pandas.read_parquet` | `Parquet2.jl` / `Arrow.jl` + `DataFrames.jl` | ⚠️ |
| `numcodecs.Blosc` | `Zarr.jl`'s `BloscCompressor` (Blosc.jl / `Blosc_jll`) | ⚠️ zstd availability |
| GPU offload (none in Python) | `KernelAbstractions.jl` + CUDA/ROCm/Metal/oneAPI | ⚠️ §5.5 |

### 5.2 `Zarr.jl` + `DiskArrays.jl`

- Zarr v2 with `blosc` + `_ARRAY_DIMENSIONS` is `Zarr.jl`'s home ground; nothing in these
  archives uses filters, object dtypes, groups-of-groups, or consolidated metadata.
- `Zarr.ZArray` is a `DiskArrays.AbstractDiskArray`, so `eachchunk`, lazy `getindex`, and
  lazy broadcasting come for free. ⚠️ Confirm the version's `eachchunk` returns the
  on-disk `GridChunks` (needed for the chunk loop in §4.5).
- ⚠️ **Dimension order.** Julia array libraries that wrap C-order formats conventionally
  present dimensions reversed relative to the JSON metadata (column-major). For the 1-D
  arrays here this is a no-op; for the 2-D `cell_id_ranges` a `(R, 2)` Python array is
  expected to appear as `(2, R)` in Julia, with `_ARRAY_DIMENSIONS`
  `["ranges","bounds"]` needing reversal to `(:bounds, :ranges)`. This is the single most
  likely interop surprise — test it first (§8.2 T2). A `(2, R)` layout is actually the
  *better* Julia layout (start/end contiguous per range), so if the reversal happens,
  embrace it rather than transposing.
- Compressor: the `.zarray` blocks map 1:1 onto `Zarr.BloscCompressor(; cname, clevel,
  shuffle, blocksize)`. ⚠️ Verify the bundled `Blosc_jll` exposes **zstd** (needed for r12)
  and **lz4** (needed for r10).
- ⚠️ `fill_value: "NaN"` (string) and `fill_value: null` must both decode.

### 5.3 `DimensionalData.jl` — the index layer

There is no plugin protocol to reimplement: xarray's custom-`Index` machinery
(`from_variables`, `create_variables`, `sel`, `isel`, `Coordinates.from_xindex`) collapses
in Julia to *"provide a lookup vector, optionally a custom `Lookup` type"*.

**Dense archive (`compression="none"`).** Nothing custom needed:

```julia
dim = Dim{:cell_ids}(Sampled(cell_ids;                 # Vector{UInt64}, ascending
                             order = ForwardOrdered(),
                             span  = Irregular(),
                             sampling = Points()))
```

`At(id)` then resolves via `searchsortedfirst` in `O(log N)` — the same complexity as
`IGEO7Index`'s pandas index, without the eager `PandasIndex` construction cost that the
README abstract flags as a scaling limitation.

**Ranges archive (`compression="ranges"`).** Two options, in increasing order of payoff:

1. *Lazy vector.* Define `Z7MonotonicIds <: AbstractVector{UInt64}` holding
   `(ms, me, off, level)` and implementing `size` + `getindex` (§3.3 forward map). Wrap it
   in `Sampled(...; order=ForwardOrdered())`. `searchsortedfirst` works through
   `getindex` without materialising N ids — an `O(log N)` `At` with `O(R)` memory. This
   alone reproduces `Z7MonotonicIndex` behaviour.
2. *Custom lookup.* Subtype `DimensionalData.Lookups.Lookup` and specialise
   `Lookups.selectindices` / `Lookups.at` to do the `O(log R)` range search directly,
   skipping the `O(log N)` binary search over reconstructed ids, and to accept **vectors**
   of ids in one pass (what the halo path needs). ⚠️ The abstract type was renamed
   (`LookupArrays.LookupArray` → `Lookups.Lookup`) in recent DimensionalData — pin the
   version and check the interface methods.

Either way, `create_variables`-style behaviour (materialising a lazy `cell_ids` coord on
demand) is just `collect` on the lazy vector.

### 5.4 `YAXArrays.jl` — the dataset layer

- `open_dataset(zopen(path))` gives a `Dataset` whose axes come from `_ARRAY_DIMENSIONS`;
  `savedataset(ds; path, driver=:zarr)` writes them back. This is the xarray-compatible
  path and should be preferred over hand-rolling Zarr writes **for data variables**.
- ⚠️ The group-level `dggs` / `zarr_conventions` attributes and the per-array xdggs attrs
  are custom; confirm `YAXArrays` round-trips arbitrary nested attribute dicts, otherwise
  write them with `Zarr.jl` directly after `savedataset`.
- ⚠️ In the ranges flavour the data dimension `cell_ids` has **no** coordinate array on
  disk. Check how `YAXArrays` handles a dimension with no matching array (xarray copes
  because the index is attached after opening). Fall back to `Zarr.jl` + manual
  `DimensionalData` construction if it does not.
- `setchunks(cube, (cell_ids = 117649,))` for the §3.4 chunk policy.
- `mapCube(f, cube; indims=InDims("cell_ids"), outdims=OutDims("cell_ids"))` is the
  `map_blocks` analogue — usable for elementwise post-processing, **not** for the halo
  kernel (§4.5).

### 5.5 `KernelAbstractions.jl` — the compute layer

Facts that constrain the design:

- **No heap allocation inside kernels.** `z7py.get_neighbours` allocates `np.empty(6)`
  per call; the Julia kernel must write into a preallocated `(6,N)` output or use
  `MVector{6,UInt64}` / unrolled scalars.
- **Constant tables.** The four 7×7 `UInt8` GBT tables, `_BASE_CELL_NEIGHBOURS_RAW`
  (12×6), `_EXCLUDE_NEIGHBOURS` (12), `_ROTATIONS` (12), `_POLE_0_ROTATIONS` (6×6) total
  under 300 bytes. As `SMatrix`/`SVector` constants they are baked into the compiled
  kernel — no device-array plumbing, no `@localmem` needed.
- **Divergence.** The carry cascade is a bounded (`≤ resolution ≤ 20`) data-dependent loop;
  base-cell crossings and pentagon rotations are rare paths. On GPU this costs some
  divergence within a warp but the loop trip count is small and bounded — acceptable.
  Avoid early `return`s in favour of predicated writes where convenient.
- **`% 7`.** Used in the monotonic transform and in the digit rotations `(d*mult) % 7`.
  Integer division is slow on GPUs. `_MOD_7_TABLE` already exists for `a+b ≤ 13`; add a
  7×7 multiply-mod table for `(d*mult) % 7` (`d ≤ 6`, `mult ∈ {1,3,5}` in practice).
- **`Float64` on Metal.** `pixi.toml` declares `platforms = ["osx-arm64"]`, so the likely
  first GPU backend is Metal — which has **no `Float64`**. The FDA math is currently
  `float64` (`slope.py:197`), while the stored elevation is `float32`. Decide explicitly:
  `Float32` accumulation on Metal (fine for slope in m/m; validate against the tilted-plane
  test), `Float64` on CPU/CUDA. Keep the kernel generic in `T`.
- **`UInt64` on GPU.** Supported everywhere, but 64-bit integer ops are slower than 32-bit.
  The positions array can safely be `Int32` (N ≪ 2^31 per chunk).
- Launch shape: `kernel!(backend, 256)(args...; ndrange = N)` +
  `KernelAbstractions.synchronize(backend)`; `backend = get_backend(cell_ids)` so the same
  code runs on `CPU()`.
- ⚠️ The realistic verdict: for regional AOIs (Pori r12 = 158 430 cells) a threaded CPU
  loop will saturate; GPU pays off at global r12–r15. Write it with KA anyway so the
  backend is a parameter, but benchmark before claiming a GPU win.

Sketch (illustrative, not compiled):

```julia
@kernel function slope_fda!(out, @Const(h), @Const(pos), @Const(dk), @Const(dj), @Const(di))
    i = @index(Global, Linear)
    ok = true
    Δ = ntuple(6) do c
        p = pos[c, i]
        ok &= p > 0
        p > 0 ? h[p] - h[i] : zero(eltype(h))
    end
    dy   = (dj[i] + di[i]) / 2
    ∂x   = (Δ[1] - Δ[6]) / (2 * dk[i])
    ∂y   = (Δ[4] + Δ[5] - Δ[2] - Δ[3]) / (2 * sqrt(3f0) * dy)
    out[i] = ok ? hypot(∂x, ∂y) : NaN
end
```

---

## 6. Read/write interop contract (Julia ↔ Python)

### 6.1 Reading a Python-written archive in Julia — checklist

1. `.zgroup` → `zarr_format == 2`.
2. Group `.zattrs.dggs.compression` ∈ `{"none", "ranges"}` → choose the index path.
3. `dggs.refinement_level` → the `level` used by every monotonic conversion.
4. `dggs.spatial_dimension` → the data dimension name (`"cell_ids"`).
5. Coordinate array = `dggs.coordinate` (`"cell_ids"` or `"cell_id_ranges"`).
6. Decode dtype `<u8` as `UInt64` (never `Int64`, §6.4).
7. Build the dims from each array's `_ARRAY_DIMENSIONS`, reversing if the reader reverses
   shapes (§5.2).
8. Do **not** expect `.zmetadata`; do **not** require consolidated metadata.
9. Tolerate coordinate/data chunk mismatch (§3.2).

### 6.2 Writing from Julia so Python/xarray can read it — checklist

1. `zarr_format: 2`, one `.zgroup` at the root.
2. `_ARRAY_DIMENSIONS` on **every** array (xarray refuses the dataset otherwise), in
   C order (slowest dimension first) — reverse the Julia dim order when writing.
3. Reproduce the group `dggs` block **and** the per-coordinate xdggs attrs
   (`grid_name`, `level`, `igeo7_dggs_vert0_lon`, `igeo7_wgs84_geodetic_conversion`)
   — both are required; `xdggs.decode` reads the latter, the dggs convention reads
   the former (`z7_zarr.py:138-155`).
4. `zarr_conventions` list with the registration block from `config.py:60-66`
   (keep the `uuid` verbatim).
5. Compressor `blosc/zstd, clevel=3, shuffle=1 (byte shuffle), blocksize=0` to match
   `config.py:default_compressor()`.
6. `fill_value`: `null` for `<u8`, `"NaN"` for float arrays that carry missing data.
7. Chunk keys with `.` separator (Zarr v2 default flat encoding — the shipped archives use
   `0`, `1`, … and `0.0`), not the nested `/` layout.
8. Keep `cell_ids` **sorted ascending** — every downstream assumption depends on it
   (§2.3, `z7_zarr.py:98-99` raises otherwise).
9. Dense flavour: emit `cell_ids` on dim `cell_ids`. Ranges flavour: emit
   `cell_id_ranges` on dims `("ranges","bounds")` shaped `(R,2)` **in the JSON**, set the
   root `coordinates: "cell_id_ranges"` attribute, and emit no `cell_ids` array.
10. Write the data variables on dim `cell_ids` with length `N = Σ range lengths`.

### 6.3 Round-trip test to run first

Open `data/working/pori_z7_r12_ranges.zarr` in Julia, expand the ranges, and assert the
result equals the dense `cell_ids` of `data/working/pori_z7_r12.zarr` (this exact
equivalence was ✅ verified for the r10 pair during this analysis). Then write the same
dataset back out from Julia to a new path and re-open it with
`z7_xarray_paper.z7_zarr.open_dataset(..., decode=True)`.

### 6.4 The signedness trap (✅, and silent in the shipped fixtures)

Base cell occupies bits 63–60, so base cells **8–11** set bit 63 and the packed id exceeds
`typemax(Int64)` — read as `Int64` it becomes negative, and sorting/searchsorted break
globally. All shipped Pori fixtures live in **base cell 0** (verified), so this bug would
pass every test in this repo and only appear on other AOIs. Use `UInt64` end to end;
if a Parquet/Arrow column arrives as signed, `reinterpret(UInt64, x)` rather than convert.

Related: `INVALID = typemax(UInt64)` is `-1` as `Int64`, which would sort *first* instead
of last and silently corrupt the neighbour position projection.

---

## 7. Suggested Julia layout

### 7.1 Packages

```
Z7.jl            # upstream (wrenoud/Z7 / z7jl) — bit layout, GBT neighbours, authalic lat
Z7Zarr.jl        # NEW: archive read/write, dggs convention metadata, range table
Z7Index.jl       # NEW: Z7MonotonicIds lazy vector + DimensionalData lookup
Z7Kernels.jl     # NEW: neighbour table, halo chunking, FDA slope via KernelAbstractions
```

`Z7Zarr.jl` and `Z7Index.jl` could reasonably be one package; keep `Z7Kernels.jl` separate
so the storage layer does not pull in GPU dependencies (use package extensions for the
CUDA/Metal/ROCm backends).

### 7.2 API sketch, mirroring the Python entry points

| Python | Julia |
|---|---|
| `z7_zarr.write(path; cell_ids, data, level, compression, …)` | `Z7Zarr.write(path; cell_ids, data, level, compression=:none, chunk_cells=7^5)` |
| `z7_zarr.open_dataset(path; decode=True)` | `Z7Zarr.open_dataset(path; decode=true) -> YAXArrays.Dataset` |
| `z7_zarr.find_monotonic_ranges(ids, level)` | `Z7Zarr.monotonic_ranges(ids, level) -> (starts, ends)` |
| `kernels.neighbours.get_neighbours_batch` | `Z7Kernels.neighbours!(out::Matrix{UInt64}, ids)` → `(6,N)` |
| `kernels.neighbours.neighbour_positions` | `Z7Kernels.positions!(pos, ids, nbrs)` (or range-table variant) |
| `kernels.slope.slope(da, model; distance_mode)` | `Z7Kernels.slope(cube, model; distance_mode=:lookup, backend=CPU())` |
| `kernels.slope.slope_blocked(da, model; …)` | `Z7Kernels.slope_blocked(cube, model; chunk=7^6, backend=CPU())` |
| `HexGridDistortionModel` | `Z7Kernels.DistortionModel` (sorted parent keys + weights) |

### 7.3 What cannot (yet) be done in Julia

- **Grid generation / regridding.** `mapblocks_regridding`, `cell_ids2geographic` and
  `grid_cell_centroids_for_extent` come from `dggrid4py`/`xdggs-dggrid4py` driving the
  **DGGRID** binary. A Julia port would need either a `DGGRID_jll`-style wrapper or a
  shell-out + file exchange (DGGRID communicates via metafiles and GDAL-readable outputs).
  **Recommendation: out of scope for the first Julia pass.** Consume the Zarr archives
  Python produced; that is exactly what read compatibility buys.
- **Centroid lookup** is needed only by `distance_mode=:geodesic`. Keep the Julia port on
  `:lookup` (the parquet table is self-contained) and treat geodesic validation as a
  Python-side task, or precompute centroids into the archive.
- **`xdggs` plotting/`explore`** has no Julia equivalent; not on the critical path.

---

## 8. Verification matrix

### 8.1 Index-arithmetic golden vectors (from `z7py/tests/test_z7.py`, mirrored from `z7jl`)

| case | input | expected |
|---|---|---|
| hex decode | `"004291d4c313ffff"` | base 0, digits `[0,1,0,2,4,4,3,5,2,3,0,3,0,4]`, then 7s; resolution 14 |
| string ↔ index | `"0800433"` | base 8, resolution 5, digits 0,0,4,3,3 |
| monotonic | `"00"` @ L0 | 0, round-trips |
| monotonic contiguity | `"050"`,`"051"` @ L1 | `mono("051") == mono("050") + 1` |
| `first_non_zero` | `"0000000"`,`"1234000"`,`"1200567"`,`"12"` | 6, 1, 3, 0 |
| GBT CW add | `(1,1)`,`(1,2)` | `(carry,digit) = (1,4)`, `(0,3)` |
| GBT CCW add | `(1,1)`,`(1,2)` | `(1,2)`, `(0,3)` |
| base-cell nbrs | bc 0 / 1 / 6 / 11 | `[5,4,2,1,3]` / `[5,0,6,10,2]` / `[10,2,1,11,7]` / `[9,6,10,8,7]` |
| single nbr | `"0103"` dir 3 / dir 5 | `"0136"` carry 0 / `"0101"` carry 0 |
| multi-level carry | `"0166"` dir 6 | carry 6, digit1 = 3, digit2 = 5 |
| all nbrs | `"0103"` | `{"0106","0161","0136","0134","0101","0100"}` |
| all nbrs | `"0132"` | `{"0163","0055","0051","0133","0130","0136"}` |
| pentagon | `"000"` | slot 2 (1-based) is `typemax(UInt64)`; 5 valid: `{"001","003","004","005","006"}` |
| cell counts | any L | `cells(L) == 10·7^L + 2` ✅ |

### 8.2 Storage/interop tests

| id | test | expected |
|---|---|---|
| T1 | Open all four `data/working/*.zarr` in Julia | no error; correct shapes/dtypes |
| T2 | Shape of `cell_id_ranges` as seen in Julia | `(1073,2)` or `(2,1073)` — **record which**, then fix the convention (§5.2) |
| T3 | Expand r10 ranges → compare to dense r10 `cell_ids` | bitwise equal (✅ in Python) |
| T4 | Expand r12 ranges → compare to dense r12 `cell_ids` | bitwise equal |
| T5 | `issorted(cell_ids)` and `argsort(packed) == argsort(mono)` | true (✅ both archives) |
| T6 | R/N counts | r10: N 3 101 / R 136; r12: N 158 430 / R 1 073 (✅) |
| T7 | Decode blosc **lz4** (r10) and **zstd** (r12) | both succeed |
| T8 | `fill_value` `"NaN"` and `null` | decode without error |
| T9 | Julia-written archive re-opened by `z7_zarr.open_dataset(decode=True)` | `IGEO7Index` / `Z7MonotonicIndex` attaches; group attrs intact |
| T10 | Julia-written archive: `sel` on 4 sample ids | positions `[0, N÷3, N÷2, N-1]` (mirrors `demo_regrid_pori_to_z7.py:231-237`) |

### 8.3 Kernel tests

| id | test | expected |
|---|---|---|
| K1 | Neighbour table vs `z7py` on all 158 430 r12 cells | bitwise equal `(6,N)` |
| K2 | Boundary mask on r10 | matches Python's `(pos < 0).any(axis=1)` |
| K3 | Synthetic tilted plane (`sx=0.05, sy=0.02`, closed form `0.0538516…`) | reproduce the relative-error statistics of `phase3b_slope_pori_r10.py` |
| K4 | Real r10 DEM slope, `distance_mode=:lookup` | matches the `slope_lookup` variable already stored in `pori_z7_r10.zarr` |
| K5 | Same, geodesic | matches the stored `slope_geodesic`; the two differ by ≈1.86 % |
| K6 | Blocked vs eager slope at r12, chunk 7^6 | identical except AOI-boundary NaNs |
| K7 | `Float32` vs `Float64` FDA (Metal readiness) | quantify the difference on K3/K4 before shipping a Metal path |

**K4/K5 are the strongest available oracle**: `data/working/pori_z7_r10.zarr` already
ships both slope variables (`<f4`, `units="m/m"`) computed by the Python kernel, so a
Julia implementation can be validated end to end without running any Python.

---

## 9. Open questions and risks

1. ⚠️ **`Zarr.jl` dimension reversal** for `cell_id_ranges` — decide the convention before
   writing anything (T2). Everything else here is 1-D and immune.
2. ⚠️ **`YAXArrays` with a coordinate-less dimension** (the ranges flavour). If it refuses,
   drop to `Zarr.jl` + hand-built `DimensionalData` dims.
3. ⚠️ **Nested attribute round-trip** (the `dggs` object, the `zarr_conventions` array of
   objects, the `ellipsoid` sub-object). Attribute loss would silently break `xdggs.decode`
   on the Python side.
4. ⚠️ **`Blosc_jll` codec coverage** — zstd *and* lz4 both required by the shipped fixtures.
5. ⚠️ **DimensionalData API churn** (`LookupArrays` → `Lookups`); pin versions in `Project.toml`.
6. **Pentagon coverage is untested by the fixtures.** All Pori cells are in base cell 0 and
   `phase3b` asserts zero pentagons in the AOI. Base-cell crossing, polar rotation and
   exclusion-zone code paths therefore have *no* regional fixture — use the synthetic
   golden vectors in §8.1 (`"0132"` crosses into base cell 0's neighbours; `"000"` is the
   pentagon case) and consider generating a small cross-face AOI fixture.
7. **Signedness (§6.4)** would pass all current tests and fail in production.
8. **Range-table position lookup in the kernel** (§4.2) is a genuine improvement over the
   Python implementation, not a port. Worth doing, worth benchmarking, worth stating as
   such in the paper rather than presenting as parity.
9. **`z7jl` scope is unknown from here.** Confirm whether it already has
   `z7_to_monotonic_int` / `monotonic_int_to_z7` / `get_parent_at`, or whether those were
   added on the Python side only (`z7py/README.md` presents the monotonic mapping as the
   `z7py` contribution, which suggests the latter).
10. **Chunk-size policy is documented but not measured.** The 7^5/7^6/7^7 halo estimates in
    the notebook are analytic (∝ √N), not benchmarked; a Julia port can cheaply produce
    the real numbers, since the halo manifest is computed from ids alone.

---

## 10. Reference constants (copy-ready)

```
IGEO7 / DGGRID metafile      dggs_vert0_lon      = 11.20
                             dggs_vert0_lat      = 58.28252559
                             dggs_vert0_azimuth  = 0.0
                             input/output HIERNDX, Z7, INT64
WGS84 ellipsoid              a = 6378137.0, 1/f = 298.257223563
INVALID sentinel             0xFFFF_FFFF_FFFF_FFFF  (typemax(UInt64))
Base-cell field              bits 63..60           (values 0..11)
Digit i field (1-based)      shift = 57 - 3*(i-1)  (3 bits, 0..6; 7 = padding)
Rotation parity              odd resolution → CW, even → CCW
Cell count                   cells(L) = 10·7^L + 2
Monotonic space              12·7^L  (density 5/6 ≈ 83.3 %)
Max monotonic value          12·7^20 = 957_507_195_571_344_012  < typemax(Int64)
Default chunk                7^5 = 16_807   (recommended 7^6 = 117_649)
Default compressor           blosc(cname="zstd", clevel=3, shuffle=1, blocksize=0)
Slope base distance factor   sqrt(π / (2√3)) ≈ 0.9523279
cls_m                        L10 = 479.4882 m,  L12 = 68.4983 m
Distortion table             data/working/dist_lookup_level4.parquet, 24_012 level-4 parents
dggs convention uuid         7b255807-140c-42ca-97f6-7a1cfecdbc38
```

---

## 11. References

- Repository code: `src/z7_xarray_paper/`, `z7py/`, `scripts/`, `data/working/*.zarr`.
- Zarr DGGS convention v1: <https://github.com/zarr-conventions/dggs/blob/v1/README.md>
- Z7 (Julia reference implementation): <https://github.com/wrenoud/Z7/>
- `dggrid4py`: <https://github.com/allixender/dggrid4py>
- `xdggs-dggrid4py`: <https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py/>
- Kmoch, A., Sahr, K., Chan, W. T., & Uuemaa, E. (2025). *IGEO7: A new hierarchically
  indexed hexagonal equal-area discrete global grid system.* AGILE: GIScience Series, 6(32).
- Sahr, K. (2019). *Central Place Indexing.* Cartographica 54(1).
- Li, M., McGrath, H., & Stefanakis, E. (2022). *Multi-resolution topographic analysis in
  hexagonal Discrete Global Grid Systems.* Int. J. Appl. Earth Obs. Geoinf. (the FDA slope).
- White, D., Kimerling, J. A., & Overton, S. W. (1992) — alternating CW/CCW GBT rotation.
