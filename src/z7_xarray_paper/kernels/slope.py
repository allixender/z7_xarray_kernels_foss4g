"""Z7-native slope (gradient magnitude) kernel.

Implements the FDA (finite-difference approximation) slope formula adapted
for the Z7 hexagonal DGGS, using z7py canonical axis labelling:

    d=1 (col 0): k+    d=6 (col 5): k-
    d=2 (col 1): j+    d=5 (col 4): j-
    d=4 (col 3): i+    d=3 (col 2): i-

k axis maps to the local "x" direction; j and i axes together span the
perpendicular "y":

    ∂h/∂x = (h[k+] − h[k−]) / (2 d_k)
    ∂h/∂y = (h[i+] + h[j−] − h[j+] − h[i−]) / (2√3 d_y),  d_y = (d_j+d_i)/2
    slope  = √((∂h/∂x)² + (∂h/∂y)²)   [m/m]

Two distance modes are supported (see `slope()` docstring).

Two execution paths are provided:
  slope()         — eager, in-memory, works on any IGEO7/Z7MonotonicIndex-backed DA.
  slope_blocked() — lazy, dask map_blocks, fetches 1-ring halo per chunk via
                    Z7MonotonicIndex.sel (read-time halo strategy, PHASE3.md §3.2).
"""

from __future__ import annotations

from pathlib import Path

import numba as nb
import numpy as np
import xarray as xr
from pyproj import Geod

from z7py import z7
from z7py.z7 import RESOLUTION_STATS

from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7_xarray_paper.kernels.neighbours import (
    INVALID,
    get_neighbours_batch,
    neighbour_positions,
)
from z7_xarray_paper.kernels.halo import HaloChunk, build_halo_chunk

_PARENT_LEVEL = 4
_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# distance helpers — both return (d_k, d_j, d_i) arrays of shape (N,) in metres
# ---------------------------------------------------------------------------

@nb.njit(cache=True, parallel=True, nogil=True)
def _parent_at_batch(cell_ids: np.ndarray, resolution: int) -> np.ndarray:
    """Thread-parallel level-`resolution` ancestor for a batch of cells.

    z7py ships only the scalar `get_parent_at`; driving it from a Python
    generator is the same pathology the paper documents for `get_neighbours`
    (Section 5.1) — ~2.0M cells/s and GIL-bound, versus ~470M cells/s here,
    a 233x speed-up measured bit-identical at Pori r12. `nogil=True` matters
    as much as the speed: it lets `slope_blocked`'s dask threads actually run
    concurrently instead of serialising on the interpreter lock.
    """
    out = np.empty(cell_ids.shape[0], dtype=np.uint64)
    for i in nb.prange(cell_ids.shape[0]):
        out[i] = z7.get_parent_at(cell_ids[i], resolution)
    return out


def _distances_lookup(
    cell_ids: np.ndarray,
    level: int,
    model: HexGridDistortionModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell axis distances via level-4 parent lookup.

    O(unique level-4 parents) DGGRID calls — effectively O(1) per cell for
    any AOI smaller than a full icosahedron face.
    """
    cls_m = float(RESOLUTION_STATS[level]["cls_m"])
    cell_ids = np.ascontiguousarray(cell_ids, dtype=np.uint64)
    parents = (
        _parent_at_batch(cell_ids, _PARENT_LEVEL)
        if cell_ids.size
        else np.empty(0, dtype=np.uint64)
    )
    uniq, inv = np.unique(parents, return_inverse=True)
    d_k_u = np.empty(uniq.size)
    d_j_u = np.empty(uniq.size)
    d_i_u = np.empty(uniq.size)
    for k, p in enumerate(uniq):
        d = model.get_neighbor_distances(int(p), cls_m)
        d_k_u[k], d_j_u[k], d_i_u[k] = d["d_k"], d["d_j"], d["d_i"]
    return d_k_u[inv], d_j_u[inv], d_i_u[inv]


def _distances_geodesic(
    cell_ids: np.ndarray,
    neighbours: np.ndarray,
    grid_info,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell axis distances computed via geodesic (pyproj Geod.inv).

    O(N×6) geodesic calls + one DGGRID centroid lookup for all unique
    neighbour IDs. Use for validation and high-precision runs.

    Boundary / pentagon slots (INVALID sentinel) yield NaN for that
    direction; slope_fda will already NaN the affected cells via the
    boundary mask.
    """
    # centroids of cells in AOI
    c_lon, c_lat = grid_info.cell_ids2geographic(cell_ids)
    # centroids of all unique neighbour cells (may be outside AOI)
    flat = neighbours.ravel()
    valid_mask = flat != INVALID
    unique_nbrs = np.unique(flat[valid_mask]).astype(np.uint64)
    nb_lon_arr, nb_lat_arr = grid_info.cell_ids2geographic(unique_nbrs)
    nb_lookup: dict[int, tuple[float, float]] = {
        int(z): (float(lo), float(la))
        for z, lo, la in zip(unique_nbrs, nb_lon_arr, nb_lat_arr)
    }

    n = cell_ids.size
    d_per_dir = np.full((n, 6), np.nan)
    for i in range(n):
        for k in range(6):
            nz = int(neighbours[i, k])
            if nz == int(INVALID) or nz not in nb_lookup:
                continue
            nlo, nla = nb_lookup[nz]
            _, _, dist = _GEOD.inv(c_lon[i], c_lat[i], nlo, nla)
            d_per_dir[i, k] = dist

    # collapse to per-axis (mean of the two antipodal ends)
    d_k = 0.5 * (d_per_dir[:, 0] + d_per_dir[:, 5])
    d_j = 0.5 * (d_per_dir[:, 1] + d_per_dir[:, 4])
    d_i = 0.5 * (d_per_dir[:, 2] + d_per_dir[:, 3])
    return d_k, d_j, d_i


# ---------------------------------------------------------------------------
# core FDA — pure numpy, index-agnostic
# ---------------------------------------------------------------------------

def _slope_fda(
    h: np.ndarray,
    pos: np.ndarray,
    boundary: np.ndarray,
    d_k: np.ndarray,
    d_j: np.ndarray,
    d_i: np.ndarray,
    h_center: np.ndarray | None = None,
) -> np.ndarray:
    """Return slope magnitude (m/m) for each cell.

    h        : (M,) float64, elevation values addressed by `pos`.
    pos      : (N, 6) int, position of each neighbour in h; -1 = outside AOI.
    boundary : (N,) bool, True if any neighbour is outside AOI.
    d_k/j/i  : (N,) float, per-cell axis distances in metres.
    h_center : (N,) float64 or None. The centre-cell elevations. Defaults to
               `h`, i.e. the whole-AOI case where every value array row is
               also an evaluated cell (M == N).

               The blocked path passes them separately: there `h` is the
               chunk's core+halo values (M = owned + halo) while only the
               owned cells are evaluated (N = owned), so halo cells
               contribute values without costing a neighbour table, a
               distance lookup, or an output slot.

    Boundary and pentagon cells are set to NaN.
    """
    if h_center is None:
        h_center = h
    pos_safe = np.where(pos < 0, 0, pos)
    dh = h[pos_safe] - h_center[:, None]   # (N, 6)

    d_y = 0.5 * (d_j + d_i)
    dh_dx = (dh[:, 0] - dh[:, 5]) / (2.0 * d_k)
    dh_dy = (dh[:, 3] + dh[:, 4] - dh[:, 1] - dh[:, 2]) / (2.0 * np.sqrt(3.0) * d_y)
    result = np.hypot(dh_dx, dh_dy)
    result[boundary] = np.nan
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def slope(
    elevation_da: xr.DataArray,
    model: HexGridDistortionModel | None = None,
    *,
    distance_mode: str = "lookup",
) -> xr.DataArray:
    """Compute slope magnitude (m/m) on a Z7 DGGS elevation DataArray.

    Parameters
    ----------
    elevation_da:
        DataArray with a `cell_ids` dimension backed by an IGEO7Index.
        May be dask-backed; values are computed eagerly for this in-memory
        implementation.
    model:
        A `HexGridDistortionModel` instance, required when
        ``distance_mode="lookup"``.  Typically built once from
        ``data/working/dist_lookup_level4.parquet`` and reused.
    distance_mode:
        ``"lookup"``   — per-axis distances from the level-4 parent distortion
                         table.  Fast; ~1.86 % systematic under-estimate due to
                         ISEA shape distortion (see AGENT.md §Science backlog B1).
        ``"geodesic"`` — per-cell geodesic distances via pyproj + DGGRID centroid
                         lookup.  Sub-percent accuracy; O(N×6) pyproj calls.

    Returns
    -------
    xr.DataArray
        Slope magnitude in m/m, same ``cell_ids`` coordinates and IGEO7Index
        as the input.  Boundary cells (any neighbour outside the AOI) and
        pentagon cells are NaN.  attrs: ``units="m/m"``,
        ``long_name="slope magnitude (FDA)"``.
    """
    if distance_mode == "lookup" and model is None:
        raise ValueError("model must be provided when distance_mode='lookup'")

    idx = elevation_da.xindexes["cell_ids"]
    grid_info = idx.grid_info
    level = grid_info.level

    cell_ids = elevation_da.coords["cell_ids"].values.astype(np.uint64)
    h = np.asarray(elevation_da.values, dtype=np.float64)

    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)
    boundary = (pos < 0).any(axis=1)

    if distance_mode == "lookup":
        d_k, d_j, d_i = _distances_lookup(cell_ids, level, model)
    elif distance_mode == "geodesic":
        d_k, d_j, d_i = _distances_geodesic(cell_ids, nbrs, grid_info)
    else:
        raise ValueError(f"unknown distance_mode {distance_mode!r}; use 'lookup' or 'geodesic'")

    s = _slope_fda(h, pos, boundary, d_k, d_j, d_i)

    return xr.DataArray(
        s,
        coords=elevation_da.coords,
        dims=elevation_da.dims,
        attrs={"units": "m/m", "long_name": "slope magnitude (FDA)"},
    )


def _slope_on_halo_chunk(
    hc: HaloChunk,
    level: int,
    model: HexGridDistortionModel | None,
    distance_mode: str,
    grid_info,
) -> np.ndarray:
    """Run the FDA kernel on a HaloChunk; return slope only for owned cells."""
    cell_ids = hc.cell_ids
    h = hc.values
    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)
    boundary = (pos < 0).any(axis=1)

    if distance_mode == "lookup":
        d_k, d_j, d_i = _distances_lookup(cell_ids, level, model)
    else:
        d_k, d_j, d_i = _distances_geodesic(cell_ids, nbrs, grid_info)

    s_full = _slope_fda(h, pos, boundary, d_k, d_j, d_i)
    return s_full[hc.owned_mask]


def _slope_chunk_from_store(
    shared: dict,
    start: int,
    stop: int,
) -> np.ndarray:
    """Compute slope for the owned cells of one chunk, resolving its own halo.

    Runs entirely inside a dask task. The only global state it receives is
    ``shared`` — the O(R) range-table lookups plus small scalars — which dask
    stores once in the graph and every task references, so per-task memory is
    O(chunk), independent of both N and the chunk count.

    Sequence (one GBT pass, one core read, one run-coalesced halo read):

    1. derive this chunk's cell ids from the range table  (no dense array)
    2. read the chunk's own values as a single contiguous Zarr slice
    3. GBT the 1-ring neighbours of the owned cells
    4. resolve every neighbour to a *global* position via the range table;
       ``-1`` means absent from the archive → true AOI boundary → NaN
    5. anything resolving outside ``[start, stop)`` is halo: read those values
       run-coalesced
    6. FDA over the combined core+halo values, emitting owned cells only
    """
    import zarr

    from z7_xarray_paper.kernels.halo import (
        cell_ids_for_slice,
        positions_from_cell_ids,
        read_positions,
    )

    start, stop = int(start), int(stop)
    level = shared["level"]
    start_mono, end_mono, offsets = shared["start_mono"], shared["end_mono"], shared["offsets"]

    z = zarr.open_array(shared["array_path"], mode="r")

    chunk_ids = cell_ids_for_slice(start, stop, start_mono, offsets, level)
    core_vals = np.asarray(z[start:stop], dtype=np.float64)

    nbrs = get_neighbours_batch(chunk_ids)
    gpos = positions_from_cell_ids(
        nbrs.ravel(), start_mono, end_mono, offsets, level
    ).reshape(nbrs.shape)

    present = gpos >= 0
    boundary = ~present.all(axis=1)
    is_halo = present & ((gpos < start) | (gpos >= stop))

    halo_pos = np.unique(gpos[is_halo])
    halo_vals = (
        np.asarray(read_positions(z, halo_pos), dtype=np.float64)
        if halo_pos.size
        else np.empty(0, dtype=np.float64)
    )

    # Combined value array, ordered by global position. Owned cells occupy the
    # contiguous block [start, stop), so their global positions are already
    # sorted; merging the (sorted) halo positions keeps the whole thing sorted,
    # which is what lets the neighbour lookup be a searchsorted.
    combined_gpos = np.concatenate([np.arange(start, stop, dtype=np.int64), halo_pos])
    combined_vals = np.concatenate([core_vals, halo_vals])
    order = np.argsort(combined_gpos, kind="stable")
    combined_gpos = combined_gpos[order]
    combined_vals = combined_vals[order]

    local = np.searchsorted(combined_gpos, gpos)
    local = np.where(present, local, -1)

    if shared["distance_mode"] == "lookup":
        d_k, d_j, d_i = _distances_lookup(chunk_ids, level, shared["model"])
    else:
        d_k, d_j, d_i = _distances_geodesic(chunk_ids, nbrs, shared["grid_info"])

    return _slope_fda(
        combined_vals, local, boundary, d_k, d_j, d_i, h_center=core_vals
    )


def _assemble_and_slope(
    core_vals: np.ndarray,
    halo_vals: np.ndarray,
    chunk_ids: np.ndarray,
    halo_ids: np.ndarray,
    level: int,
    model,
    distance_mode: str,
    grid_info,
) -> np.ndarray:
    """Build HaloChunk from pre-fetched arrays and return owned-cell slopes.

    Module-level so dask.delayed can serialize it cleanly.
    """
    n_owned = chunk_ids.size
    combined_ids = np.concatenate([chunk_ids, halo_ids])
    combined_vals = np.concatenate([
        np.asarray(core_vals, dtype=np.float64),
        np.asarray(halo_vals, dtype=np.float64),
    ])
    owned_flag = np.zeros(combined_ids.size, dtype=bool)
    owned_flag[:n_owned] = True
    sort_idx = np.argsort(combined_ids, kind="stable")
    hc = HaloChunk(
        cell_ids=combined_ids[sort_idx],
        values=combined_vals[sort_idx],
        owned_mask=owned_flag[sort_idx],
        n_owned=n_owned,
    )
    return _slope_on_halo_chunk(hc, level, model, distance_mode, grid_info)


def slope_blocked(
    elevation_da: xr.DataArray,
    model: HexGridDistortionModel | None = None,
    *,
    distance_mode: str = "lookup",
    source: tuple[str, str] | None = None,
    chunk_cells: int | None = None,
) -> xr.DataArray:
    """Lazy, chunk-parallel slope with task-local halo resolution.

    Avoids xr.map_blocks, which raises "inconsistent chunks" when the
    cell_ids coordinate (Z7MonotonicIndex range-backed) has different dask
    chunk sizes than the data variable.

    Strategy (the scalable path, taken when ``source`` is given)
    -----------------------------------------------------------
    Graph construction is O(n_chunks) of trivial work: derive the range-table
    lookups once (O(R)), then emit one ``dask.delayed`` per chunk carrying
    nothing but two integers. No GBT calls, no dense ``cell_ids`` array.

    Every task then resolves its own halo (PHASE3.md §3.2's committed
    "read-time halo via GBT"): it reconstructs its cell ids from the range
    table, GBTs its 1-ring, binary-searches each neighbour to a global
    position (O(log R)), and run-coalesces the out-of-chunk positions into a
    few contiguous Zarr slice reads.

    This is what makes the kernel scale. Peak memory is O(chunk) + O(R) per
    worker and graph-build cost is independent of N, so the same code runs on
    an archive far larger than memory. The earlier implementation instead
    materialised the whole dense ``cell_ids`` array and ran a *serial* Python
    loop of per-chunk GBT calls before any parallelism started — which made
    graph construction O(N) in memory and, at 4,938 chunks, cost 26 s of the
    65 s total (see RESULTS_MEMO.md Task D, PHASE3.md §3.2 as-built).

    Parameters
    ----------
    elevation_da:
        DataArray with a ``cell_ids`` dim backed by a ``Z7MonotonicIndex``.
        Supplies the index, dtype and coords; its values are read from
        ``source`` rather than from this object when ``source`` is given.
    model:
        Required when ``distance_mode="lookup"``.
    distance_mode:
        ``"lookup"`` or ``"geodesic"`` — same semantics as ``slope()``.
    source:
        ``(archive_path, var_name)`` of the ``compression="ranges"`` Zarr
        archive backing ``elevation_da``. Enables the scalable path above.
        Recover it with ``ds.encoding["source"]`` after
        ``z7_zarr.open_dataset``. When omitted, falls back to the legacy
        eager-manifest path, which is correct but materialises the dense
        cell-ids array and does not scale past a few million cells.
    chunk_cells:
        Override the chunking used for parallelism. Defaults to the dask
        chunking of ``elevation_da`` (i.e. the on-disk layout). Only
        meaningful together with ``source``, since the scalable path reads
        by position slice and is free to choose its own chunk boundaries.

    Returns
    -------
    xr.DataArray
        Dask-backed slope DataArray, same shape and index as the input.
        AOI-boundary and pentagon cells are NaN; chunk-interior boundary
        cells are not — halo resolution makes chunking invisible to results.
    """
    if distance_mode == "lookup" and model is None:
        raise ValueError("model must be provided when distance_mode='lookup'")
    if distance_mode not in ("lookup", "geodesic"):
        raise ValueError(f"unknown distance_mode {distance_mode!r}")

    if source is None:
        return _slope_blocked_legacy(
            elevation_da, model, distance_mode=distance_mode
        )

    import dask
    import dask.array as da

    from z7_xarray_paper.kernels.halo import range_table_lookups

    idx = elevation_da.xindexes["cell_ids"]
    grid_info = idx.grid_info
    level = grid_info.level

    archive_path, var_name = source
    n_total = int(elevation_da.sizes["cell_ids"])

    start_mono, end_mono, offsets = range_table_lookups(idx.range_table, level)
    if int(offsets[-1]) != n_total:
        raise ValueError(
            f"range table covers {int(offsets[-1])} cells but the array has "
            f"{n_total} — source and index disagree"
        )

    # One graph entry shared by every task: dask stores it once and each task
    # references it, so the O(R) tables are not re-serialised per chunk.
    shared = dask.delayed(
        {
            "array_path": str(Path(archive_path) / var_name),
            "level": level,
            "start_mono": start_mono,
            "end_mono": end_mono,
            "offsets": offsets,
            "model": model,
            "distance_mode": distance_mode,
            "grid_info": grid_info,
        },
        pure=True,
    )

    if chunk_cells is None:
        bounds = np.concatenate([[0], np.cumsum(elevation_da.data.chunks[0])]).astype(int)
    else:
        edges = list(range(0, n_total, int(chunk_cells))) + [n_total]
        bounds = np.asarray(edges, dtype=int)

    blocks = []
    for i in range(bounds.size - 1):
        start, stop = int(bounds[i]), int(bounds[i + 1])
        blocks.append(
            da.from_delayed(
                dask.delayed(_slope_chunk_from_store)(shared, start, stop),
                shape=(stop - start,),
                dtype=np.float64,
            )
        )

    result = elevation_da.copy(data=da.concatenate(blocks))
    result.attrs = {"units": "m/m", "long_name": "slope magnitude (FDA)"}
    return result


def _slope_blocked_legacy(
    elevation_da: xr.DataArray,
    model: HexGridDistortionModel | None,
    *,
    distance_mode: str,
) -> xr.DataArray:
    """Pre-2026-08 blocked path: eager per-chunk halo manifest.

    Retained so callers that only have a DataArray (no archive path) keep
    working. Materialises the dense cell-ids array and builds every chunk's
    halo manifest serially at graph-construction time, so it does not scale —
    prefer passing ``source=`` to ``slope_blocked``. The one improvement kept
    from the rewrite is that the neighbour table is now computed in a single
    batched call rather than one small call per chunk.
    """
    import dask
    import dask.array as da

    h_dask = elevation_da.data
    if not hasattr(h_dask, "chunks"):
        raise ValueError(
            "slope_blocked requires a dask-backed DataArray; "
            "use slope() for eager computation"
        )

    idx = elevation_da.xindexes["cell_ids"]
    grid_info = idx.grid_info
    level = grid_info.level

    cell_ids_all = elevation_da.coords["cell_ids"].values.astype(np.uint64)

    # Single batched GBT over all cells, then slice per chunk — one parallel
    # njit call instead of n_chunks small serial ones.
    nbrs_all = get_neighbours_batch(cell_ids_all)
    pos_all = neighbour_positions(cell_ids_all, nbrs_all)

    chunks_tuple = h_dask.chunks[0]
    chunk_starts = np.concatenate([[0], np.cumsum(chunks_tuple)]).astype(int)

    slope_blocks = []
    for i, chunk_size in enumerate(chunks_tuple):
        start = int(chunk_starts[i])
        end = int(chunk_starts[i + 1])
        chunk_ids = cell_ids_all[start:end]

        pos_slice = pos_all[start:end]
        outside = (pos_slice >= 0) & ((pos_slice < start) | (pos_slice >= end))
        halo_indices = np.unique(pos_slice[outside]).astype(np.intp)
        halo_ids = (
            cell_ids_all[halo_indices] if halo_indices.size > 0
            else np.array([], dtype=np.uint64)
        )

        core_dask = h_dask[start:end]
        halo_dask = (
            h_dask[halo_indices] if halo_indices.size > 0
            else da.from_array(np.array([], dtype=np.float64))
        )

        delayed_slope = dask.delayed(_assemble_and_slope)(
            core_dask, halo_dask,
            chunk_ids, halo_ids,
            level, model, distance_mode, grid_info,
        )
        slope_blocks.append(
            da.from_delayed(delayed_slope, shape=(int(chunk_size),), dtype=np.float64)
        )

    result = elevation_da.copy(data=da.concatenate(slope_blocks))
    result.attrs = {"units": "m/m", "long_name": "slope magnitude (FDA)"}
    return result
