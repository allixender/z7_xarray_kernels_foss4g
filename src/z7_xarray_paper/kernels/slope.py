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
    parents = np.fromiter(
        (int(z7.get_parent_at(np.uint64(c), _PARENT_LEVEL)) for c in cell_ids),
        dtype=np.uint64,
        count=cell_ids.size,
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
) -> np.ndarray:
    """Return slope magnitude (m/m) for each cell.

    h        : (N,) float64, elevation.
    pos      : (N, 6) int, position of each neighbour in h; -1 = outside AOI.
    boundary : (N,) bool, True if any neighbour is outside AOI.
    d_k/j/i  : (N,) float, per-cell axis distances in metres.

    Boundary and pentagon cells are set to NaN.
    """
    pos_safe = np.where(pos < 0, 0, pos)
    dh = h[pos_safe] - h[:, None]          # (N, 6)

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
) -> xr.DataArray:
    """Lazy, chunk-parallel slope using pre-built halo manifest + dask.delayed.

    Avoids xr.map_blocks, which raises "inconsistent chunks" when the
    cell_ids coordinate (Z7MonotonicIndex range-backed) has different dask
    chunk sizes than the data variable.

    Strategy
    --------
    At construction time (eager, cheap):
      - Materialize cell_ids index (uint64, not the elevation data).
      - For each chunk, call get_neighbours on boundary cells to determine
        which halo cell indices are needed (GBT calls only).

    At compute time (lazy):
      - Each dask.delayed task receives core_vals and halo_vals as dask slices
        from h_dask, assembles a HaloChunk, and runs the FDA kernel.
      - Halo slices are plain integer-index fancy-selects from h_dask — no
        xr.DataArray.sel() inside tasks, no nested dask graphs.

    Parameters
    ----------
    elevation_da:
        Dask-backed DataArray with a ``cell_ids`` dimension.  Each chunk
        should be a contiguous monotonic-int slice for compact halos.
    model:
        Required when ``distance_mode="lookup"``.
    distance_mode:
        ``"lookup"`` or ``"geodesic"`` — same semantics as ``slope()``.

    Returns
    -------
    xr.DataArray
        Dask-backed slope DataArray, same shape and index as the input.
        AOI-boundary cells are NaN; chunk-interior boundary cells are not.
    """
    if distance_mode == "lookup" and model is None:
        raise ValueError("model must be provided when distance_mode='lookup'")
    if distance_mode not in ("lookup", "geodesic"):
        raise ValueError(f"unknown distance_mode {distance_mode!r}")

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

    # Materialize cell_ids index once — cheap, uint64 only, not the elevation data
    cell_ids_all = elevation_da.coords["cell_ids"].values.astype(np.uint64)

    chunks_tuple = h_dask.chunks[0]
    chunk_starts = np.concatenate([[0], np.cumsum(chunks_tuple)]).astype(int)

    slope_blocks = []
    for i, chunk_size in enumerate(chunks_tuple):
        start = int(chunk_starts[i])
        end = int(chunk_starts[i + 1])
        chunk_ids = cell_ids_all[start:end]

        # Pre-build halo manifest: which global indices are needed as halo?
        nbrs = get_neighbours_batch(chunk_ids)
        pos_local = neighbour_positions(chunk_ids, nbrs)
        outside = (pos_local < 0) & (nbrs != INVALID)
        halo_candidates = np.unique(nbrs[outside].astype(np.uint64))

        if halo_candidates.size > 0:
            hpos = np.searchsorted(cell_ids_all, halo_candidates)
            hpos_safe = np.minimum(hpos, cell_ids_all.size - 1)
            in_aoi = (hpos < cell_ids_all.size) & (cell_ids_all[hpos_safe] == halo_candidates)
            halo_indices = hpos[in_aoi]  # integer positions into h_dask
        else:
            halo_indices = np.array([], dtype=np.intp)

        halo_ids = (
            cell_ids_all[halo_indices] if halo_indices.size > 0
            else np.array([], dtype=np.uint64)
        )

        # Lazy slices — no xr.DataArray.sel() inside tasks
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

    result_dask = da.concatenate(slope_blocks)
    # copy() preserves coords, dims, and the Z7MonotonicIndex from elevation_da
    result = elevation_da.copy(data=result_dask)
    result.attrs = {"units": "m/m", "long_name": "slope magnitude (FDA)"}
    return result
