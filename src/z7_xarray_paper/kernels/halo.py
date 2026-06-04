"""HaloChunk container and builder for Z7 focal kernels.

For each dask chunk, the 1-ring halo cells (neighbours that live outside the
chunk's cell_ids slice) are fetched from the full DataArray via
xr.DataArray.sel() before the focal kernel runs.  This is the "read-time halo
via GBT" strategy settled in PHASE3.md §3.2.

The combined (core + halo) array is kept sorted so that neighbour_positions()
(searchsorted-based) works without modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from z7_xarray_paper.kernels.neighbours import INVALID, get_neighbours_batch, neighbour_positions


@dataclass
class HaloChunk:
    """Core cells plus 1-ring halo cells merged into one sorted flat array.

    cell_ids[owned_mask]  — the N owned (core) cells.
    cell_ids[~owned_mask] — H halo cells fetched from outside the chunk.

    The whole array is sorted so neighbour_positions() (searchsorted) works
    on it directly.  The kernel returns results only for owned cells.
    """

    cell_ids: np.ndarray     # (N+H,) uint64, sorted
    values: np.ndarray       # (N+H,) float64
    owned_mask: np.ndarray   # (N+H,) bool; True for core cells
    n_owned: int             # == owned_mask.sum()


def build_halo_chunk(chunk_da: xr.DataArray, full_da: xr.DataArray) -> HaloChunk:
    """Build a HaloChunk by fetching 1-ring halo from full_da.

    Parameters
    ----------
    chunk_da:
        One chunk's DataArray (contiguous cell_ids slice of full_da).
    full_da:
        Full AOI DataArray.  Halo values are fetched via full_da.sel().
        For Z7MonotonicIndex-backed arrays this collapses to a small number
        of Zarr chunk reads (range-aware sel).

    Notes
    -----
    Out-of-AOI neighbours (cells whose IDs are not in full_da at all) are
    simply absent from the halo.  The slope kernel will mark those boundary
    cells NaN via the boundary mask.
    """
    chunk_ids = chunk_da.coords["cell_ids"].values.astype(np.uint64)
    chunk_vals = np.asarray(chunk_da.values, dtype=np.float64)

    nbrs = get_neighbours_batch(chunk_ids)
    pos = neighbour_positions(chunk_ids, nbrs)

    # Candidates: neighbours outside this chunk, not the INVALID pentagon sentinel
    outside = (pos < 0) & (nbrs != INVALID)
    halo_candidates = np.unique(nbrs[outside].astype(np.uint64))

    if halo_candidates.size > 0:
        full_ids = full_da.coords["cell_ids"].values.astype(np.uint64)
        hpos = np.searchsorted(full_ids, halo_candidates)
        hpos_safe = np.minimum(hpos, full_ids.size - 1)
        in_aoi = (hpos < full_ids.size) & (full_ids[hpos_safe] == halo_candidates)
        halo_ids = halo_candidates[in_aoi]
    else:
        halo_ids = halo_candidates  # empty uint64

    if halo_ids.size > 0:
        halo_da = full_da.sel(cell_ids=halo_ids)
        halo_vals = np.asarray(halo_da.values, dtype=np.float64)
    else:
        halo_vals = np.array([], dtype=np.float64)

    combined_ids = np.concatenate([chunk_ids, halo_ids])
    combined_vals = np.concatenate([chunk_vals, halo_vals])
    owned_flag = np.zeros(combined_ids.size, dtype=bool)
    owned_flag[: chunk_ids.size] = True

    sort_idx = np.argsort(combined_ids, kind="stable")
    return HaloChunk(
        cell_ids=combined_ids[sort_idx],
        values=combined_vals[sort_idx],
        owned_mask=owned_flag[sort_idx],
        n_owned=int(chunk_ids.size),
    )
