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


# ---------------------------------------------------------------------------
# Range-table primitives — the scalable substrate for chunk-parallel kernels
# ---------------------------------------------------------------------------
#
# Everything below works off the (R, 2) packed-Z7 range table alone, never the
# dense length-N cell-ids array.  That is what makes a chunk-parallel kernel
# scale: the per-task state is O(R) (tens of KB) plus O(chunk), independent of
# N.  R is the number of contiguous monotonic-int runs in the archive — 39,697
# for Estonia r12, i.e. 0.33 % of N.


def range_table_lookups(range_z7: np.ndarray, level: int):
    """Derive the search structures for a (R, 2) packed-Z7 range table.

    Returns ``(range_start_mono, range_end_mono, data_offsets)`` — the same
    three arrays ``Z7MonotonicIndex`` builds internally, but as a plain tuple
    that is cheap to ship into a dask task (O(R), ~1 MB at Estonia r12).

    ``data_offsets[i]`` is the global position of the first cell of range i,
    so ``data_offsets[-1] == N``.
    """
    from z7_xarray_paper.z7_zarr import z7_to_monotonic_int_batch

    range_z7 = np.ascontiguousarray(range_z7, dtype=np.uint64)
    if range_z7.ndim != 2 or range_z7.shape[1] != 2:
        raise ValueError(f"range_z7 must be (R, 2) uint64, got {range_z7.shape}")
    if range_z7.size == 0:
        empty = np.empty(0, dtype=np.uint64)
        return empty, empty, np.array([0], dtype=np.int64)

    start_mono = z7_to_monotonic_int_batch(range_z7[:, 0], level)
    end_mono = z7_to_monotonic_int_batch(range_z7[:, 1], level)
    lengths = (end_mono - start_mono + np.uint64(1)).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return start_mono, end_mono, offsets


def positions_from_cell_ids(
    cell_ids: np.ndarray,
    start_mono: np.ndarray,
    end_mono: np.ndarray,
    offsets: np.ndarray,
    level: int,
) -> np.ndarray:
    """Map packed-Z7 cell ids to their global positions in the archive.

    O(len(cell_ids) · log R) via binary search on the range table — the same
    arithmetic as ``Z7MonotonicIndex.sel``, but returning ``-1`` for cells
    that are absent from the archive instead of raising.  Absent cells are
    the normal case here: AOI-boundary neighbours and the ``INVALID``
    pentagon sentinel both land outside every range.
    """
    from z7_xarray_paper.z7_zarr import z7_to_monotonic_int_batch

    cell_ids = np.ascontiguousarray(cell_ids, dtype=np.uint64)
    if cell_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    if start_mono.size == 0:
        return np.full(cell_ids.size, -1, dtype=np.int64)

    mono = z7_to_monotonic_int_batch(cell_ids, level)
    idx = np.searchsorted(start_mono, mono, side="right") - 1
    clipped = np.maximum(idx, 0)
    within = (idx >= 0) & (mono <= end_mono[clipped])
    # int64 before subtracting: a not-within query can sit below its clipped
    # range start, and uint64 would wrap to a huge value.
    delta = mono.astype(np.int64) - start_mono[clipped].astype(np.int64)
    return np.where(within, offsets[clipped] + delta, -1).astype(np.int64)


def cell_ids_for_slice(
    start: int,
    stop: int,
    start_mono: np.ndarray,
    offsets: np.ndarray,
    level: int,
) -> np.ndarray:
    """Reconstruct ``cell_ids[start:stop]`` from the range table alone.

    Touches only the ranges overlapping the requested slice, so the cost is
    O(chunk + ranges_in_chunk) rather than O(N).  This is what lets a dask
    task know which cells it owns without any global array.
    """
    from z7_xarray_paper.z7_zarr import monotonic_int_to_z7_batch

    start, stop = int(start), int(stop)
    if stop <= start:
        return np.empty(0, dtype=np.uint64)

    first = int(np.searchsorted(offsets, start, side="right") - 1)
    last = int(np.searchsorted(offsets, stop, side="left") - 1)
    out_mono = np.empty(stop - start, dtype=np.uint64)
    written = 0
    for r in range(max(first, 0), min(last + 1, offsets.size - 1)):
        r_lo, r_hi = int(offsets[r]), int(offsets[r + 1])
        lo, hi = max(r_lo, start), min(r_hi, stop)
        if hi <= lo:
            continue
        s = int(start_mono[r]) + (lo - r_lo)
        n = hi - lo
        out_mono[written:written + n] = np.arange(s, s + n, dtype=np.uint64)
        written += n
    return monotonic_int_to_z7_batch(out_mono[:written], level)


def read_positions(zarr_array, positions: np.ndarray) -> np.ndarray:
    """Read ``zarr_array`` at arbitrary global positions, storage-chunk-coalesced.

    Grouped by **Zarr storage chunk**, issuing exactly one read per distinct
    chunk touched. That granularity is the point: Zarr decompresses a whole
    chunk however few elements you ask for, so a read costs the same whether
    it returns 1 value or the entire chunk — but *repeating* reads into the
    same chunk repeats the decompression.

    An earlier version here coalesced by contiguous run instead. A 1-ring
    halo is contiguous only locally; the scattered tail (see RESULTS_MEMO.md
    Task C, where interior-cell neighbour spread runs to 10^6 positions)
    produced hundreds of runs inside a single 23 MB storage chunk, and so
    hundreds of redundant decompressions of it. That cost 7.57 s to fetch
    2,283 halo cells — 98 % of a task — versus milliseconds now.

    Within each chunk group only the ``[min, max]`` span actually needed is
    requested: the decompression is identical, but the intermediate copy is
    bounded by the span rather than the full chunk.
    """
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    if positions.size == 0:
        return np.empty(0, dtype=zarr_array.dtype)

    chunk_size = int(zarr_array.chunks[0])
    order = np.argsort(positions, kind="stable")
    ordered = positions[order]

    chunk_of = ordered // chunk_size
    breaks = np.flatnonzero(np.diff(chunk_of) != 0)
    grp_starts = np.concatenate(([0], breaks + 1))
    grp_ends = np.concatenate((breaks + 1, [ordered.size]))

    gathered = np.empty(ordered.size, dtype=zarr_array.dtype)
    for s, e in zip(grp_starts, grp_ends):
        lo = int(ordered[s])
        hi = int(ordered[e - 1]) + 1
        block = zarr_array[lo:hi]
        gathered[s:e] = block[ordered[s:e] - lo]

    out = np.empty_like(gathered)
    out[order] = gathered
    return out


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
