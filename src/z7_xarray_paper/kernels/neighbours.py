"""Z7 neighbour utilities for focal kernels.

Provides the (N, 6) uint64 neighbour table and its positional projection
into a sorted cell_ids array. These are the building blocks for any
map_blocks focal kernel; they do not touch elevation values.

z7py canonical direction labelling (settled in phase3a_slope_math.py):
    d=1 (col 0): k+    d=6 (col 5): k-
    d=2 (col 1): j+    d=5 (col 4): j-
    d=4 (col 3): i+    d=3 (col 2): i-

The batch neighbour table is filled by a Numba-compiled kernel that loops
over the cell axis with ``prange``. The per-cell GBT carry chain is
inherently serial over the digit axis, but the cell axis is independent,
so the batch fill parallelises across cores without any synchronisation.
"""

from __future__ import annotations

import os

import numpy as np
import numba as nb

from z7py import z7

INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------
# compiled batch kernels
# ---------------------------------------------------------------------------

@nb.njit(cache=True, nogil=True)
def _fill_neighbours_serial(cell_ids, out):
    """Serial njit fill of the (N, 6) neighbour table."""
    n = cell_ids.shape[0]
    for i in range(n):
        row = z7.get_neighbours(cell_ids[i])
        for d in range(6):
            out[i, d] = row[d]
    return out


@nb.njit(cache=True, parallel=True, nogil=True)
def _fill_neighbours_parallel(cell_ids, out):
    """Thread-parallel njit fill of the (N, 6) neighbour table."""
    n = cell_ids.shape[0]
    for i in nb.prange(n):
        row = z7.get_neighbours(cell_ids[i])
        for d in range(6):
            out[i, d] = row[d]
    return out


# Below this many cells the thread launch dominates the work.
_PARALLEL_THRESHOLD = int(os.environ.get("Z7_PARALLEL_THRESHOLD", 4096))


def get_neighbours_batch(
    cell_ids: np.ndarray, *, parallel: bool | None = None
) -> np.ndarray:
    """Return (N, 6) uint64 of canonical d=1..6 neighbours for each cell.

    Pentagon cells have INVALID (0xFFFF...) in the missing slot.

    Parameters
    ----------
    cell_ids:
        1-D array of Z7 indices; cast to ``uint64`` if needed.
    parallel:
        Force the threaded or the serial kernel. The default picks the
        threaded kernel above ``Z7_PARALLEL_THRESHOLD`` cells (4096), below
        which the thread launch costs more than the work itself.
    """
    cell_ids = np.ascontiguousarray(cell_ids, dtype=np.uint64)
    if cell_ids.ndim != 1:
        raise ValueError(f"cell_ids must be 1-D, got shape {cell_ids.shape}")
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    if cell_ids.size == 0:
        return out
    if parallel is None:
        parallel = cell_ids.size >= _PARALLEL_THRESHOLD
    if parallel:
        return _fill_neighbours_parallel(cell_ids, out)
    return _fill_neighbours_serial(cell_ids, out)


def get_neighbours_batch_pyloop(cell_ids: np.ndarray) -> np.ndarray:
    """Reference implementation: Python loop over the njit single-cell kernel.

    Retained as the correctness oracle for the compiled batch kernels and as
    a baseline in the throughput benchmark.
    """
    cell_ids = np.ascontiguousarray(cell_ids, dtype=np.uint64)
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i, cid in enumerate(cell_ids):
        out[i, :] = z7.get_neighbours(np.uint64(cid))
    return out


# ---------------------------------------------------------------------------
# positional projection
# ---------------------------------------------------------------------------

def neighbour_positions(
    cell_ids: np.ndarray, neighbours: np.ndarray
) -> np.ndarray:
    """Map an (N, 6) uint64 neighbour array to positions in sorted cell_ids.

    Out-of-AOI neighbours (including INVALID sentinel) are mapped to -1.
    The caller is responsible for masking cells where any position is -1
    (boundary cells).
    """
    flat = neighbours.ravel()
    pos = np.searchsorted(cell_ids, flat)
    safe = np.minimum(pos, cell_ids.size - 1)
    hit = (pos < cell_ids.size) & (cell_ids[safe] == flat)
    pos = np.where(hit, pos, -1)
    return pos.reshape(neighbours.shape)
