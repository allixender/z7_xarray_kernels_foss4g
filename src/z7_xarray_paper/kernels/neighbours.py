"""Z7 neighbour utilities for focal kernels.

Provides the (N, 6) uint64 neighbour table and its positional projection
into a sorted cell_ids array. These are the building blocks for any
map_blocks focal kernel; they do not touch elevation values.

z7py canonical direction labelling (settled in phase3a_slope_math.py):
    d=1 (col 0): k+    d=6 (col 5): k-
    d=2 (col 1): j+    d=5 (col 4): j-
    d=4 (col 3): i+    d=3 (col 2): i-
"""

from __future__ import annotations

import numpy as np
from z7py import z7

INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)


def get_neighbours_batch(cell_ids: np.ndarray) -> np.ndarray:
    """Return (N, 6) uint64 of canonical d=1..6 neighbours for each cell.

    Pentagon cells have INVALID (0xFFFF…) in the missing slot.
    """
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i, cid in enumerate(cell_ids):
        out[i, :] = z7.get_neighbours(np.uint64(cid))
    return out


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
