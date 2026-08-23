from z7_xarray_paper.kernels.slope import slope, slope_blocked
from z7_xarray_paper.kernels.halo import (
    HaloChunk,
    build_halo_chunk,
    cell_ids_for_slice,
    positions_from_cell_ids,
    range_table_lookups,
    read_positions,
)

__all__ = [
    "slope",
    "slope_blocked",
    "HaloChunk",
    "build_halo_chunk",
    # range-table primitives — the substrate for chunk-parallel kernels
    "range_table_lookups",
    "positions_from_cell_ids",
    "cell_ids_for_slice",
    "read_positions",
]
