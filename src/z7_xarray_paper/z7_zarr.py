"""Z7 / IGEO7 Zarr archive writer & reader — dggs convention v1.

Phase 1 emits archives following the published DGGS Zarr convention with
``compression: "none"`` — i.e. a dense 1-D ``cell_ids`` coordinate of length N.
The archive is a plain xarray Dataset on disk; readers can open it with
``xr.open_zarr(...)`` and attach the IGEO7Index via ``xdggs.decode``.

Phase 2 will introduce ``compression: "ranges"`` archives (with a ``(R, 2)``
coordinate) and a lazy ``Z7MonotonicIndex`` that consumes them; the same
``write()`` / ``open_dataset()`` API will gain a ``compression`` kwarg at that
point. We keep range computation utilities here now because Phase 2 reuses them
as-is.

Convention reference:
    https://github.com/zarr-conventions/dggs/blob/v1/README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import numba as nb
import xarray as xr

from z7py.z7 import z7_to_monotonic_int, monotonic_int_to_z7

from z7_xarray_paper.config import (
    DEFAULT_CHUNK_CELLS,
    DGGS_CONVENTION_REGISTRATION,
    WGS84_ELLIPSOID,
    default_compressor,
)

_IGEO7_DGGS_NAME = "igeo7"
# Dim name == coord name, matching the xdggs ecosystem convention so
# `xdggs.decode` produces a single indexed dim coord rather than a separate
# eager dim coord plus a redundant lazy `cell_ids` coord.
_DEFAULT_SPATIAL_DIMENSION = "cell_ids"
_DEFAULT_COORDINATE_NAME = "cell_ids"

# For compression="ranges" we use a separate variable name and a separate
# (R-sized) dim because xarray cannot have one dim with two different sizes
# (the data dim is N, but the (R, 2) coord's first dim is R).
_RANGES_COORD_NAME = "cell_id_ranges"
_RANGES_DIM_NAME = "ranges"
_RANGES_BOUNDS_DIM = "bounds"

Compression = Literal["none", "ranges"]


# ---------------------------------------------------------------------------
# Numba batch helpers (z7py only ships scalar versions)
# ---------------------------------------------------------------------------


@nb.njit(cache=True, parallel=True)
def z7_to_monotonic_int_batch(raw: np.ndarray, resolution: int) -> np.ndarray:
    out = np.empty(raw.shape[0], dtype=np.uint64)
    for i in nb.prange(raw.shape[0]):
        out[i] = z7_to_monotonic_int(raw[i], resolution)
    return out


@nb.njit(cache=True, parallel=True)
def monotonic_int_to_z7_batch(values: np.ndarray, resolution: int) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.uint64)
    for i in nb.prange(values.shape[0]):
        out[i] = monotonic_int_to_z7(values[i], resolution)
    return out


def find_monotonic_ranges(
    cell_ids_uint64: np.ndarray,
    level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contiguous monotonic-int ranges for a sorted Z7 array.

    Returns
    -------
    range_start_z7  : uint64[R]   packed Z7 ID at the first cell of each range
    range_end_z7    : uint64[R]   packed Z7 ID at the last cell of each range (inclusive)

    Phase 1 doesn't write these to disk; Phase 2 will use them as the
    ``(R, 2)`` coordinate under ``compression: "ranges"``.
    """
    if cell_ids_uint64.ndim != 1:
        raise ValueError("cell_ids must be 1-D")
    cell_ids_uint64 = np.ascontiguousarray(cell_ids_uint64, dtype=np.uint64)
    n = cell_ids_uint64.size
    if n == 0:
        empty = np.empty(0, dtype=np.uint64)
        return empty, empty

    monotonic = z7_to_monotonic_int_batch(cell_ids_uint64, level)
    diffs = np.diff(monotonic.astype(np.int64))
    if (diffs < 0).any():
        raise ValueError("cell_ids are not sorted ascending by Z7 monotonic int")
    gap_idx = np.flatnonzero(diffs != 1)
    starts_idx = np.concatenate(([0], gap_idx + 1)).astype(np.int64)
    ends_idx   = np.concatenate((gap_idx, [n - 1])).astype(np.int64)
    return cell_ids_uint64[starts_idx], cell_ids_uint64[ends_idx]


# ---------------------------------------------------------------------------
# Convention metadata builders
# ---------------------------------------------------------------------------


def _dggs_block(
    *,
    level: int,
    spatial_dimension: str,
    coordinate: str,
    compression: str,
    dggs_vert0_lon: float,
    dggs_vert0_lat: float,
    dggs_vert0_azimuth: float,
    rotation_pattern: str = "alternating_cw_odd_ccw_even",
) -> dict[str, Any]:
    """Build the ``dggs`` attribute object for an IGEO7 archive."""
    return {
        "name":              _IGEO7_DGGS_NAME,
        "refinement_level":  int(level),
        "spatial_dimension": spatial_dimension,
        "coordinate":        coordinate,
        "compression":       compression,
        "ellipsoid":         dict(WGS84_ELLIPSOID),
        # IGEO7-specific extras (the convention permits additional fields).
        "dggs_vert0_lon":    float(dggs_vert0_lon),
        "dggs_vert0_lat":    float(dggs_vert0_lat),
        "dggs_vert0_azimuth": float(dggs_vert0_azimuth),
        "rotation_pattern":  rotation_pattern,
    }


def _coord_attrs_for_xdggs(
    *,
    level: int,
    dggs_vert0_lon: float,
) -> dict[str, Any]:
    """Per-coord attrs that the xdggs/IGEO7Index decode path expects.

    Distinct from the group-level dggs convention block — these are what
    ``xdggs.decode`` reads off the cell_ids coord variable to instantiate the
    index. Both must be present until xdggs natively understands the dggs
    convention metadata.
    """
    return {
        "grid_name":                       _IGEO7_DGGS_NAME,
        "level":                           int(level),
        "igeo7_dggs_vert0_lon":            float(dggs_vert0_lon),
        "igeo7_wgs84_geodetic_conversion": True,
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write(
    path: str | Path,
    *,
    cell_ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    level: int,
    compression: Compression = "none",
    dggs_vert0_lon: float = 11.20,
    dggs_vert0_lat: float = 58.28252559,
    dggs_vert0_azimuth: float = 0.0,
    chunk_cells: int = DEFAULT_CHUNK_CELLS,
    extra_attrs: Mapping | None = None,
) -> Path:
    """Write a dggs-convention IGEO7 archive.

    Parameters
    ----------
    cell_ids : uint64[N]
        Sorted ascending by Z7 monotonic int.
    data : mapping name -> array[N]
        Data variables aligned to cell_ids.
    level : int
        Z7/IGEO7 refinement level.
    compression : {"none", "ranges"}
        ``"none"`` writes a dense 1-D ``cell_ids`` coord (length N).
        ``"ranges"`` writes a ``(R, 2)`` ``cell_id_ranges`` coord (packed Z7
        start/end pairs) along ``(ranges, bounds)`` dims; the data dim
        ``cell_ids`` (length N) carries the data variables only.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")

    cell_ids = np.ascontiguousarray(cell_ids, dtype=np.uint64)
    if cell_ids.ndim != 1:
        raise ValueError("cell_ids must be 1-D")
    n = cell_ids.size
    for name, arr in data.items():
        if arr.shape != (n,):
            raise ValueError(
                f"data[{name!r}].shape={arr.shape} does not match cell_ids.shape=({n},)"
            )

    if compression == "none":
        ds = _build_compression_none(
            cell_ids=cell_ids, data=data, level=level,
            dggs_vert0_lon=dggs_vert0_lon,
            chunk_cells=chunk_cells,
        )
        ds.attrs["dggs"] = _dggs_block(
            level=level,
            spatial_dimension=_DEFAULT_SPATIAL_DIMENSION,
            coordinate=_DEFAULT_COORDINATE_NAME,
            compression="none",
            dggs_vert0_lon=dggs_vert0_lon,
            dggs_vert0_lat=dggs_vert0_lat,
            dggs_vert0_azimuth=dggs_vert0_azimuth,
        )
    elif compression == "ranges":
        ds = _build_compression_ranges(
            cell_ids=cell_ids, data=data, level=level,
            dggs_vert0_lon=dggs_vert0_lon,
            chunk_cells=chunk_cells,
        )
        ds.attrs["dggs"] = _dggs_block(
            level=level,
            spatial_dimension=_DEFAULT_SPATIAL_DIMENSION,
            coordinate=_RANGES_COORD_NAME,
            compression="ranges",
            dggs_vert0_lon=dggs_vert0_lon,
            dggs_vert0_lat=dggs_vert0_lat,
            dggs_vert0_azimuth=dggs_vert0_azimuth,
        )
    else:
        raise ValueError(f"unknown compression={compression!r}; expected 'none' or 'ranges'")

    ds.attrs["zarr_conventions"] = [DGGS_CONVENTION_REGISTRATION]
    if extra_attrs:
        for k, v in dict(extra_attrs).items():
            ds.attrs[k] = v

    # Apply project-default Blosc/zstd to every array. Beats xarray's default
    # Blosc/LZ4 by ~2× on DEM-style float32 and dramatically more on the
    # uint64 cell_ids / cell_id_ranges (sorted, mostly-shared high bits).
    compressor = default_compressor()
    encoding = {name: {"compressor": compressor} for name in ds.variables}

    ds.to_zarr(str(path), mode="w-", encoding=encoding, consolidated=False)
    return path


def _build_compression_none(*, cell_ids, data, level, dggs_vert0_lon, chunk_cells) -> xr.Dataset:
    spatial_dim = _DEFAULT_SPATIAL_DIMENSION
    coord_name  = _DEFAULT_COORDINATE_NAME
    coord_attrs = _coord_attrs_for_xdggs(level=level, dggs_vert0_lon=dggs_vert0_lon)
    ds = xr.Dataset(
        data_vars={name: ((spatial_dim,), arr) for name, arr in data.items()},
        coords={coord_name: ((spatial_dim,), cell_ids, coord_attrs)},
    )
    return ds.chunk({spatial_dim: chunk_cells})


def _build_compression_ranges(*, cell_ids, data, level, dggs_vert0_lon, chunk_cells) -> xr.Dataset:
    range_start_z7, range_end_z7 = find_monotonic_ranges(cell_ids, level)
    range_table = np.stack([range_start_z7, range_end_z7], axis=1).astype(np.uint64)

    spatial_dim = _DEFAULT_SPATIAL_DIMENSION
    coord_attrs = _coord_attrs_for_xdggs(level=level, dggs_vert0_lon=dggs_vert0_lon)

    # Data variables on the data dim (length N); no 1-D cell_ids coord.
    # The (R, 2) cell_id_ranges variable sits on (ranges, bounds), distinct from
    # the data dim; it's the source of truth for the index at read time.
    ds = xr.Dataset(
        data_vars={name: ((spatial_dim,), arr) for name, arr in data.items()},
        coords={
            _RANGES_COORD_NAME: (
                (_RANGES_DIM_NAME, _RANGES_BOUNDS_DIM),
                range_table,
                coord_attrs,
            ),
        },
    )
    return ds.chunk({spatial_dim: chunk_cells})


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def open_dataset(path: str | Path, *, decode: bool = True) -> xr.Dataset:
    """Open a dggs-convention IGEO7 archive as an xarray Dataset.

    Dispatches on ``.zattrs.dggs.compression``:

    - ``"none"``: the dense 1-D ``cell_ids`` coord is decoded via stock
      ``xdggs.decode`` → ``IGEO7Index`` (eager ``PandasIndex``).
    - ``"ranges"``: the ``(R, 2)`` ``cell_id_ranges`` coord is decoded by
      our project-local helper → ``Z7MonotonicIndex`` (lazy, range-table
      backed). Stock ``xdggs.decode`` rejects 2-D coordinate variables, so
      we cannot route through it for this case.

    Parameters
    ----------
    path : Path-like
    decode : bool, default True
        If False, returns the raw ``xr.open_zarr`` Dataset with no DGGS
        index attached.
    """
    ds = xr.open_zarr(str(path), consolidated=False)
    if not decode:
        return ds

    compression = ds.attrs.get("dggs", {}).get("compression", "none")
    if compression == "none":
        import xdggs                                            # noqa: F401
        from xdggs_dggrid4py.index import IGEO7Index            # noqa: F401  - registers grid
        return ds.pipe(xdggs.decode)
    elif compression == "ranges":
        return _decode_ranges(ds)
    else:
        raise ValueError(f"unsupported dggs.compression={compression!r}")


def _decode_ranges(ds: xr.Dataset) -> xr.Dataset:
    """Attach a Z7MonotonicIndex on the data dim of a compression='ranges' archive."""
    from xdggs_dggrid4py.monotonic_index import Z7MonotonicIndex

    if _RANGES_COORD_NAME not in ds.variables:
        raise ValueError(
            f"compression='ranges' archive missing {_RANGES_COORD_NAME!r} variable"
        )
    var = ds[_RANGES_COORD_NAME]

    # The dggs convention block at root carries `spatial_dimension` (the data dim).
    spatial_dim = ds.attrs.get("dggs", {}).get("spatial_dimension", _DEFAULT_SPATIAL_DIMENSION)

    # Make sure the variable's attrs carry the canonical IGEO7 fields the
    # index expects. If the writer set them on the coord, they're already
    # there; otherwise carry over from the group-level dggs block.
    grid_attrs = ds.attrs.get("dggs", {})
    coord_attrs = dict(var.attrs)
    coord_attrs.setdefault("grid_name", grid_attrs.get("name", "igeo7"))
    coord_attrs.setdefault("level", grid_attrs.get("refinement_level"))
    coord_attrs.setdefault("igeo7_dggs_vert0_lon", grid_attrs.get("dggs_vert0_lon", 11.20))
    coord_attrs.setdefault("igeo7_wgs84_geodetic_conversion", True)

    # Re-wrap with the merged attrs so the index sees them.
    var_with_attrs = var.copy()
    var_with_attrs.attrs = coord_attrs

    index = Z7MonotonicIndex.from_variables(
        {_RANGES_COORD_NAME: var_with_attrs},
        options={"dim": spatial_dim},
    )

    # Drop the (R, 2) variable and the now-unused range/bounds dims;
    # the index supplies a lazy length-N `cell_ids` coord via
    # `Z7MonotonicIndex.create_variables`. `from_xindex` wires that up.
    ds = ds.drop_vars(_RANGES_COORD_NAME)
    return ds.assign_coords(xr.Coordinates.from_xindex(index))
