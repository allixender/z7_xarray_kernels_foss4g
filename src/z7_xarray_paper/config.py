"""Project-wide constants for IGEO7 / Z7 regridding & storage."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_INPUT   = REPO_ROOT / "data" / "input"
DATA_WORKING = REPO_ROOT / "data" / "working"
DATA_OUTPUT  = REPO_ROOT / "data" / "output"

# ---------------------------------------------------------------------------
# IGEO7 DGGRID metafile config (pydggsapi-canonical)
# ---------------------------------------------------------------------------
# Z7 hierarchical INT64 ndx form is the only one we use end-to-end —
# DGGRID returns uint64 directly, no string parsing in the hot path.
#
# dggs_vert0_lat / dggs_vert0_azimuth are DGGRID's internal IGEO7 defaults
# (declared `Final` in pydggsapi.dependencies.dggrs_providers.igeo7); we
# expose them here for explicitness even though the plugin does not pass
# them by default.
IGEO7_META: dict = {
    "dggs_vert0_lon":         11.20,
    "dggs_vert0_lat":         58.28252559,
    "dggs_vert0_azimuth":     0.0,
    "input_address_type":     "HIERNDX",
    "input_hier_ndx_system":  "Z7",
    "input_hier_ndx_form":    "INT64",
    "output_address_type":    "HIERNDX",
    "output_cell_label_type": "OUTPUT_ADDRESS_TYPE",
    "output_hier_ndx_system": "Z7",
    "output_hier_ndx_form":   "INT64",
}


def clipper_scale_factor_for(level: int) -> int:
    """Resolution-dependent DGGRID clipper scale factor.

    DGGRID's default is 1_000_000 — adequate for coarse levels, but the
    integer-arithmetic clipper truncates polygon intersections at fine
    levels and produces gappy / duplicated cells on the clip boundary.
    Bumping the factor expands the integer grid the clipper works on.
    """
    if level <= 8:
        return 1_000_000
    if level <= 11:
        return 10_000_000
    return 100_000_000


# ---------------------------------------------------------------------------
# Zarr archive — DGGS convention v1
# https://github.com/zarr-conventions/dggs/blob/v1/README.md
# ---------------------------------------------------------------------------

DGGS_CONVENTION_REGISTRATION = {
    "schema_url":  "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url":    "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "uuid":        "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name":        "dggs",
    "description": "Discrete Global Grid Systems convention for zarr",
}

# WGS84 ellipsoid block for the dggs convention metadata.
WGS84_ELLIPSOID = {
    "name":               "wgs84",
    "semimajor_axis":     6378137.0,
    "inverse_flattening": 298.257223563,
}

# Default chunk size for the cell_ids dim (multiple of 7 — see
# Z7_ZARR_ARCHIVE_INDEX_JULIA.md).
DEFAULT_CHUNK_CELLS = 16_807   # 7^5  — tune to ~10–100 MB target per dtype


def default_compressor():
    """Project-wide default Zarr compressor: Blosc/zstd, clevel=3, byte shuffle.

    Lazy import of numcodecs so the config module stays light. zstd at
    clevel=3 is the sweet spot for our DEM and cell-id arrays — better
    ratio than LZ4 (xarray's Zarr v2 default) at comparable read speed.
    Sorted uint64 cell ids in particular compress dramatically because
    most cells in a contiguous AOI share their upper bits.
    """
    import numcodecs
    return numcodecs.Blosc(
        cname="zstd",
        clevel=3,
        shuffle=numcodecs.Blosc.SHUFFLE,
    )


# ---------------------------------------------------------------------------
# Phase 1 AOI definitions
# ---------------------------------------------------------------------------

PORI_DEM_PATH = DATA_INPUT / "merit_dem_pori_cog.tif"
PORI_NODATA   = -9999.0
