#!/usr/bin/env python
"""Phase 1 demo — regrid the Pori MERIT DEM to IGEO7/Z7, write Zarr, round-trip.

Run with:

    pixi run python scripts/demo_regrid_pori_to_z7.py [--level 12]

Outputs:
    data/working/pori_z7_r{level}.zarr   (z7_sparse_index_v1)

Stages timed and reported on stdout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  — registers the .rio accessor on xarray
import xarray as xr

from xdggs_dggrid4py.regridding import mapblocks_regridding

import z7_xarray_paper.z7_zarr as z7_zarr
from z7_xarray_paper.config import (
    DATA_WORKING,
    IGEO7_META,
    PORI_DEM_PATH,
    PORI_NODATA,
    clipper_scale_factor_for,
)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


class Timer:
    """Minimal context-manager timer that records stages on a parent dict."""

    def __init__(self, store: dict, label: str):
        self.store = store
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"\n[{self.label}] start ...")
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.store[self.label] = dt
        print(f"[{self.label}] done in {dt:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=12, help="IGEO7/Z7 refinement level (default 12)")
    parser.add_argument(
        "--chunks",
        type=int,
        default=512,
        help="Raster chunk size in pixels per axis (default 512)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output Zarr archive first",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "ranges"],
        default="none",
        help="dggs convention compression mode (default: none — dense 1-D cell_ids; "
             "ranges = (R, 2) cell_id_ranges + Z7MonotonicIndex on read)",
    )
    args = parser.parse_args()

    timings: dict[str, float] = {}
    DATA_WORKING.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.compression}" if args.compression != "none" else ""
    out_path = DATA_WORKING / f"pori_z7_r{args.level}{suffix}.zarr"

    if out_path.exists():
        if args.overwrite:
            print(f"removing existing {out_path}")
            shutil.rmtree(out_path)
        else:
            raise SystemExit(f"output exists: {out_path} (pass --overwrite to replace)")

    if "DGGRID_PATH" not in os.environ:
        # Fallback when the script is run outside `pixi run`.
        candidate = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "dggrid"
        if candidate.exists():
            os.environ["DGGRID_PATH"] = str(candidate)
        else:
            raise SystemExit("DGGRID_PATH not set and no candidate dggrid found")

    # ---------------------------------------------------------------- stage 1
    with Timer(timings, "open_dem"):
        ds = xr.open_dataset(
            PORI_DEM_PATH,
            engine="rasterio",
            chunks={"x": args.chunks, "y": args.chunks},
            mask_and_scale=False,
        )
        # rasterio engine returns a Dataset with a "band_data" variable; drop band dim.
        if "band_data" in ds:
            ds = ds.rename({"band_data": "elevation"})
        if "band" in ds.elevation.dims:
            ds = ds.isel(band=0).drop_vars("band", errors="ignore")
        # Mask NoData explicitly (we kept mask_and_scale=False to inspect raw values).
        nodata = ds.elevation.attrs.get("_FillValue", PORI_NODATA)
        ds["elevation"] = ds.elevation.where(ds.elevation != nodata)

        print(f"  source CRS:    {ds.rio.crs}")
        print(f"  shape:         {dict(ds.sizes)}")
        print(f"  bounds:        {ds.rio.bounds()}")
        print(f"  nodata used:   {nodata}")
        print(f"  dask chunks:   {ds.elevation.chunksizes}")

    # ---------------------------------------------------------------- stage 2
    csf = clipper_scale_factor_for(args.level)
    print(f"\nclipper_scale_factor for level {args.level}: {csf}")
    print(f"DGGRID meta config (excerpt): dggs_vert0_lon={IGEO7_META['dggs_vert0_lon']}")

    with Timer(timings, "mapblocks_regridding"):
        z7_ds = mapblocks_regridding(
            ds=ds[["elevation"]],
            grid_name="igeo7",
            method="mapblocks_nearestcentroid",
            refinement_level=args.level,
            zone_id_repr="int",
            wgs84_geodetic_conversion=True,
            dggs_vert0_lon=IGEO7_META["dggs_vert0_lon"],
            sort_index=True,           # sort by zone_id ascending (uint64)
            clipper_scale_factor=csf,  # passes through **dggrid_kwargs
        )
        print(f"  regridded dims: {dict(z7_ds.sizes)}")
        print(f"  zone_id dtype:  {z7_ds.zone_id.dtype}")

    # ---------------------------------------------------------------- stage 3
    with Timer(timings, "to_uint64_and_aggregate"):
        # zone_id may already be uint64; coerce defensively.
        cell_ids = z7_ds.zone_id.values.astype(np.uint64)
        elevation = z7_ds.elevation.values.astype(np.float32)
        # Drop any NoData rows that survived (NaN in elevation).
        valid = ~np.isnan(elevation)
        cell_ids = cell_ids[valid]
        elevation = elevation[valid]
        # Multiple raster pixels can land in the same Z7 cell; aggregate by mean.
        # Using pandas groupby for a one-shot zonal mean (fast, in-memory; Pori is small).
        import pandas as pd
        agg = (
            pd.DataFrame({"cell_id": cell_ids, "elevation": elevation})
            .groupby("cell_id", sort=False)["elevation"]
            .mean()
        )
        cell_ids = agg.index.to_numpy(dtype=np.uint64)
        elevation = agg.to_numpy(dtype=np.float32)
        print(f"  unique cells after zonal mean: {cell_ids.size:,}")

    # ---------------------------------------------------------------- stage 4
    with Timer(timings, "sort_by_monotonic_int"):
        monotonic = z7_zarr.z7_to_monotonic_int_batch(cell_ids, args.level)
        order = np.argsort(monotonic, kind="stable")
        cell_ids = cell_ids[order]
        elevation = elevation[order]

    # ---------------------------------------------------------------- stage 5
    with Timer(timings, "find_monotonic_ranges"):
        # Phase-1 reporting only; not stored in the dggs/none archive.
        range_start_z7, range_end_z7 = z7_zarr.find_monotonic_ranges(cell_ids, args.level)
        n, r = cell_ids.size, range_start_z7.size
        # Range lengths in monotonic-int space:
        starts_mono = z7_zarr.z7_to_monotonic_int_batch(range_start_z7, args.level)
        ends_mono   = z7_zarr.z7_to_monotonic_int_batch(range_end_z7,   args.level)
        avg_len = (ends_mono.astype(np.int64) - starts_mono.astype(np.int64) + 1).mean() if r else 0
        print(f"  N (cells)            : {n:,}")
        print(f"  R (ranges)           : {r:,}")
        print(f"  R/N compression      : {r / n:.4%}")
        print(f"  mean range length    : {avg_len:.1f}")

    # ---------------------------------------------------------------- stage 6
    with Timer(timings, "zarr_write"):
        z7_zarr.write(
            out_path,
            cell_ids=cell_ids,
            data={"elevation": elevation},
            level=args.level,
            compression=args.compression,
            dggs_vert0_lon=IGEO7_META["dggs_vert0_lon"],
            dggs_vert0_lat=IGEO7_META["dggs_vert0_lat"],
            dggs_vert0_azimuth=IGEO7_META["dggs_vert0_azimuth"],
            extra_attrs={
                "source_path": str(PORI_DEM_PATH),
                "source_crs":  str(ds.rio.crs),
                "regridder":   "xdggs_dggrid4py.mapblocks_nearestcentroid",
                "clipper_scale_factor": int(csf),
            },
        )
        print(f"  wrote {out_path}  (compression={args.compression})")

    # ---------------------------------------------------------------- stage 7
    with Timer(timings, "zarr_roundtrip"):
        loaded = z7_zarr.open_dataset(out_path, decode=True)
        loaded_elev  = loaded["elevation"].values
        assert np.allclose(loaded_elev, elevation, equal_nan=True), "elevation round-trip mismatch"
        dggs_attrs = loaded.attrs.get("dggs", {})
        assert dggs_attrs.get("name")             == "igeo7"
        assert dggs_attrs.get("refinement_level") == args.level
        assert dggs_attrs.get("compression")      == args.compression

        idx = loaded.xindexes["cell_ids"]
        if args.compression == "none":
            from xdggs_dggrid4py.index import IGEO7Index
            assert isinstance(idx, IGEO7Index), \
                f"expected IGEO7Index for compression=none, got {type(idx).__name__}"
            loaded_cells = loaded["cell_ids"].values
            assert np.array_equal(loaded_cells, cell_ids), "cell_ids round-trip mismatch"
        else:
            from xdggs_dggrid4py.monotonic_index import Z7MonotonicIndex
            assert isinstance(idx, Z7MonotonicIndex), \
                f"expected Z7MonotonicIndex for compression=ranges, got {type(idx).__name__}"
            # Spot-check: a few sel queries hit the right positions.
            sample_cells = cell_ids[[0, n // 3, n // 2, n - 1]]
            sel_result = idx.sel({"cell_ids": sample_cells})
            sel_positions = sel_result.dim_indexers["cell_ids"]
            expected = np.array([0, n // 3, n // 2, n - 1])
            assert np.array_equal(sel_positions, expected), (
                f"sel positions mismatch: got {sel_positions}, expected {expected}"
            )
            # And isel materialises back to the same Z7 IDs.
            full = idx.values()
            assert np.array_equal(full, cell_ids), "values() reconstruction mismatch"

        print(f"  round-trip OK (index={idx._repr_inline_(80)})")
        print(f"  group attrs : {sorted(loaded.attrs.keys())}")
        print(f"  dggs.compression: {dggs_attrs.get('compression')!r}")

    # ---------------------------------------------------------------- summary
    print("\n=== timings ===")
    for k, v in timings.items():
        print(f"  {k:<28s} {v:7.2f}s")
    total = sum(timings.values())
    print(f"  {'TOTAL':<28s} {total:7.2f}s")

    print("\n=== output ===")
    print(f"  {out_path}")
    du = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    print(f"  on-disk size: {du / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
