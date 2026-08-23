"""Task D — Section 5.3, end-to-end slope timing and a raster reference.

Two independent sub-experiments, deliberately on two different AOIs:

1. **Stage timing + eager-vs-blocked speedup** on the Task B archive
   (`data/working/eesti_z7_r12_ranges.zarr`, N=11,853,867, 3 chunks at
   chunk_cells=7**8). This is the only archive at HANDOFF's requested scale,
   so it is used for the chunk-parallel-speedup question even though 3
   chunks caps the theoretical speedup low; a second run reuses one of Task
   C's finer layouts (aligned_k4, chunk_cells=2,401, ~4,938 chunks) as a
   ranges archive to show the speedup story at a chunk count where it can
   actually show up.

2. **Raster reference** (`gdaldem slope` + `xarray_spatial.slope`) against
   `data/working/pori_z7_r12_ranges.zarr` / `data/input/merit_dem_pori_cog.tif`.
   Estonia's only local raster (`Copernicus_DSM_COG_10_N58_00_E026_00_DEM.tif`)
   covers a single 1x1 degree tile, a small fragment of the DEM archive's
   full-Estonia extent — not a matched pair — so the raster comparison uses
   Pori, where the archive and the 60 m source raster are the same AOI.

3. Pentagon/AOI-boundary NaN accounting on the Estonia archive.

Run: pixi run python scripts/bench_slope_end_to_end.py
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from z7_xarray_paper import z7_zarr
from z7_xarray_paper.config import DATA_INPUT, DATA_OUTPUT, DATA_WORKING, PORI_DEM_PATH, PORI_NODATA
from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7_xarray_paper.kernels.neighbours import get_neighbours_batch, neighbour_positions, INVALID
from z7_xarray_paper.kernels.slope import slope, slope_blocked

OUT = DATA_OUTPUT
DIST_LOOKUP_PATH = DATA_WORKING / "dist_lookup_level4.parquet"
ESTONIA_RANGES = DATA_WORKING / "eesti_z7_r12_ranges.zarr"
PORI_RANGES = DATA_WORKING / "pori_z7_r12_ranges.zarr"


def _model() -> HexGridDistortionModel:
    empirical = pd.read_parquet(DIST_LOOKUP_PATH).to_dict("index")
    return HexGridDistortionModel(empirical)


# ---------------------------------------------------------------------------
# 1. stage timing + eager vs blocked, on the Estonia archive
# ---------------------------------------------------------------------------

def stage_timed_eager(elevation_da: xr.DataArray, model) -> dict:
    idx = elevation_da.xindexes["cell_ids"]
    level = idx.grid_info.level
    cell_ids = elevation_da.coords["cell_ids"].values.astype(np.uint64)

    t0 = time.perf_counter()
    h = np.asarray(elevation_da.values, dtype=np.float64)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)
    t_neighbours = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = slope(elevation_da, model, distance_mode="lookup")
    _ = result.values  # force materialisation
    t_slope_total = time.perf_counter() - t0
    t_fda = max(t_slope_total - t_neighbours, 0.0)  # slope() redoes the neighbour step internally

    t0 = time.perf_counter()
    write_path = DATA_WORKING / "_bench_scratch_eager_slope.zarr"
    if write_path.exists():
        import shutil
        shutil.rmtree(write_path)
    z7_zarr.write(
        write_path,
        cell_ids=cell_ids,
        data={"slope": np.asarray(result.values, dtype=np.float32)},
        level=level, compression="ranges",
    )
    t_write = time.perf_counter() - t0
    import shutil
    shutil.rmtree(write_path)

    return {
        "path": "eager", "n_chunks": None,
        "t_read": t_read, "t_neighbours": t_neighbours, "t_halo": 0.0,
        "t_fda": t_fda, "t_write": t_write,
        "t_total": t_read + t_neighbours + t_fda + t_write,
        "n_nan": int(np.isnan(np.asarray(result.values)).sum()),
    }


def stage_timed_blocked(archive_path: Path, model, label: str,
                         var_name: str = "elevation") -> dict:
    import dask

    t0 = time.perf_counter()
    ds = z7_zarr.open_dataset(archive_path, decode=True)
    elevation_da = ds[var_name]
    if not hasattr(elevation_da.data, "chunks"):
        elevation_da = elevation_da.chunk({"cell_ids": elevation_da.sizes["cell_ids"]})
    t_read = time.perf_counter() - t0
    n_chunks = len(elevation_da.data.chunks[0])

    t0 = time.perf_counter()
    result = slope_blocked(elevation_da, model, distance_mode="lookup")
    t_graph_build = time.perf_counter() - t0  # includes per-chunk halo-manifest construction

    t0 = time.perf_counter()
    with dask.config.set(scheduler="threads"):
        computed = result.compute()
    t_compute = time.perf_counter() - t0

    t0 = time.perf_counter()
    cell_ids = ds["cell_ids"].values.astype(np.uint64)
    level = ds.xindexes["cell_ids"].grid_info.level
    write_path = DATA_WORKING / f"_bench_scratch_blocked_slope_{label}.zarr"
    if write_path.exists():
        import shutil
        shutil.rmtree(write_path)
    z7_zarr.write(
        write_path,
        cell_ids=cell_ids,
        data={"slope": np.asarray(computed.values, dtype=np.float32)},
        level=level, compression="ranges",
    )
    t_write = time.perf_counter() - t0
    import shutil
    shutil.rmtree(write_path)

    return {
        "path": f"blocked_{label}", "n_chunks": n_chunks,
        "t_read": t_read, "t_neighbours": None, "t_halo": t_graph_build,
        "t_fda": t_compute, "t_write": t_write,
        "t_total": t_read + t_graph_build + t_compute + t_write,
        "n_nan": int(np.isnan(np.asarray(computed.values)).sum()),
    }


def build_fine_ranges_archive() -> Path:
    """Reuse Task C's aligned_k4 raw layout (chunk_cells=2401) as a proper
    ranges archive with an `elevation` variable, so slope_blocked() has
    ~4,938 chunks to actually parallelise over."""
    out_path = DATA_WORKING / "_bench_eesti_r12_ranges_fine.zarr"
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)

    ds = z7_zarr.open_dataset(ESTONIA_RANGES, decode=True)
    idx = ds.xindexes["cell_ids"]
    cell_ids = np.asarray(idx.values(), dtype=np.uint64)
    elevation = np.asarray(ds["elevation_mean"].values, dtype=np.float32)
    level = idx.grid_info.level

    z7_zarr.write(
        out_path, cell_ids=np.asarray(cell_ids, dtype=np.uint64),
        data={"elevation": elevation}, level=level, compression="ranges",
        chunk_cells=2401,
    )
    return out_path


# ---------------------------------------------------------------------------
# 2. raster reference (Pori)
# ---------------------------------------------------------------------------

def raster_reference() -> dict:
    slope_tif = OUT / "bench_pori_gdaldem_slope.tif"
    t0 = time.perf_counter()
    subprocess.run(
        # default output unit is degrees (no -p); -compute_edges fills the border row/col
        # instead of leaving them NoData.
        ["gdaldem", "slope", str(PORI_DEM_PATH), str(slope_tif), "-compute_edges"],
        check=True, capture_output=True, text=True,
    )
    t_gdal = time.perf_counter() - t0

    import rioxarray  # noqa: F401
    # chunks= is required: mapblocks_regridding needs a dask-backed array
    # (it reads ds_dask_array.numblocks), a plain numpy DataArray fails there.
    raster_slope = xr.open_dataarray(
        slope_tif, engine="rasterio", chunks={"x": 512, "y": 512},
    ).squeeze("band", drop=True)
    n_raster_cells = int(raster_slope.size)

    # Z7 slope on Pori r12 ranges archive, timed for comparison.
    ds = z7_zarr.open_dataset(PORI_RANGES, decode=True)
    model = _model()
    t0 = time.perf_counter()
    z7_result = slope(ds["elevation"], model, distance_mode="lookup")
    _ = z7_result.values
    t_z7 = time.perf_counter() - t0
    n_z7_cells = int(ds.sizes["cell_ids"])

    # Regrid the raster slope (degrees) onto the same Z7 r12 cells for a distribution comparison.
    from xdggs_dggrid4py.regridding import mapblocks_regridding
    from z7_xarray_paper.config import IGEO7_META, clipper_scale_factor_for
    import rioxarray  # noqa: F401

    raster_slope_ds = xr.Dataset({"slope_deg": raster_slope})
    level = ds.xindexes["cell_ids"].grid_info.level
    csf = clipper_scale_factor_for(level)
    regridded = mapblocks_regridding(
        ds=raster_slope_ds, grid_name="igeo7", method="mapblocks_nearestcentroid",
        refinement_level=level, zone_id_repr="int", wgs84_geodetic_conversion=True,
        dggs_vert0_lon=IGEO7_META["dggs_vert0_lon"], sort_index=True,
        clipper_scale_factor=csf,
    )
    r_ids = regridded.zone_id.values.astype(np.uint64)
    r_slope_deg = regridded.slope_deg.values.astype(np.float64)
    valid = ~np.isnan(r_slope_deg)
    r_ids, r_slope_deg = r_ids[valid], r_slope_deg[valid]
    df_r = pd.DataFrame({"cell_id": r_ids, "slope_deg": r_slope_deg}).groupby(
        "cell_id", sort=False)["slope_deg"].mean()

    z7_cell_ids = ds["cell_ids"].values.astype(np.uint64)
    z7_slope_mm = np.asarray(z7_result.values)
    z7_slope_deg = np.degrees(np.arctan(z7_slope_mm))  # m/m -> degrees, matching gdaldem's default unit

    common = np.intersect1d(z7_cell_ids, df_r.index.to_numpy())
    z7_series = pd.Series(z7_slope_deg, index=z7_cell_ids).reindex(common)
    raster_series = df_r.reindex(common)
    diff = z7_series.to_numpy() - raster_series.to_numpy()
    valid_diff = ~np.isnan(diff)
    diff = diff[valid_diff]

    _plot_raster_comparison(raster_series.to_numpy(), z7_series.to_numpy(), diff)

    return {
        "gdal_wall_s": t_gdal,
        "gdal_cells_per_million_s": t_gdal / (n_raster_cells / 1e6),
        "z7_wall_s": t_z7,
        "z7_cells_per_million_s": t_z7 / (n_z7_cells / 1e6),
        "n_raster_cells": n_raster_cells,
        "n_z7_cells": n_z7_cells,
        "n_common_cells_compared": int(common.size),
        "raster_slope_deg_median": float(np.nanmedian(raster_series)),
        "raster_slope_deg_mean": float(np.nanmean(raster_series)),
        "raster_slope_deg_p99": float(np.nanpercentile(raster_series.dropna(), 99)),
        "z7_slope_deg_median": float(np.nanmedian(z7_series)),
        "z7_slope_deg_mean": float(np.nanmean(z7_series)),
        "z7_slope_deg_p99": float(np.nanpercentile(z7_series.dropna(), 99)),
        "diff_median": float(np.median(diff)) if diff.size else None,
        "diff_mean": float(np.mean(diff)) if diff.size else None,
        "diff_p99": float(np.percentile(diff, 99)) if diff.size else None,
        "diff_std": float(np.std(diff)) if diff.size else None,
    }


def _plot_raster_comparison(raster_deg: np.ndarray, z7_deg: np.ndarray, diff_deg: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    bins = np.linspace(0, max(np.nanpercentile(raster_deg, 99.5), np.nanpercentile(z7_deg, 99.5)), 60)
    ax.hist(raster_deg[~np.isnan(raster_deg)], bins=bins, alpha=0.6, label="gdaldem (raster)")
    ax.hist(z7_deg[~np.isnan(z7_deg)], bins=bins, alpha=0.6, label="z7_slope (lookup)")
    ax.set_xlabel("slope (degrees)")
    ax.set_ylabel("cell count")
    ax.set_title("Pori r12: slope distribution, raster vs. Z7")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(diff_deg, bins=60, color="tab:purple", alpha=0.8)
    ax.set_xlabel("Z7 slope − raster slope (degrees)")
    ax.set_ylabel("cell count")
    ax.set_title(f"per-cell difference (median={np.median(diff_deg):+.3f}°, "
                 f"std={np.std(diff_deg):.3f}°)")

    fig.tight_layout()
    fig.savefig(OUT / "fig_bench_slope_raster_vs_z7.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. pentagon / boundary accounting
# ---------------------------------------------------------------------------

def pentagon_boundary_accounting(archive_path: Path) -> dict:
    ds = z7_zarr.open_dataset(archive_path, decode=True)
    cell_ids = ds["cell_ids"].values.astype(np.uint64)
    n = cell_ids.size
    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)

    is_pentagon = (nbrs == INVALID).any(axis=1)
    # boundary: some neighbour ID is valid (not the pentagon sentinel) but not
    # present in this AOI's cell_ids (pos == -1 there).
    out_of_aoi = (pos < 0) & (nbrs != INVALID)
    is_boundary_only = out_of_aoi.any(axis=1) & ~is_pentagon

    return {
        "n_cells": int(n),
        "n_pentagon": int(is_pentagon.sum()),
        "n_boundary_only": int(is_boundary_only.sum()),
        "n_nan_total": int(is_pentagon.sum() + is_boundary_only.sum()),
        "pct_pentagon": float(is_pentagon.sum() / n * 100),
        "pct_boundary": float(is_boundary_only.sum() / n * 100),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=== 1. stage timing: eager vs blocked, Estonia r12 (Task B archive) ===")
    ds = z7_zarr.open_dataset(ESTONIA_RANGES, decode=True)
    model = _model()
    elevation_mean = ds["elevation_mean"]
    elevation_mean = elevation_mean.rename("elevation")
    eager = stage_timed_eager(elevation_mean, model)
    print(json.dumps(eager, indent=2))

    blocked = stage_timed_blocked(ESTONIA_RANGES, model, label="task_b_3chunks",
                                   var_name="elevation_mean")
    print(json.dumps(blocked, indent=2))

    results["stage_timing"] = [eager, blocked]
    (OUT / "bench_slope_end_to_end.json").write_text(json.dumps(results, indent=2))

    print("\n=== 1b. bonus: finer chunking (aligned_k4, ~4,938 chunks) ===")
    fine_path = build_fine_ranges_archive()
    blocked_fine = stage_timed_blocked(fine_path, model, label="fine_k4",
                                        var_name="elevation")
    print(json.dumps(blocked_fine, indent=2))
    results["stage_timing"].append(blocked_fine)
    import shutil
    shutil.rmtree(fine_path)
    (OUT / "bench_slope_end_to_end.json").write_text(json.dumps(results, indent=2))

    print("\n=== 2. raster reference (Pori) ===")
    try:
        results["raster_reference"] = raster_reference()
        print(json.dumps(results["raster_reference"], indent=2))
    except Exception as exc:
        results["raster_reference"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"raster_reference FAILED: {exc}")
    (OUT / "bench_slope_end_to_end.json").write_text(json.dumps(results, indent=2))

    print("\n=== 3. pentagon / boundary accounting (Estonia r12) ===")
    results["pentagon_boundary"] = pentagon_boundary_accounting(ESTONIA_RANGES)
    print(json.dumps(results["pentagon_boundary"], indent=2))

    (OUT / "bench_slope_end_to_end.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results["stage_timing"]).to_parquet(OUT / "bench_slope_end_to_end.parquet", index=False)
    pd.DataFrame(results["stage_timing"]).to_csv(OUT / "bench_slope_end_to_end.csv", index=False)

    _plot(results)
    print(f"\nwrote {OUT/'bench_slope_end_to_end.json'}, "
          f"{OUT/'bench_slope_end_to_end.parquet'}, {OUT/'fig_bench_slope_end_to_end.png'}")


def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results["stage_timing"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stages = ["t_read", "t_neighbours", "t_halo", "t_fda", "t_write"]
    bottom = np.zeros(len(df))
    for s in stages:
        vals = df[s].fillna(0).to_numpy()
        ax.bar(df["path"], vals, bottom=bottom, label=s)
        bottom += vals
    ax.set_ylabel("wall-clock (s)")
    ax.set_title("Slope: stage-by-stage wall-clock, eager vs. blocked")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bench_slope_end_to_end.png", dpi=200)


if __name__ == "__main__":
    main()
