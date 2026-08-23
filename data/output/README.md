# data/output/

Benchmark results, validation statistics, figures, and IGEO7 cell-geometry
exports produced by the scripts in `scripts/`. These are the numbers and
plots referenced by the FOSS4G Europe 2026 paper's Results section.

## Benchmarks (paper Section 5)

| File(s) | Produced by | Description |
|---|---|---|
| `bench_neighbours.csv`, `bench_neighbours.parquet`, `bench_neighbours_env.json`, `fig_bench_neighbours.png` | `scripts/bench_neighbours.py` | Section 5.1: throughput of the Z7 GBT neighbour kernels (pure Python, njit scalar, njit batch/parallel), tidy one-row-per-(impl, resolution, N, rep) table + environment metadata + summary figure. |
| `bench_storage.csv`, `bench_storage.parquet`, `bench_storage_env.json`, `fig_bench_storage.png` | `scripts/bench_storage.py` | Section 5.2: storage-layout efficiency of chunk-aligned vs. naive (round-robin) Zarr chunking of `eesti_z7_r12.zarr`'s elevation array, for chunk sizes `7**{2,3,4,5}`. |
| `bench_slope_before_after.json` | `scripts/bench_slope_before_after.py` | Slope kernel timing before/after the njit `prange` batch-parallel optimisation of the neighbour-fill step, each variant run in its own subprocess. |
| `bench_slope_end_to_end.csv`, `bench_slope_end_to_end.parquet`, `bench_slope_end_to_end.json`, `fig_bench_slope_end_to_end.png`, `fig_bench_slope_raster_vs_z7.png` | `scripts/bench_slope_end_to_end.py` | Section 5.3: end-to-end slope stage timing and eager-vs-blocked speedup on the Estonia r12 archive, plus a raster reference comparison (`gdaldem slope` / `xarray-spatial.slope`) against the Pori DEM. |
| `bench_pori_gdaldem_slope.tif` | `scripts/bench_slope_end_to_end.py` | `gdaldem slope` raster computed on `merit_dem_pori_cog.tif`, the raster-pipeline reference baseline for the above comparison. |
| `bench_slope_eesti_blocked.csv`, `bench_slope_eesti_blocked.parquet`, `bench_slope_eesti_blocked.json`, `fig_bench_slope_eesti_blocked.png` | `scripts/slope_eesti_r12_blocked.py` | Chunk-parallel `slope_blocked()` vs. eager `slope()` timing over the full Estonia res12 archive, at several matched storage/compute chunkings plus one deliberately misaligned pair. |
| `bench_halo_chunk_amplification.csv`, `bench_halo_chunk_amplification.parquet`, `bench_halo_chunk_amplification.json`, `fig_halo_chunk_amplification.png` | `scripts/bench_halo_chunk_amplification.py` | Read-amplification study: storage-chunk decompression touches per task, as a function of chunk size, for a 1-ring DGGS halo kernel. |

## Synthetic validation (paper Section 5, Task E)

| File(s) | Produced by | Description |
|---|---|---|
| `validate_slope_synthetic.csv`, `validate_slope_synthetic.parquet`, `validate_slope_synthetic_summary.json` | `scripts/validate_slope_synthetic.py` | FDA slope kernel validated against closed-form analytic slope (tilted plane, paraboloid) over the Pori AOI at IGEO7 res 10-13. |

## Global ISEA7H anisotropy sweep (Phase 3a)

| File(s) | Produced by | Description |
|---|---|---|
| `phase3a_global_l4_anisotropy.parquet` | `scripts/phase3a_global_band_sweep.py` | One row per global IGEO7 level-4 hex cell (~24,000 cells): per-axis geodesic neighbour distances, bearings, squeeze ratio, and squeezed-axis label. |
| `phase3a_global_l4_parent_summary.parquet`, `phase3a_global_l3_parent_summary.parquet` | same | Aggregated to level-3 hex parents (~3,420 cells): means/stds and modal squeezed axis per parent. |
| `phase3a_global_l5_anisotropy.parquet` | same | Companion sweep at level-5 resolution, same per-cell anisotropy metrics. |

## Raster reprojection/slope demo (`scripts/reproject_and_slope_dem.py`)

| File(s) | Description |
|---|---|
| `merit_dem_pori_4326.tif` (+ `.aux.xml`) | Pori DEM reprojected from EPSG:3301 to EPSG:4326, for the raster-pipeline comparison demo. |
| `merit_dem_pori_slope.tif` (+ `.aux.xml`) | `xarray-spatial`/raster slope computed on the reprojected DEM above. |
| `dem_4326.png`, `slope.png`, `dem_and_slope_analysis.png` | Matplotlib visualisations of the reprojected DEM and its slope. |

## IGEO7 multi-resolution cell geometries (`scripts/generate_igeo7_children.py`)

Polygon exports of IGEO7 cells for two seed cells (`0000003`, `0723233`) at
levels 5 (parent), 6, and 7 (children), via `dggrid4py`. Each pair is
exported in two coordinate flavours: `_authalic` (coordinates on the ISEA
authalic sphere, pre-ellipsoid correction, CRS-less) and `_wgs84` (geodetic
WGS84 lon/lat, EPSG:4326, with an added `z7_string` column).

| File(s) | Description |
|---|---|
| `igeo7_parent_0000003_l5_{authalic,wgs84}.gpkg`, `igeo7_parent_0723233_l5_{authalic,wgs84}.gpkg` | Level-5 parent cell polygon, one per seed cell. |
| `igeo7_level_6_0000003_l6_{authalic,wgs84}.gpkg`, `igeo7_level_6_0723233_l6_{authalic,wgs84}.gpkg` | Level-6 children of each seed cell. |
| `igeo7_level_7_0000003_l7_{authalic,wgs84}.gpkg`, `igeo7_level_7_0723233_l7_{authalic,wgs84}.gpkg` | Level-7 children of each seed cell. |
| `igeo7_children_0000003_l7_{authalic,wgs84}.gpkg`, `igeo7_children_0723233_l7_{authalic,wgs84}.gpkg`, `igeo7_children_all_l7_wgs84.gpkg` | Earlier-run naming variant of the same level-7 children generation (kept for provenance; superseded by the `igeo7_level_7_*` files). |
| `igeo7_all_cells_multi_resolution_wgs84.gpkg` | Combined export: all seed cells, all three levels, WGS84 coordinates, in one file. |

## License

CC-BY-4.0 for all files in this folder — every product here is computed by
this project's own code (benchmark timings, validation statistics, IGEO7
cell geometries, and their figures), independent of any third-party data
license. The reprojected DEM/slope rasters and PNGs in the "raster
reprojection/slope demo" section carry MERIT DEM elevation values and are
subject to the same caveat noted in `data/input/README.md`.
