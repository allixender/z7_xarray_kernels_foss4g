# data/working/

Intermediate and worked-example DGGS Zarr archives (IGEO7/Z7 grid, [DGGS Zarr
Convention v1](https://github.com/zarr-conventions/dggs/blob/v1/README.md))
plus supporting lookup tables. These are the archives of the FOSS4G paper build on and benchmark against. Each `.zarr` is a
directory store (Zarr v2) — deposit it as a zipped folder on Zenodo, not as
individual files.

Every archive uses the IGEO7 grid definition
(`dggs_vert0_lon=11.2`, `dggs_vert0_lat=58.28252559`, WGS84 ellipsoid,
`rotation_pattern=alternating_cw_odd_ccw_even`). `compression: "none"` means
the array is indexed by a dense `cell_ids` coordinate (`IGEO7Index`);
`compression: "ranges"` means it is indexed by a compact `(R,2)` sorted-range
table (`Z7MonotonicIndex`) — same data, far smaller coordinate metadata,
requires the project's `z7_xarray_paper.z7_zarr.open_dataset()` (or the
`xdggs-dggrid4py` plugin) to open correctly.

## Pori DEM archives (Porijõgi catchment, from `merit_dem_pori_cog.tif`)

| File | Description |
|---|---|
| `pori_z7_r10.zarr` | Pori DEM regridded to IGEO7 res10 (3,101 cells), dense cell-id encoding. Carries `elevation`, plus `slope_geodesic` and `slope_lookup` — FDA slope computed with per-cell geodesic distances vs. the level-4 distortion-lookup distances, the two variants compared in the synthetic validation study (Task E). |
| `pori_z7_r10_ranges.zarr` | Same res10 archive, ranges-compressed. Carries `elevation` and `slope_lookup`. |
| `pori_z7_r12.zarr` | Pori DEM at IGEO7 res12 (~158k cells), dense encoding. The original "toy scale" demo archive (`demo_regrid_pori_to_z7.py`), used throughout Phase 1-3 plumbing and the before/after njit-parallel benchmark. |
| `pori_z7_r12_ranges.zarr` | Same res12 archive, ranges-compressed. |

## Estonia DEM archives (`igeo7_zonal_res12_eesti_dem_merged.parquet`)

| File | Description |
|---|---|
| `eesti_z7_r12.zarr` | Full-Estonia DEM zonal statistics on IGEO7 res12 (11,853,867 cells), dense encoding. Variables: `elevation_{mean,median,min,max,stdev,variance,count,majority,coefficient_of_variation}` (9 zonal stats over each res12 cell's source DEM pixels). Production-scale archive for the storage-layout benchmark (`bench_storage.py`, paper Section 5.2). |
| `eesti_z7_r12_ranges.zarr` | Same archive, ranges-compressed; chunked at `7**8` cells/chunk (3 chunks). Source for the end-to-end slope benchmark and the worked slope example below. |
| `eesti_z7_r12_slope_ranges.zarr` | **The FOSS4G paper's worked example output.** Slope (FDA, units m/m) computed from `elevation_mean` over the full Estonia res12 archive (`scripts/slope_eesti_r12_blocked.py`), written back as a ranges-compressed archive paired with `eesti_z7_r12_ranges.zarr`. 78,339 of 11,853,867 cells are `NaN` (AOI-boundary and pentagon cells). Intended for standalone publication alongside its source archive. |


## Other files

| File | Description |
|---|---|
| `dist_lookup_level4.parquet` | Per-cell axis-distance correction lookup table at IGEO7 res4, derived from the global level-4 ISEA7H anisotropy sweep. Used by the slope FDA kernel in `distance_mode="lookup"` to correct neighbour distances for icosahedral shape distortion. |
| `igeo7_zonal_res12_eesti_dem_merged.parquet` | Intermediate zonal-statistics table: Estonia DEM elevation aggregated onto IGEO7 res12 cells by an external zonal-stats step (9 stat columns + `z7int` cell id + a redundant `name` string). Source for `eesti_z7_r12*.zarr` via `scripts/build_eesti_dem_res12_zarr.py`. |

## License

CC-BY-4.0 for the project-computed structure and derived statistics
(zonal-stat aggregation, ranges encoding, slope kernel output, distance
lookup table). The elevation *values* underneath `pori_z7_r10*`,
`pori_z7_r12*`, `eesti_z7_r12*`, and `eesti_z7_r12_slope_ranges.zarr` are
carried over from the MERIT DEM source tile — check MERIT DEM's
redistribution terms (see `data/input/README.md`) before depositing these
archives publicly.
