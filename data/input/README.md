# data/input/

Source rasters used as inputs to the Z7/IGEO7 regridding pipeline
(`scripts/demo_regrid_pori_to_z7.py`, `scripts/build_eesti_dem_res12_zarr.py`,
Phase 1-3 of `AGENT.md`). Everything under `data/working/` and `data/output/`
that mentions "elevation", "slope", or "Pori"/"Eesti" DEM is derived from
these files.

| File | Description |
|---|---|
| `merit_dem_pori_cog.tif` | MERIT DEM tile over the Porijõgi catchment, Estonia. Cloud-Optimized GeoTIFF, CRS EPSG:3301 (Estonian National Grid), NoData = -9999. The primary/original test DEM for the whole project — sampled in its native source CRS per project convention (never reprojected before DGGS regridding). Feeds `pori_z7_r10*.zarr` and `pori_z7_r12*.zarr`. |
| `Copernicus_DSM_COG_10_N58_00_E026_00_DEM.tif` | Copernicus GLO-30 DSM tile (10 m COG), covering the N58/E026 tile (Tartu, Estonia area). Larger/second test tile, flagged in `AGENT.md` as a deferred stress-test input — not yet exercised through the pipeline as of this writing. |
| `Copernicus_DSM_COG_10_N58_00_E026_00_WBM.tif` | Water Body Mask companion raster for the Copernicus DSM tile above (same footprint/grid). |

## License

These are **third-party source datasets**, not project-generated outputs —
do **not** apply the project's CC-BY-4.0 default to this folder without
checking the upstream terms first:

- **MERIT DEM** (`merit_dem_pori_cog.tif`): distributed by the original
  author (Yamazaki et al.) under a research/education license that
  historically restricts redistribution without permission. Verify current
  terms at the MERIT Hydro/MERIT DEM homepage before including this file in
  a public Zenodo deposit; consider depositing only the derived Z7/Zarr
  products instead of the raw tile if redistribution is not clearly
  permitted.
- **Copernicus DEM GLO-30** (`Copernicus_DSM_COG_10_*`): distributed under
  the Copernicus DEM End User License Agreement, not CC-BY-4.0. It permits
  free use and redistribution with attribution, but check the exact clause
  in force at deposit time.
