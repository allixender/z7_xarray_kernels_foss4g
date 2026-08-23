"""Convert the Estonia DEM zonal-stats Parquet (res12) into IGEO7/Z7 Zarr archives.

Source: ``data/working/igeo7_zonal_res12_eesti_dem_merged.parquet`` — DEM
elevation zonal statistics already aggregated onto the IGEO7 res12 grid
(9 double stat columns + a ``z7int`` uint64 cell id + a redundant ``name``
string, dropped on read).

Writes two archives, mirroring the ``pori_z7_r10.zarr`` / ``pori_z7_r10_ranges.zarr``
pair but at full-Estonia / res12 scale (~11.9M cells) so storage-layout
experiments (HANDOFF Task C) have a realistic archive to work with instead of
Pori's 158k-cell / 2-chunk toy case:

    data/working/eesti_z7_r12.zarr          compression="none"   (dense cell_ids)
    data/working/eesti_z7_r12_ranges.zarr   compression="ranges" (Z7MonotonicIndex)

Chunk size is 7**8 = 5,764,801 cells (not the project default 7**5): at this
row count that lands each float32 variable chunk at ~23 MB and the uint64
cell-id coordinate chunk at ~46 MB, both inside the requested 10-50 MB band,
while still splitting the archive into 3 chunks per array instead of 1.

Run: pixi run python scripts/build_eesti_dem_res12_zarr.py
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pyarrow.parquet as pq

from eesti_soil_conversion.sort import build_sort
from z7_xarray_paper.config import DATA_WORKING
from z7_xarray_paper.z7_zarr import write, open_dataset, expand_monotonic_ranges

LEVEL = 12
CHUNK_CELLS = 7**8  # 5,764,801 -> ~23 MB float32 / ~46 MB uint64 per chunk

SOURCE_PARQUET = DATA_WORKING / "igeo7_zonal_res12_eesti_dem_merged.parquet"
DENSE_PATH = DATA_WORKING / "eesti_z7_r12.zarr"
RANGES_PATH = DATA_WORKING / "eesti_z7_r12_ranges.zarr"

STAT_COLUMNS = (
    "mean", "median", "count", "majority", "stdev",
    "coefficient_of_variation", "variance", "min", "max",
)


def _dir_size_bytes(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(overwrite: bool = False) -> None:
    for path in (DENSE_PATH, RANGES_PATH):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite existing path: {path}")
            shutil.rmtree(path)

    t0 = time.perf_counter()
    print(f"reading {SOURCE_PARQUET} ...")
    table = pq.read_table(SOURCE_PARQUET, columns=["z7int", *STAT_COLUMNS])
    n = table.num_rows
    print(f"  {n:,} rows")

    z7int = table.column("z7int").to_numpy(zero_copy_only=False).astype(np.uint64)
    uniq = np.unique(z7int)
    if uniq.size != n:
        raise ValueError(f"duplicate cell ids: {n:,} rows -> {uniq.size:,} unique")

    print(f"sorting into Z7 monotonic order (level={LEVEL}) ...")
    sort = build_sort(z7int, LEVEL)
    del z7int
    print(f"  R={sort.n_ranges:,} ranges ({sort.n_ranges / n:.4%} of N)")

    order = sort.order
    sorted_cell_ids = expand_monotonic_ranges(sort.range_table, LEVEL)
    assert sorted_cell_ids.shape[0] == n

    data = {}
    for name in STAT_COLUMNS:
        col = table.column(name)
        if col.null_count:
            arr = col.to_pandas(zero_copy_only=False).to_numpy(np.float64)
        else:
            arr = col.to_numpy(zero_copy_only=False).astype(np.float64)
        data[f"elevation_{name}"] = arr.astype(np.float32)[order]
    del table, order

    extra_attrs = {
        "source_path": str(SOURCE_PARQUET),
        "source_rows": int(n),
        "regridder": "external zonal-stats aggregation onto IGEO7 res12",
        "data_dtype": "float32",
        "converter": "scripts/build_eesti_dem_res12_zarr.py",
    }

    print(f"writing dense (compression='none') archive -> {DENSE_PATH}")
    write(
        DENSE_PATH,
        cell_ids=sorted_cell_ids,
        data=data,
        level=LEVEL,
        compression="none",
        chunk_cells=CHUNK_CELLS,
        extra_attrs=extra_attrs,
    )

    print(f"writing ranges (compression='ranges') archive -> {RANGES_PATH}")
    write(
        RANGES_PATH,
        cell_ids=sorted_cell_ids,
        data=data,
        level=LEVEL,
        compression="ranges",
        chunk_cells=CHUNK_CELLS,
        extra_attrs=extra_attrs,
    )

    wall = time.perf_counter() - t0

    dense_bytes = _dir_size_bytes(DENSE_PATH)
    ranges_bytes = _dir_size_bytes(RANGES_PATH)
    dense_cellids_bytes = n * 8  # uncompressed uint64 dense coord
    ranges_coord_bytes = sort.n_ranges * 2 * 8  # uncompressed (R,2) uint64

    print("\n=== summary ===")
    print(f"N cells        : {n:,}")
    print(f"R ranges       : {sort.n_ranges:,} ({sort.n_ranges / n:.4%} of N)")
    print(f"chunk_cells    : {CHUNK_CELLS:,} (7^8)")
    print(f"wall           : {wall:.2f}s")
    print(f"dense archive  : {DENSE_PATH} = {dense_bytes / 1e6:.1f} MB on disk")
    print(f"ranges archive : {RANGES_PATH} = {ranges_bytes / 1e6:.1f} MB on disk")
    print(
        "ranges-vs-dense coord (uncompressed): "
        f"{ranges_coord_bytes / 1e6:.3f} MB vs {dense_cellids_bytes / 1e6:.1f} MB "
        f"-> {ranges_coord_bytes / dense_cellids_bytes:.4%}"
    )

    print("\nvalidating round-trip open ...")
    ds_dense = open_dataset(DENSE_PATH, decode=True)
    ds_ranges = open_dataset(RANGES_PATH, decode=True)
    assert ds_dense.sizes["cell_ids"] == n
    assert ds_ranges.sizes["cell_ids"] == n
    np.testing.assert_array_equal(
        np.asarray(ds_dense["elevation_mean"].values[:1000]),
        np.asarray(ds_ranges["elevation_mean"].values[:1000]),
    )
    print("OK: both archives open, N matches, spot-check of elevation_mean agrees.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
