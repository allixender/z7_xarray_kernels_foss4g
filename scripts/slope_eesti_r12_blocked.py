"""Chunk-parallel slope over the full Estonia r12 archive, and persist it.

Three jobs:

1. **Benchmark** `slope_blocked()` (rewritten 2026-08-16 to resolve halos
   inside each task from the (R,2) range table — see PHASE3.md §3.2 as-built)
   against eager `slope()` over all 11,853,867 Estonia r12 cells.

2. **Quantify storage/compute chunk alignment.** A Zarr read of *any* size
   costs a full storage-chunk decompression, so a compute chunk finer than
   the storage chunk re-decompresses the same block once per task. The
   source archive is written at 7**8 (3 chunks of 23 MB); running compute at
   7**5 against it means 706 tasks each paying a 23 MB decompression. We
   therefore write the DEM at several storage chunkings and run each with
   *matched* compute chunks (the honest scaling curve), plus one deliberately
   misaligned pair to put a number on the penalty.

3. **Persist** the result as `data/working/eesti_z7_r12_slope_ranges.zarr`,
   a `compression="ranges"` archive sitting parallel to its source
   `eesti_z7_r12_ranges.zarr`, intended for publication alongside the paper.

Run: pixi run python scripts/slope_eesti_r12_blocked.py
     pixi run python scripts/slope_eesti_r12_blocked.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time

import dask
import numpy as np
import pandas as pd

from z7_xarray_paper import z7_zarr
from z7_xarray_paper.config import DATA_OUTPUT, DATA_WORKING, IGEO7_META
from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7_xarray_paper.kernels.slope import slope, slope_blocked

SOURCE = DATA_WORKING / "eesti_z7_r12_ranges.zarr"
SOURCE_VAR = "elevation_mean"
SLOPE_ARCHIVE = DATA_WORKING / "eesti_z7_r12_slope_ranges.zarr"
DIST_LOOKUP = DATA_WORKING / "dist_lookup_level4.parquet"
SCRATCH = DATA_WORKING / "_bench_blocked_storage"

# Storage chunkings to test, each run with compute chunks matched to it.
# 7**8 is the source archive's own layout (3 chunks); 7**6 gives 101 chunks,
# which is the sensible range for a 12-core box.
ALIGNED_CHUNKS = (7**8, 7**7, 7**6)
THREAD_SWEEP = (1, 2, 4, 8, 12)
THREAD_CHUNK = 7**6


def _dir_size_bytes(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _write_source_at(chunk_cells: int, cell_ids, values, level) -> tuple:
    """Write a single-variable DEM archive at a given storage chunking."""
    path = SCRATCH / f"dem_chunk{chunk_cells}.zarr"
    if path.exists():
        shutil.rmtree(path)
    z7_zarr.write(
        path, cell_ids=cell_ids, data={"elevation": values}, level=level,
        compression="ranges", chunk_cells=chunk_cells,
    )
    return path, "elevation"


def _run_blocked(da, model, source, chunk_cells, threads):
    blocked = slope_blocked(
        da, model, distance_mode="lookup", source=source, chunk_cells=chunk_cells,
    )
    n_chunks = len(blocked.data.chunks[0])
    t0 = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=threads):
        vals = np.asarray(blocked.compute().values)
    return vals, n_chunks, time.perf_counter() - t0


def main(quick: bool = False) -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    ds = z7_zarr.open_dataset(SOURCE, decode=True)
    idx = ds.xindexes["cell_ids"]
    level = idx.grid_info.level
    n = int(ds.sizes["cell_ids"])
    model = HexGridDistortionModel(pd.read_parquet(DIST_LOOKUP).to_dict("index"))
    da = ds[SOURCE_VAR]

    print(f"source : {SOURCE}")
    print(f"         N={n:,}  level={level}  R={idx.range_table.shape[0]:,}")
    print(f"         storage chunking on disk: {da.data.chunks[0][:1]} "
          f"({len(da.data.chunks[0])} chunks)")

    aligned = ALIGNED_CHUNKS[:2] if quick else ALIGNED_CHUNKS
    threads = (1, 12) if quick else THREAD_SWEEP

    rows: list[dict] = []

    # ---- eager reference -------------------------------------------------
    print("\n=== eager slope() ===")
    t0 = time.perf_counter()
    eager_vals = np.asarray(slope(da, model, distance_mode="lookup").values)
    eager_s = time.perf_counter() - t0
    print(f"  {eager_s:.2f}s  ({n / eager_s:,.0f} cells/s)")
    rows.append({
        "case": "eager", "storage_chunk": None, "compute_chunk": None,
        "n_chunks": 1, "threads": 1, "wall_s": eager_s, "cells_per_s": n / eager_s,
        "n_nan": int(np.isnan(eager_vals).sum()), "identical_to_eager": True,
    })

    cell_ids = np.asarray(idx.values(), dtype=np.uint64)
    dem_vals = np.asarray(da.values, dtype=np.float32)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # ---- aligned storage/compute chunking --------------------------------
    print("\n=== slope_blocked(), storage chunk == compute chunk (12 threads) ===")
    reference = None
    for cc in aligned:
        src = _write_source_at(cc, cell_ids, dem_vals, level)
        vals, n_chunks, wall = _run_blocked(da, model, src, cc, 12)
        identical = np.array_equal(
            np.nan_to_num(vals, nan=-9999.0), np.nan_to_num(eager_vals, nan=-9999.0)
        )
        if not identical:
            raise SystemExit(f"blocked result differs from eager at chunk_cells={cc}")
        if reference is None:
            reference = vals
        print(f"  chunk={cc:>9,}  n_chunks={n_chunks:>5,}  {wall:>7.2f}s  "
              f"({n / wall:>12,.0f} cells/s)  speedup_vs_eager={eager_s / wall:.2f}x")
        rows.append({
            "case": "aligned", "storage_chunk": int(cc), "compute_chunk": int(cc),
            "n_chunks": n_chunks, "threads": 12, "wall_s": wall,
            "cells_per_s": n / wall, "n_nan": int(np.isnan(vals).sum()),
            "identical_to_eager": True, "speedup_vs_eager": eager_s / wall,
        })

    # ---- deliberate misalignment ----------------------------------------
    print("\n=== misaligned: fine compute chunks against a coarse 7**8 store ===")
    coarse = SCRATCH / f"dem_chunk{7**8}.zarr", "elevation"
    if not coarse[0].exists():
        coarse = _write_source_at(7**8, cell_ids, dem_vals, level)
    mis_cc = 7**6
    vals, n_chunks, wall = _run_blocked(da, model, coarse, mis_cc, 12)
    aligned_ref = next(r for r in rows if r["case"] == "aligned"
                       and r["compute_chunk"] == mis_cc)
    print(f"  compute={mis_cc:,} vs storage={7**8:,}  n_chunks={n_chunks:,}  "
          f"{wall:.2f}s  ({n / wall:,.0f} cells/s)")
    print(f"  penalty vs the same compute chunking on a matched store: "
          f"{wall / aligned_ref['wall_s']:.1f}x slower")
    rows.append({
        "case": "misaligned", "storage_chunk": int(7**8), "compute_chunk": int(mis_cc),
        "n_chunks": n_chunks, "threads": 12, "wall_s": wall, "cells_per_s": n / wall,
        "n_nan": int(np.isnan(vals).sum()),
        "identical_to_eager": bool(np.array_equal(
            np.nan_to_num(vals, nan=-9999.0), np.nan_to_num(eager_vals, nan=-9999.0))),
        "penalty_vs_aligned": wall / aligned_ref["wall_s"],
    })

    # ---- thread sweep on a matched store ---------------------------------
    print(f"\n=== thread sweep (storage=compute={THREAD_CHUNK:,}) ===")
    src = SCRATCH / f"dem_chunk{THREAD_CHUNK}.zarr", "elevation"
    if not src[0].exists():
        src = _write_source_at(THREAD_CHUNK, cell_ids, dem_vals, level)
    serial_wall = None
    for nt in threads:
        vals, n_chunks, wall = _run_blocked(da, model, src, THREAD_CHUNK, nt)
        if nt == 1:
            serial_wall = wall
        speedup = serial_wall / wall
        print(f"  threads={nt:>3}  {wall:>7.2f}s  ({n / wall:>12,.0f} cells/s)  "
              f"speedup={speedup:.2f}x")
        rows.append({
            "case": "thread_sweep", "storage_chunk": int(THREAD_CHUNK),
            "compute_chunk": int(THREAD_CHUNK), "n_chunks": n_chunks, "threads": nt,
            "wall_s": wall, "cells_per_s": n / wall,
            "n_nan": int(np.isnan(vals).sum()),
            "speedup_vs_1_thread": speedup,
        })

    shutil.rmtree(SCRATCH, ignore_errors=True)

    # ---- persist ---------------------------------------------------------
    print(f"\n=== persist -> {SLOPE_ARCHIVE} ===")
    if SLOPE_ARCHIVE.exists():
        shutil.rmtree(SLOPE_ARCHIVE)
    slope_final = np.asarray(reference, dtype=np.float32)

    t0 = time.perf_counter()
    z7_zarr.write(
        SLOPE_ARCHIVE, cell_ids=cell_ids, data={"slope": slope_final}, level=level,
        compression="ranges", chunk_cells=7**8,
        dggs_vert0_lon=IGEO7_META["dggs_vert0_lon"],
        dggs_vert0_lat=IGEO7_META["dggs_vert0_lat"],
        dggs_vert0_azimuth=IGEO7_META["dggs_vert0_azimuth"],
        extra_attrs={
            "title": "Slope (FDA) on IGEO7/Z7 res12 over Estonia",
            "source_archive": SOURCE.name,
            "source_variable": SOURCE_VAR,
            "kernel": "z7_xarray_paper.kernels.slope.slope_blocked",
            "distance_mode": "lookup",
            "distance_lookup_table": DIST_LOOKUP.name,
            "units": "m/m",
            "long_name": "slope magnitude (FDA)",
            "nan_policy": "AOI-boundary and pentagon cells are NaN",
            "n_cells": int(n),
            "n_nan": int(np.isnan(slope_final).sum()),
            "producer": "scripts/slope_eesti_r12_blocked.py",
            "intended_publication": (
                "To be published as an open dataset alongside the FOSS4G Europe "
                "2026 paper, together with its source archive "
                "eesti_z7_r12_ranges.zarr."
            ),
        },
    )
    write_s = time.perf_counter() - t0
    size_b = _dir_size_bytes(SLOPE_ARCHIVE)
    print(f"  wrote in {write_s:.2f}s, {size_b / 1e6:.1f} MB on disk")

    back = z7_zarr.open_dataset(SLOPE_ARCHIVE, decode=True)
    rt = np.asarray(back["slope"].values)
    np.testing.assert_array_equal(np.isnan(rt), np.isnan(slope_final))
    print(f"  round-trip OK (index={back.xindexes['cell_ids']._repr_inline_(60)})")

    # ---- outputs ---------------------------------------------------------
    df = pd.DataFrame(rows)
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_OUTPUT / "bench_slope_eesti_blocked.parquet", index=False)
    df.to_csv(DATA_OUTPUT / "bench_slope_eesti_blocked.csv", index=False)

    finite = slope_final[~np.isnan(slope_final)]
    best = min((r for r in rows if r["case"] == "aligned"), key=lambda r: r["wall_s"])
    summary = {
        "machine": {"platform": platform.platform(),
                     "processor": platform.processor() or platform.machine()},
        "source_archive": str(SOURCE),
        "slope_archive": str(SLOPE_ARCHIVE),
        "slope_archive_bytes": int(size_b),
        "slope_archive_write_s": write_s,
        "n_cells": int(n), "n_ranges": int(idx.range_table.shape[0]),
        "n_nan": int(np.isnan(slope_final).sum()),
        "eager_wall_s": eager_s,
        "best_blocked_wall_s": best["wall_s"],
        "best_blocked_chunk_cells": best["compute_chunk"],
        "best_blocked_speedup_vs_eager": eager_s / best["wall_s"],
        "slope_stats_m_per_m": {
            "min": float(finite.min()), "median": float(np.median(finite)),
            "mean": float(finite.mean()), "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
        },
        "slope_stats_deg": {
            "median": float(np.degrees(np.arctan(np.median(finite)))),
            "mean": float(np.degrees(np.arctan(finite.mean()))),
            "p99": float(np.degrees(np.arctan(np.percentile(finite, 99)))),
        },
    }
    (DATA_OUTPUT / "bench_slope_eesti_blocked.json").write_text(json.dumps(summary, indent=2))

    _plot(df, eager_s)
    print(f"\nwrote {DATA_OUTPUT/'bench_slope_eesti_blocked.parquet'}, "
          f"{DATA_OUTPUT/'bench_slope_eesti_blocked.json'}, "
          f"{DATA_OUTPUT/'fig_bench_slope_eesti_blocked.png'}")


def _plot(df: pd.DataFrame, eager_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    al = df[df.case == "aligned"].sort_values("n_chunks")
    ax.plot(al.n_chunks, al.wall_s, marker="o", label="aligned storage/compute")
    mis = df[df.case == "misaligned"]
    if len(mis):
        ax.scatter(mis.n_chunks, mis.wall_s, marker="X", s=110, color="tab:red",
                   zorder=5, label="misaligned (fine compute, coarse store)")
    ax.axhline(eager_s, ls="--", color="grey", label=f"eager slope() = {eager_s:.1f}s")
    ax.set_xscale("log")
    ax.set_xlabel("number of chunks")
    ax.set_ylabel("wall-clock (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Estonia r12: chunking and the alignment penalty")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    th = df[df.case == "thread_sweep"].sort_values("threads")
    if len(th) > 1:
        ax.plot(th.threads, th.wall_s, marker="s", color="tab:green", label="measured")
        t1 = float(th.wall_s.iloc[0])
        ax.plot(th.threads, t1 / th.threads, ls=":", color="grey", label="linear")
        ax.set_xlabel("dask threads")
        ax.set_ylabel("wall-clock (s)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Thread scaling (chunk={THREAD_CHUNK:,})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(DATA_OUTPUT / "fig_bench_slope_eesti_blocked.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
