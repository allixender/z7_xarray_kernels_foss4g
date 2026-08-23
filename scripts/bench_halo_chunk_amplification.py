"""Read amplification of a 1-ring DGGS kernel as a function of chunk size.

Motivation
----------
Fixing the halo read (RESULTS_MEMO.md Task D-bis, defect 2) showed that the
cost of a chunk-parallel DGGS kernel is governed by **how many storage-chunk
decompressions the whole task graph performs**, not by FLOPs or by bytes
requested. Every task must read its own cells plus its 1-ring halo, and the
halo reaches into other chunks — so across the DAG the same storage chunk is
decompressed once per task that touches it.

This script measures that directly, and machine-independently, the same way
`bench_storage.py` counts chunk touches for point queries:

    amplification = (total storage-chunk touches across all tasks)
                    / (number of storage chunks)

An amplification of 1.0 would mean every chunk is decompressed exactly once
(the ideal a perfect scheduler would achieve). Anything above that is
repeated work that better graph construction could in principle remove.

Counting is analytic — derived from the neighbour table and the range table,
not from timings — so the numbers are properties of the grid, the ordering
and the chunking, reproducible on any machine.

Run: pixi run python scripts/bench_halo_chunk_amplification.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from z7_xarray_paper import z7_zarr
from z7_xarray_paper.config import DATA_OUTPUT, DATA_WORKING
from z7_xarray_paper.kernels.halo import positions_from_cell_ids, range_table_lookups
from z7_xarray_paper.kernels.neighbours import get_neighbours_batch

SOURCE = DATA_WORKING / "eesti_z7_r12_ranges.zarr"
CHUNK_EXPONENTS = (4, 5, 6, 7, 8)


def main() -> None:
    ds = z7_zarr.open_dataset(SOURCE, decode=True)
    idx = ds.xindexes["cell_ids"]
    level = idx.grid_info.level
    start_mono, end_mono, offsets = range_table_lookups(idx.range_table, level)
    cell_ids = np.asarray(idx.values(), dtype=np.uint64)
    n = cell_ids.size
    print(f"{SOURCE.name}: N={n:,}  level={level}  R={idx.range_table.shape[0]:,}")

    print("computing neighbour table and global positions once ...")
    nbrs = get_neighbours_batch(cell_ids)
    gpos = positions_from_cell_ids(
        nbrs.ravel(), start_mono, end_mono, offsets, level
    ).reshape(nbrs.shape)
    present = gpos >= 0
    print(f"  {int(present.sum()):,} of {gpos.size:,} neighbour slots resolve inside the AOI")

    rows = []
    for k in CHUNK_EXPONENTS:
        cc = 7**k
        n_chunks = int(np.ceil(n / cc))
        own = np.arange(n, dtype=np.int64) // cc          # each cell's chunk
        nbr_chunk = np.where(present, gpos // cc, -1)     # each neighbour's chunk

        # A task touches its own chunk plus every distinct chunk any of its
        # cells' neighbours live in.
        foreign = nbr_chunk != own[:, None]
        pairs = np.stack([
            np.repeat(own, 6)[(foreign & present).ravel()],
            nbr_chunk[foreign & present],
        ], axis=1)
        distinct_foreign = (
            np.unique(pairs, axis=0).shape[0] if pairs.size else 0
        )
        touches = n_chunks + distinct_foreign            # own reads + halo reads
        amplification = touches / n_chunks

        halo_cells = int((foreign & present).sum())
        rows.append({
            "chunk_exponent": k,
            "chunk_cells": cc,
            "n_chunks": n_chunks,
            "chunk_touches_total": int(touches),
            "amplification": amplification,
            "foreign_chunk_reads": int(distinct_foreign),
            "halo_cell_refs": halo_cells,
            "halo_to_core_ratio": halo_cells / n,
            "mean_foreign_chunks_per_task": distinct_foreign / n_chunks,
        })
        print(f"  7^{k}={cc:>9,}  chunks={n_chunks:>6,}  touches={touches:>8,}  "
              f"amplification={amplification:>6.2f}x  "
              f"halo/core={halo_cells / n:>7.2%}  "
              f"foreign chunks/task={distinct_foreign / n_chunks:>6.2f}")

    df = pd.DataFrame(rows)
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_OUTPUT / "bench_halo_chunk_amplification.parquet", index=False)
    df.to_csv(DATA_OUTPUT / "bench_halo_chunk_amplification.csv", index=False)
    (DATA_OUTPUT / "bench_halo_chunk_amplification.json").write_text(
        json.dumps({"source": str(SOURCE), "n_cells": int(n), "rows": rows}, indent=2)
    )

    _plot(df)
    print(f"\nwrote {DATA_OUTPUT/'bench_halo_chunk_amplification.parquet'} and "
          f"{DATA_OUTPUT/'fig_halo_chunk_amplification.png'}")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6.6, 4.3))
    ax1.plot(df.n_chunks, df.amplification, marker="o", color="tab:red",
             label="read amplification")
    ax1.axhline(1.0, ls="--", color="grey", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("number of chunks (= available task parallelism)")
    ax1.set_ylabel("storage-chunk decompressions / chunk", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df.n_chunks, 100 * df.halo_to_core_ratio, marker="s",
             color="tab:blue", label="halo / core")
    ax2.set_ylabel("halo cell references (% of N)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("1-ring kernel: parallelism vs. repeated chunk loads")
    fig.tight_layout()
    fig.savefig(DATA_OUTPUT / "fig_halo_chunk_amplification.png", dpi=200)


if __name__ == "__main__":
    main()
