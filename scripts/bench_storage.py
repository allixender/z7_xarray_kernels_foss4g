"""Task C — Section 5.2, storage layout efficiency (HANDOFF_LOCAL_CLAUDE.md).

Writes the Task B DEM data (`data/working/eesti_z7_r12.zarr`, N=11,853,867,
Estonia at IGEO7 res12) five ways from the same sorted (Z7-monotonic-order)
`elevation_mean` array:

- 4 "aligned" layouts: chunk_cells = 7**k for k in {2,3,4,5} (i.e. each
  chunk holds exactly one r-k parent's worth of leaf cells wherever the AOI
  is locally dense; near the AOI boundary a chunk may hold fewer cells but
  chunk *boundaries* still fall on parent boundaries because the source
  array is sorted in Z7 monotonic order).
- 1 "naive" control: same chunk byte size as the k=5 (7**5=16807, the
  project's existing default chunk size) aligned layout, but the array is
  reordered by `original_position % n_chunks` (round-robin / modulo
  partitioning) before writing, so physically adjacent Z7 cells are
  scattered across chunks near-uniformly at random. This is what you get if
  you chunk without sorting by the DGGS hierarchy at all.

For each layout, measures (>=20 reps, median reported) wall-clock and Zarr
**chunk touches** (via a store wrapper counting `__getitem__` calls on
chunk keys) for:

  1. single-cell lookup
  2. parent-zone retrieval, at each of the four r-k granularities
  3. cell + 6 neighbours, for an "interior" cell and a "worst-case" cell

On worst-case/seam cells: Estonia r12 sits entirely within one ISEA base
cell (verified separately — `unique base cells present: [0]`), so there is
no genuine inter-base-cell seam anywhere in this AOI (base cells span
~1/12 of Earth's surface, far larger than any single country; PHASE3.md's
plan to use "Tartu r11" would hit the same wall). We substitute the
empirical worst case within the AOI: the present cell whose sorted-array
position is farthest from any of its 6 neighbours' positions. This is a
*rotation-domain* locality break (the alternating_cw_odd_ccw_even pattern
flipping at a coarse subdivision boundary), not a literal face crossing,
and it is the honest worst case a real query on this archive can hit.

Page-cache clearing (HANDOFF item 4): `sudo -n purge` requires an
interactive password in this environment (confirmed, exit 1) and is not
attempted per rep. All timings below are **warm-cache** numbers, reported
as such.

Run: pixi run python scripts/bench_storage.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from z7_xarray_paper import z7_zarr
from z7_xarray_paper.config import DATA_WORKING, DATA_OUTPUT, default_compressor
from z7_xarray_paper.kernels.neighbours import get_neighbours_batch, neighbour_positions, INVALID
from z7py import z7

SOURCE_ARCHIVE = DATA_WORKING / "eesti_z7_r12.zarr"
LAYOUT_ROOT = DATA_WORKING / "bench_storage_layouts"
OUT = DATA_OUTPUT

LEVEL = 12
K_VALUES = (2, 3, 4, 5)
NAIVE_MATCH_K = 5  # naive control's chunk_cells matches this aligned layout
REPS = 25


# ---------------------------------------------------------------------------
# counting store
# ---------------------------------------------------------------------------

class CountingStore(zarr.DirectoryStore):
    """DirectoryStore that counts __getitem__ calls on chunk keys (not metadata)."""

    def __init__(self, path):
        super().__init__(str(path))
        self.reset()

    def reset(self):
        self.n_getitem = 0
        self.touched_keys = set()

    def __getitem__(self, key):
        if not key.startswith("."):
            self.n_getitem += 1
            self.touched_keys.add(key)
        return super().__getitem__(key)


# ---------------------------------------------------------------------------
# layout construction
# ---------------------------------------------------------------------------

def build_layouts(cell_ids: np.ndarray, values: np.ndarray) -> dict:
    """Write the 5 layouts; return dict name -> {"path", "chunk_cells", "kind"}."""
    if LAYOUT_ROOT.exists():
        shutil.rmtree(LAYOUT_ROOT)
    LAYOUT_ROOT.mkdir(parents=True)

    n = cell_ids.size
    compressor = default_compressor()
    layouts = {}

    for k in K_VALUES:
        chunk_cells = 7 ** k
        name = f"aligned_k{k}"
        path = LAYOUT_ROOT / f"{name}.zarr"
        arr = zarr.open_array(
            str(path), mode="w", shape=(n,), chunks=(chunk_cells,),
            dtype=values.dtype, compressor=compressor, fill_value=np.nan,
        )
        arr[:] = values
        layouts[name] = {"path": path, "chunk_cells": chunk_cells, "kind": "aligned", "k": k}
        print(f"  wrote {name}: chunk_cells={chunk_cells:,} "
              f"n_chunks={-(-n // chunk_cells):,}")

    naive_chunk_cells = 7 ** NAIVE_MATCH_K
    n_chunks = -(-n // naive_chunk_cells)
    naive_order = np.argsort(np.arange(n) % n_chunks, kind="stable")
    naive_values = values[naive_order]
    path = LAYOUT_ROOT / "naive.zarr"
    arr = zarr.open_array(
        str(path), mode="w", shape=(n,), chunks=(naive_chunk_cells,),
        dtype=values.dtype, compressor=compressor, fill_value=np.nan,
    )
    arr[:] = naive_values
    layouts["naive"] = {
        "path": path, "chunk_cells": naive_chunk_cells, "kind": "naive",
        "n_chunks": n_chunks,
    }
    print(f"  wrote naive: chunk_cells={naive_chunk_cells:,} n_chunks={n_chunks:,} "
          f"(round-robin i%{n_chunks} reorder)")
    return layouts


def naive_position(original_pos: np.ndarray, n_chunks: int, n: int) -> np.ndarray:
    """Map original sorted-array positions to their position in the naive
    round-robin (i % n_chunks) reordering. Derived analytically (stable sort
    by key=i%n_chunks): bucket sizes are ceil(n/n_chunks) for the first
    n % n_chunks buckets, floor(n/n_chunks) for the rest; the modulo split
    assigns the remainder to the first buckets (standard `i % n_chunks`
    distribution for i=0..n-1).
    """
    base = n // n_chunks
    remainder = n % n_chunks
    bucket_sizes = np.full(n_chunks, base, dtype=np.int64)
    bucket_sizes[:remainder] += 1
    cum = np.concatenate([[0], np.cumsum(bucket_sizes)])[:-1]
    bucket = original_pos % n_chunks
    slot = original_pos // n_chunks
    return cum[bucket] + slot


# ---------------------------------------------------------------------------
# query execution
# ---------------------------------------------------------------------------

def map_positions(original_positions: np.ndarray, layout: dict, n: int) -> np.ndarray:
    if layout["kind"] == "aligned":
        return original_positions
    return naive_position(original_positions, layout["n_chunks"], n)


def run_query(layout: dict, positions: np.ndarray, reps: int) -> dict:
    store = CountingStore(layout["path"])
    z = zarr.open_array(store=store, mode="r")

    store.reset()
    _ = z.vindex[positions]
    n_chunks_touched = len(store.touched_keys)

    times = []
    for _ in range(reps):
        store.reset()
        t0 = time.perf_counter()
        _ = z.vindex[positions]
        times.append(time.perf_counter() - t0)

    return {
        "n_positions": int(positions.size),
        "n_chunks_touched": n_chunks_touched,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"loading source archive {SOURCE_ARCHIVE} ...")
    ds = z7_zarr.open_dataset(SOURCE_ARCHIVE, decode=False)
    cell_ids = ds["cell_ids"].values.astype(np.uint64)
    values = np.asarray(ds["elevation_mean"].values, dtype=np.float32)
    n = cell_ids.size
    print(f"  N={n:,}")

    base_cells = np.unique(cell_ids >> np.uint64(60))
    print(f"  unique base cells present: {base_cells}")

    print("computing neighbour table + positions (once) ...")
    t0 = time.perf_counter()
    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)
    print(f"  {time.perf_counter()-t0:.2f}s")

    fully_interior = (pos >= 0).all(axis=1)
    idxs = np.nonzero(fully_interior)[0]
    spread = np.abs(pos[idxs] - idxs[:, None]).max(axis=1)

    interior_local = idxs[np.argmin(spread)]
    worst_local = idxs[np.argmax(spread)]
    print(f"  interior test cell: local idx={interior_local}, id={cell_ids[interior_local]}, "
          f"max-neighbour-spread={spread.min()}")
    print(f"  worst-case test cell: local idx={worst_local}, id={cell_ids[worst_local]}, "
          f"max-neighbour-spread={spread.max()}")

    print("\nbuilding 5 layouts ...")
    layouts = build_layouts(cell_ids, values)

    results = []

    # ---- 1. single-cell lookup (interior test cell) ----
    for name, layout in layouts.items():
        orig_pos = np.array([interior_local])
        mapped = map_positions(orig_pos, layout, n)
        r = run_query(layout, mapped, REPS)
        r.update(layout=name, query="single_cell")
        results.append(r)

    # ---- 2. parent-zone retrieval at each k, for every layout ----
    for qk in K_VALUES:
        parent = int(z7.get_parent_at(cell_ids[interior_local], LEVEL - qk))
        parent_mono = z7_zarr.z7_to_monotonic_int_batch(
            np.array([np.uint64(parent)], dtype=np.uint64), LEVEL - qk
        )[0]
        mono_start = int(parent_mono) * (7 ** qk)
        mono_end = mono_start + 7 ** qk - 1
        start_z7, end_z7 = z7_zarr.monotonic_int_to_z7_batch(
            np.array([mono_start, mono_end], dtype=np.uint64), LEVEL
        )
        lo = int(np.searchsorted(cell_ids, start_z7, side="left"))
        hi = int(np.searchsorted(cell_ids, end_z7, side="right"))
        orig_pos = np.arange(lo, hi)
        for name, layout in layouts.items():
            mapped = map_positions(orig_pos, layout, n)
            r = run_query(layout, mapped, REPS)
            r.update(layout=name, query=f"parent_zone_k{qk}", n_descendants_present=hi - lo)
            results.append(r)

    # ---- 3. cell + 6 neighbours, interior vs worst-case ----
    for label, local_idx in (("interior", interior_local), ("worst_case", worst_local)):
        self_and_nbrs = np.concatenate([[local_idx], pos[local_idx][pos[local_idx] >= 0]])
        for name, layout in layouts.items():
            mapped = map_positions(self_and_nbrs, layout, n)
            r = run_query(layout, mapped, REPS)
            r.update(layout=name, query=f"ring_{label}")
            results.append(r)

    df = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "bench_storage.parquet", index=False)
    df.to_csv(OUT / "bench_storage.csv", index=False)

    print("\n=== chunk touches (median wall, ms) ===")
    show = df.copy()
    show["median_ms"] = show["median_s"] * 1000
    print(show[["query", "layout", "n_positions", "n_chunks_touched", "median_ms"]]
          .to_string(index=False))

    # ---- storage overhead ----
    print("\n=== on-disk size per layout ===")
    sizes = {}
    for name, layout in layouts.items():
        path = layout["path"]
        nbytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        sizes[name] = nbytes
        print(f"  {name:<12} chunk_cells={layout['chunk_cells']:>8,}  {nbytes/1e6:>8.2f} MB "
              f"(uncompressed {n*4/1e6:.2f} MB, ratio {n*4/nbytes:.2f}x)")

    ranges_path = DATA_WORKING / "eesti_z7_r12_ranges.zarr"
    ranges_bytes = sum(f.stat().st_size for f in (ranges_path / "cell_id_ranges").rglob("*")
                        if f.is_file())
    dense_cellids_bytes = sum(f.stat().st_size for f in (SOURCE_ARCHIVE / "cell_ids").rglob("*")
                               if f.is_file())
    print(f"\nranges cell_id_ranges on disk: {ranges_bytes/1e6:.3f} MB")
    print(f"dense cell_ids on disk:        {dense_cellids_bytes/1e6:.3f} MB")
    print(f"ratio: {ranges_bytes/dense_cellids_bytes:.4%}")

    env = {
        "sudo_purge_available": False,
        "cache_state": "warm (page cache not cleared between reps; see docstring)",
        "reps": REPS,
        "source_archive": str(SOURCE_ARCHIVE),
        "n_cells": int(n),
        "interior_cell_id": int(cell_ids[interior_local]),
        "worst_case_cell_id": int(cell_ids[worst_local]),
        "worst_case_neighbour_spread_positions": int(spread.max()),
        "layout_sizes_bytes": sizes,
        "ranges_coord_bytes": int(ranges_bytes),
        "dense_cellids_bytes": int(dense_cellids_bytes),
    }
    (OUT / "bench_storage_env.json").write_text(json.dumps(env, indent=2))

    _plot(df)
    print(f"\nwrote {OUT/'bench_storage.parquet'}, {OUT/'bench_storage_env.json'}, "
          f"{OUT/'fig_bench_storage.png'}")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout_order = ["aligned_k2", "aligned_k3", "aligned_k4", "aligned_k5", "naive"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    sub = df[df["query"].isin(["single_cell", "ring_interior", "ring_worst_case"])]
    width = 0.25
    queries = ["single_cell", "ring_interior", "ring_worst_case"]
    x = np.arange(len(layout_order))
    for i, q in enumerate(queries):
        g = sub[sub["query"] == q].set_index("layout").reindex(layout_order)
        ax.bar(x + (i - 1) * width, g["n_chunks_touched"], width, label=q)
    ax.set_xticks(x); ax.set_xticklabels(layout_order, rotation=20)
    ax.set_ylabel("Zarr chunks touched")
    ax.set_title("Chunks touched: single cell / neighbour ring")
    ax.legend(fontsize=8)

    ax = axes[1]
    sub = df[df["query"].str.startswith("parent_zone")]
    for name in layout_order:
        g = sub[sub.layout == name].copy()
        g["k"] = g["query"].str.extract(r"k(\d)").astype(int)
        g = g.sort_values("k")
        ax.plot(g.k, g.n_chunks_touched, marker="o", label=name)
    ax.set_xlabel("query granularity k (r-k parent)")
    ax.set_ylabel("Zarr chunks touched")
    ax.set_yscale("log")
    ax.set_title("Parent-zone retrieval: chunks touched vs. layout")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_bench_storage.png", dpi=200)


if __name__ == "__main__":
    main()
