#!/usr/bin/env python
"""Throughput benchmark for the Z7 GBT neighbour kernels (paper Section 5.1).

Run inside the project environment:

    pixi run python scripts/bench_neighbours.py
    pixi run python scripts/bench_neighbours.py --quick     # smaller N, for a smoke test

Writes:
    data/output/bench_neighbours.parquet   tidy results, one row per (impl, res, N, rep)
    data/output/bench_neighbours.csv       same, for eyeballing
    data/output/fig_bench_neighbours.png   throughput vs N and vs resolution

Implementations compared
------------------------
pure_python        z7py.get_neighbours.py_func, the un-compiled reference.
njit_scalar_naive  the njit single-cell kernel called from a Python loop with its
                   default arguments left implicit. This is what the project used
                   before the batch kernel existed, and it is pathologically slow
                   because every call boxes eight numpy default arrays.
njit_scalar_hoist  the same njit kernel, but with the lookup tables passed
                   explicitly so that the boxing happens once. This is the fair
                   scalar baseline.
njit_batch_serial  the compiled batch kernel, one thread.
njit_batch_prange  the compiled batch kernel, all Numba threads.

All timings exclude JIT compilation: every implementation is called once on a
small array before the timed repeats.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numba as nb  # noqa: E402
from z7py import z7  # noqa: E402
from z7_xarray_paper.kernels.neighbours import (  # noqa: E402
    get_neighbours_batch,
    get_neighbours_batch_pyloop,
)

OUT = REPO / "data" / "output"

# the lookup tables that z7.get_neighbours takes as default arguments
_TABLES = (
    z7._BASE_CELL_NEIGHBOURS_RAW,
    z7._EXCLUDE_NEIGHBOURS,
    z7._ROTATIONS,
    z7._POLE_0_ROTATIONS,
    z7._GBT_CW_0,
    z7._GBT_CW_1,
    z7._GBT_CCW_0,
    z7._GBT_CCW_1,
)


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------

def sample_cells(resolution: int, n: int, seed: int = 0) -> np.ndarray:
    """Random valid Z7 indices at `resolution`, sorted ascending.

    Built directly on the bit layout (base cell in bits 60-63, then 20 digits of
    3 bits each, digit i at shift 57 - 3(i-1)), so that generating 10^7 test
    cells does not itself dominate the benchmark.
    """
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 12, size=n).astype(np.uint64) << np.uint64(60)
    for i in range(1, 21):
        shift = np.uint64(57 - 3 * (i - 1))
        if i <= resolution:
            digit = rng.integers(0, 7, size=n).astype(np.uint64)
        else:
            digit = np.full(n, 7, dtype=np.uint64)
        raw |= digit << shift
    return np.sort(raw)


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------

def _pure_python(cell_ids: np.ndarray) -> np.ndarray:
    fn = z7.get_neighbours.py_func
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i in range(cell_ids.size):
        out[i, :] = fn(cell_ids[i], *_TABLES)
    return out


def _njit_scalar_hoisted(cell_ids: np.ndarray) -> np.ndarray:
    fn = z7.get_neighbours
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i in range(cell_ids.size):
        out[i, :] = fn(cell_ids[i], *_TABLES)
    return out


IMPLEMENTATIONS = {
    "pure_python": (_pure_python, 3_000),
    "njit_scalar_naive": (get_neighbours_batch_pyloop, 3_000),
    "njit_scalar_hoist": (_njit_scalar_hoisted, 50_000),
    "njit_batch_serial": (lambda x: get_neighbours_batch(x, parallel=False), None),
    "njit_batch_prange": (lambda x: get_neighbours_batch(x, parallel=True), None),
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def time_one(fn, cell_ids: np.ndarray, reps: int) -> list[float]:
    fn(cell_ids[:64])  # warm-up, excludes JIT compilation from the timings
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(cell_ids)
        out.append(time.perf_counter() - t0)
    return out


def measure_memory(resolution: int, n: int) -> dict:
    """uint64 array footprint vs a Python-object representation of the same cells."""
    ids = sample_cells(resolution, n)
    array_bytes = ids.nbytes
    objects = [int(x) for x in ids]
    object_bytes = sys.getsizeof(objects) + sum(sys.getsizeof(o) for o in objects)
    return {
        "n": n,
        "resolution": resolution,
        "uint64_bytes": array_bytes,
        "python_object_bytes": object_bytes,
        "ratio": object_bytes / array_bytes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small N, for a smoke test")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    resolutions = [6, 9, 12, 15]
    if args.quick:
        sizes = [1_000, 10_000, 100_000]
        reps = 3
    else:
        # 1_000 is below the pure_python / njit_scalar_naive cap (3_000) so
        # those two variants actually get exercised at least once; without a
        # size <= cap present, `n > cap` is true for every tested size and
        # they never run at all.
        sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
        reps = args.reps

    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": nb.__version__,
        "numba_threads": nb.config.NUMBA_NUM_THREADS,
        "threading_layer": None,
    }

    rows = []
    for resolution in resolutions:
        for n in sizes:
            ids = sample_cells(resolution, n)
            for name, (fn, cap) in IMPLEMENTATIONS.items():
                if cap is not None and n > cap:
                    continue  # the slow reference implementations would take hours
                for rep, dt in enumerate(time_one(fn, ids, reps)):
                    rows.append({
                        "implementation": name,
                        "resolution": resolution,
                        "n": n,
                        "rep": rep,
                        "seconds": dt,
                        "cells_per_second": n / dt,
                        "ns_per_cell": 1e9 * dt / n,
                    })
                med = np.median([r["cells_per_second"] for r in rows
                                 if r["implementation"] == name
                                 and r["resolution"] == resolution and r["n"] == n])
                print(f"r={resolution:<3} N={n:>10,}  {name:<20} {med:>14,.0f} cells/s")
            del ids

    try:
        env["threading_layer"] = nb.threading_layer()
    except Exception:
        pass

    import pandas as pd
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "bench_neighbours.parquet", index=False)
    df.to_csv(OUT / "bench_neighbours.csv", index=False)

    mem = [measure_memory(12, 100_000)]
    (OUT / "bench_neighbours_env.json").write_text(
        json.dumps({"environment": env, "memory": mem}, indent=2)
    )
    print("\nenvironment:", json.dumps(env, indent=2))
    print("memory:", json.dumps(mem, indent=2))

    _plot(df)
    print(f"\nwrote {OUT/'bench_neighbours.parquet'} and {OUT/'fig_bench_neighbours.png'}")


def _plot(df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med = (df.groupby(["implementation", "resolution", "n"])["cells_per_second"]
             .median().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    sub = med[med.resolution == 12]
    for name, g in sub.groupby("implementation"):
        g = g.sort_values("n")
        ax.plot(g.n, g.cells_per_second, marker="o", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel("throughput (cells s$^{-1}$)")
    ax.set_title("Throughput vs. batch size (refinement level 12)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    biggest = med.n.max()
    sub = med[med.n == biggest]
    for name, g in sub.groupby("implementation"):
        g = g.sort_values("resolution")
        ax.plot(g.resolution, g.cells_per_second, marker="s", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("IGEO7 refinement level $r$")
    ax.set_ylabel("throughput (cells s$^{-1}$)")
    ax.set_title(f"Throughput vs. resolution ($N = {biggest:,}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_bench_neighbours.png", dpi=200)


if __name__ == "__main__":
    main()
