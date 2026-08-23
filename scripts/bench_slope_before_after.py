"""Task A step 5 — confirm the slope path still works and is faster.

Times `z7_slope(ds["elevation"], model, distance_mode="lookup")` on
`data/working/pori_z7_r12_ranges.zarr` twice: once forcing the serial
njit neighbour-fill kernel ("before" the prange batch optimisation) and
once with the default auto-parallel kernel ("after"). The parallel/serial
choice in `get_neighbours_batch` is controlled by the module-level
`Z7_PARALLEL_THRESHOLD` env var read at import time (see
`kernels/neighbours.py`), so each variant is run in its own subprocess
with that env var set — this is a real re-exercise of the code path, not
a simulated number.

Run: pixi run python scripts/bench_slope_before_after.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = REPO / "data" / "working" / "pori_z7_r12_ranges.zarr"
DIST_LOOKUP_PATH = REPO / "data" / "working" / "dist_lookup_level4.parquet"
OUT = REPO / "data" / "output"

_WORKER = r"""
import sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, {src!r})
sys.path.insert(0, {repo!r})
from z7_xarray_paper import z7_zarr
from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7_xarray_paper.kernels.slope import slope

ds = z7_zarr.open_dataset({archive!r}, decode=True)
elevation = ds["elevation"]
elevation = elevation.copy(data=np.asarray(elevation.values, dtype=np.float64))

empirical = pd.read_parquet({dist_lookup!r}).to_dict("index")
model = HexGridDistortionModel(empirical)

# Warm-up on the full array to exclude JIT compilation from the timed reps.
# (A subset via .sel()/.isel() would be cheaper, but both degrade a
# Z7MonotonicIndex-backed selection to a plain PandasIndex -- a real bug,
# noted separately -- which slope() then rejects for lacking .grid_info.)
_ = slope(elevation, model, distance_mode="lookup")

reps = 5
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    result = slope(elevation, model, distance_mode="lookup")
    times.append(time.perf_counter() - t0)

n = int(elevation.sizes["cell_ids"])
n_nan = int(np.isnan(np.asarray(result.values)).sum())
print(json.dumps({{"n": n, "n_nan": n_nan, "times": times}}))
""".strip()


def run_variant(label: str, threshold: str) -> dict:
    src = str(REPO / "src")
    code = _WORKER.format(
        src=src, repo=str(REPO),
        archive=str(ARCHIVE_PATH), dist_lookup=str(DIST_LOOKUP_PATH),
    )
    env = dict(os.environ)
    env["Z7_PARALLEL_THRESHOLD"] = threshold
    env["PYTHONPATH"] = f"{src}:{REPO}"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{label} subprocess failed (rc={proc.returncode})")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["label"] = label
    payload["subprocess_wall_s"] = wall
    return payload


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(ARCHIVE_PATH)

    print(f"archive: {ARCHIVE_PATH}")
    # threshold way above N -> get_neighbours_batch always picks the serial kernel
    before = run_variant("before (serial njit batch, forced)", threshold="100000000")
    # threshold 0 -> always picks the prange kernel (this is the current default
    # behaviour too, since Pori r12's N=158,430 is already above the default
    # 4096 threshold, but we pin it explicitly for clarity)
    after = run_variant("after (prange njit batch, forced)", threshold="0")

    for res in (before, after):
        med = sorted(res["times"])[len(res["times"]) // 2]
        print(f"{res['label']:<38} N={res['n']:,}  median={med*1000:.2f} ms  "
              f"({res['n']/med:,.0f} cells/s)  n_nan={res['n_nan']}")

    speedup = (sorted(before["times"])[len(before["times"]) // 2]
               / sorted(after["times"])[len(after["times"]) // 2])
    print(f"\nspeedup (before/after): {speedup:.2f}x")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench_slope_before_after.json").write_text(
        json.dumps({"before": before, "after": after, "speedup": speedup}, indent=2)
    )
    print(f"\nwrote {OUT / 'bench_slope_before_after.json'}")


if __name__ == "__main__":
    main()
