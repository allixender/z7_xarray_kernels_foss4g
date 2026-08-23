"""Task E — synthetic validation, closes PHASE3 Q2 (HANDOFF_LOCAL_CLAUDE.md).

Validates the FDA slope kernel (`z7_xarray_paper.kernels.slope`) against
closed-form analytic slope on two synthetic surfaces, over the Porijogi AOI,
at r in {10, 11, 12, 13} (both resolution parities: 10/12 even, 11/13 odd
relative to the alternating_cw_odd_ccw_even rotation pattern).

Cell-id hierarchy is *derived*, not independently regridded, so all four
resolutions share exactly the same footprint (no AOI-edge discrepancies
between levels): r12 comes from the existing `pori_z7_r12.zarr`; r10/r11 are
the unique ancestors of the r12 set; r13 is the full set of children of the
r12 set (mechanical aperture-7 expansion via the monotonic-int encoding, no
DGGRID call needed).

1. Tilted plane `h = sx*x + sy*y` (closed-form slope = hypot(sx,sy), constant
   everywhere) and paraboloid `h = c*(x^2+y^2)` (closed-form slope = 2c*r,
   varies per cell) in a local equirectangular metric frame centred on the
   AOI.
2. Assert slope magnitude is parity-invariant (same target, same bias, up to
   a stated tolerance) across the four levels.
3. Quantify the lookup-mode bias against the closed form at each level,
   separately from the 1.41% lookup-vs-geodesic agreement number quoted in
   `scripts/z7_slope_demo.ipynb` (a different comparison: FDA-lookup vs.
   FDA-geodesic, not FDA vs. closed-form analytic truth).

Run: pixi run python scripts/validate_slope_synthetic.py
"""

from __future__ import annotations

import json

import numba as nb
import numpy as np
import pandas as pd

from z7_xarray_paper import z7_zarr
from z7_xarray_paper.config import DATA_OUTPUT, DATA_WORKING, IGEO7_META
from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7_xarray_paper.kernels.neighbours import get_neighbours_batch, neighbour_positions
from z7_xarray_paper.kernels.slope import _distances_lookup, _slope_fda
from z7py import z7
from xdggs_dggrid4py.dependences.grids import IGEO7Info

OUT = DATA_OUTPUT
DIST_LOOKUP_PATH = DATA_WORKING / "dist_lookup_level4.parquet"
R12_ARCHIVE = DATA_WORKING / "pori_z7_r12.zarr"
LEVELS = (10, 11, 12, 13)

TARGET_SX = 0.0500   # m/m east, tilted-plane target
TARGET_SY = 0.0200   # m/m north
PARABOLOID_C = 2.0e-6  # 1/m, curvature coefficient


@nb.njit(cache=True, parallel=True)
def _parent_at_batch(cell_ids: np.ndarray, resolution: int) -> np.ndarray:
    out = np.empty(cell_ids.shape[0], dtype=np.uint64)
    for i in nb.prange(cell_ids.shape[0]):
        out[i] = z7.get_parent_at(cell_ids[i], resolution)
    return out


def grid_info_for(level: int) -> IGEO7Info:
    return IGEO7Info.from_dict({
        "level": level,
        "_dggrid_meta_config": IGEO7_META,
        "igeo7_dggs_vert0_lon": 11.20,
        "igeo7_wgs84_geodetic_conversion": True,
    })


def derive_hierarchy(r12_ids: np.ndarray) -> dict[int, np.ndarray]:
    r12_ids = np.sort(np.ascontiguousarray(r12_ids, dtype=np.uint64))
    out = {12: r12_ids}
    for lvl in (10, 11):
        parents = _parent_at_batch(r12_ids, lvl)
        out[lvl] = np.unique(parents)

    m12 = z7_zarr.z7_to_monotonic_int_batch(r12_ids, 12)
    offsets = np.arange(7, dtype=np.uint64)
    children_mono = (m12[:, None] * np.uint64(7) + offsets[None, :]).ravel()
    out[13] = z7_zarr.monotonic_int_to_z7_batch(children_mono, 13)
    return out


def _model() -> HexGridDistortionModel:
    empirical = pd.read_parquet(DIST_LOOKUP_PATH).to_dict("index")
    return HexGridDistortionModel(empirical)


def evaluate_level(level: int, cell_ids: np.ndarray, model: HexGridDistortionModel,
                    ref_lon: float, ref_lat: float) -> dict:
    n = cell_ids.size
    ginfo = grid_info_for(level)
    lon, lat = ginfo.cell_ids2geographic(cell_ids)

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
    x = (lon - ref_lon) * m_per_deg_lon
    y = (lat - ref_lat) * m_per_deg_lat

    nbrs = get_neighbours_batch(cell_ids)
    pos = neighbour_positions(cell_ids, nbrs)
    boundary = (pos < 0).any(axis=1)
    interior = ~boundary

    d_k, d_j, d_i = _distances_lookup(cell_ids, level, model)

    # ---- tilted plane ----
    h_plane = TARGET_SX * x + TARGET_SY * y
    s_plane = _slope_fda(h_plane, pos, boundary, d_k, d_j, d_i)
    closed_plane = np.hypot(TARGET_SX, TARGET_SY)
    rel_err_plane = (s_plane[interior] - closed_plane) / closed_plane * 100.0

    # ---- paraboloid ----
    h_para = PARABOLOID_C * (x ** 2 + y ** 2)
    s_para = _slope_fda(h_para, pos, boundary, d_k, d_j, d_i)
    r_local = np.hypot(x, y)
    closed_para = 2.0 * PARABOLOID_C * r_local
    # avoid div-by-zero at the exact AOI centre
    safe = closed_para > 1e-9
    rel_err_para = ((s_para[interior & safe] - closed_para[interior & safe])
                     / closed_para[interior & safe] * 100.0)

    return {
        "level": level,
        "n_cells": int(n),
        "n_interior": int(interior.sum()),
        "parity": "even" if level % 2 == 0 else "odd",
        "plane_slope_mean": float(np.mean(s_plane[interior])),
        "plane_slope_std": float(np.std(s_plane[interior])),
        "plane_closed_form": float(closed_plane),
        "plane_rel_err_mean_pct": float(np.mean(rel_err_plane)),
        "plane_rel_err_std_pct": float(np.std(rel_err_plane)),
        "plane_rel_err_min_pct": float(np.min(rel_err_plane)),
        "plane_rel_err_max_pct": float(np.max(rel_err_plane)),
        "para_rel_err_mean_pct": float(np.mean(rel_err_para)),
        "para_rel_err_std_pct": float(np.std(rel_err_para)),
        "para_n_compared": int(rel_err_para.size),
    }


def main() -> None:
    ds = z7_zarr.open_dataset(R12_ARCHIVE, decode=False)
    r12_ids = ds["cell_ids"].values.astype(np.uint64)
    print(f"r12 anchor set: N={r12_ids.size:,} (from {R12_ARCHIVE})")

    hierarchy = derive_hierarchy(r12_ids)
    for lvl in LEVELS:
        print(f"  derived r{lvl}: N={hierarchy[lvl].size:,}")

    g12 = grid_info_for(12)
    lon12, lat12 = g12.cell_ids2geographic(r12_ids)
    ref_lon, ref_lat = float(lon12.mean()), float(lat12.mean())
    print(f"AOI reference centre: lon={ref_lon:.5f}, lat={ref_lat:.5f}")

    model = _model()

    rows = []
    for lvl in LEVELS:
        print(f"\nevaluating r{lvl} ...")
        row = evaluate_level(lvl, hierarchy[lvl], model, ref_lon, ref_lat)
        print(json.dumps(row, indent=2))
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "validate_slope_synthetic.parquet", index=False)
    df.to_csv(OUT / "validate_slope_synthetic.csv", index=False)

    print("\n=== parity check (tilted plane, target slope invariant across levels) ===")
    even = df[df.parity == "even"]["plane_rel_err_mean_pct"]
    odd = df[df.parity == "odd"]["plane_rel_err_mean_pct"]
    parity_gap = abs(even.mean() - odd.mean())
    tol_pct_points = 0.5
    print(f"  even-level (10,12) mean bias: {even.mean():+.4f}%")
    print(f"  odd-level  (11,13) mean bias: {odd.mean():+.4f}%")
    print(f"  |gap|: {parity_gap:.4f} percentage points "
          f"(tolerance {tol_pct_points} pp) -> "
          f"{'PASS: parity-invariant' if parity_gap <= tol_pct_points else 'FAIL: parity-dependent bias found'}")

    print("\n=== AGENT.md B1 cross-check (Pori r10, lookup-mode bias) ===")
    r10 = df[df.level == 10].iloc[0]
    print(f"  measured: {r10.plane_rel_err_mean_pct:+.4f}% ± {r10.plane_rel_err_std_pct:.4f}% "
          f"(AGENT.md B1 claims ~-1.86% ± 0.16%)")
    print("  NOTE: this is FDA-lookup vs. closed-form analytic truth. The 1.41% figure in "
          "scripts/z7_slope_demo.ipynb is FDA-lookup vs. FDA-geodesic (a different pair) "
          "and is NOT reproduced or contradicted by this script.")

    (OUT / "validate_slope_synthetic_summary.json").write_text(json.dumps({
        "parity_gap_pct_points": parity_gap,
        "parity_tolerance_pct_points": tol_pct_points,
        "parity_pass": bool(parity_gap <= tol_pct_points),
        "r10_plane_rel_err_mean_pct": float(r10.plane_rel_err_mean_pct),
        "r10_plane_rel_err_std_pct": float(r10.plane_rel_err_std_pct),
        "agent_md_b1_claim_mean_pct": -1.86,
        "agent_md_b1_claim_std_pct": 0.16,
    }, indent=2))

    print(f"\nwrote {OUT/'validate_slope_synthetic.parquet'}, "
          f"{OUT/'validate_slope_synthetic_summary.json'}")


if __name__ == "__main__":
    main()
