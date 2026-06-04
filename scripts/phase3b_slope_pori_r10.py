#!/usr/bin/env python
"""Phase 3b — slope FDA on Porijogi r10 with anisotropic per-axis distances.

In-memory single-shot. Reads pori_z7_r10.zarr, builds the 6×N neighbour
table via z7py, looks up per-axis distance scaling weights from the global
level-4 distortion table (dist_lookup_level4.parquet), applies the FDA on
z7py canonical axes (k/j/i), and validates against a synthetic tilted plane
matching the phase3a setup so results are directly comparable to its
Variant 3 (per-cell geodesic d).

z7py axis labelling (settled in phase3a_slope_math.py):
    d=1 (idx 0) : k+    d=6 (idx 5) : k-
    d=2 (idx 1) : j+    d=5 (idx 4) : j-
    d=4 (idx 3) : i+    d=3 (idx 2) : i-

FDA mapping (k axis treated as the local "x"; perp combines j and i):
    dh/dx = (dh[0] - dh[5]) / (2 d_k)
    dh/dy = (dh[3] + dh[4] - dh[1] - dh[2]) / (2 sqrt(3) d_y),  d_y = (d_j + d_i)/2

Run:
    pixi run python scripts/phase3b_slope_pori_r10.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import z7_xarray_paper.z7_zarr as z7_zarr
from z7_xarray_paper.distance_measures import HexGridDistortionModel
from z7py import z7
from z7py.z7 import RESOLUTION_STATS

INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)
PARENT_LEVEL = 4
DIST_LOOKUP_PATH = "data/working/dist_lookup_level4.parquet"
ARCHIVE_PATH = "data/working/pori_z7_r10.zarr"


# ---------------------------------------------------------------------------
# neighbours and per-cell distance lookup
# ---------------------------------------------------------------------------

def get_neighbours_batch(cell_ids: np.ndarray) -> np.ndarray:
    """Return (N, 6) uint64 of canonical d=1..6 neighbours per cell."""
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i, cid in enumerate(cell_ids):
        out[i, :] = z7.get_neighbours(np.uint64(cid))
    return out


def neighbour_positions(cell_ids: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Map (N, 6) uint64 neighbour IDs to positions in the sorted cell_ids
    array. Out-of-AOI neighbours get position -1 (caller masks them)."""
    flat = neighbours.ravel()
    pos = np.searchsorted(cell_ids, flat)
    safe = np.minimum(pos, cell_ids.size - 1)
    hit = (pos < cell_ids.size) & (cell_ids[safe] == flat)
    pos = np.where(hit, pos, -1)
    return pos.reshape(neighbours.shape)


def per_cell_axis_distances(
    cell_ids: np.ndarray, level: int, model: HexGridDistortionModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each cell return (d_k, d_j, d_i) in metres at the cell's resolution
    by looking up its level-PARENT_LEVEL parent in the distortion model.
    Cached over unique parents — typically O(parents) << O(cells)."""
    cls_m = float(RESOLUTION_STATS[level]["cls_m"])
    parents = np.fromiter(
        (int(z7.get_parent_at(np.uint64(c), PARENT_LEVEL)) for c in cell_ids),
        dtype=np.uint64,
        count=cell_ids.size,
    )
    uniq, inv = np.unique(parents, return_inverse=True)
    d_k_u = np.empty(uniq.size)
    d_j_u = np.empty(uniq.size)
    d_i_u = np.empty(uniq.size)
    for k, p in enumerate(uniq):
        d = model.get_neighbor_distances(int(p), cls_m)
        d_k_u[k], d_j_u[k], d_i_u[k] = d["d_k"], d["d_j"], d["d_i"]
    return d_k_u[inv], d_j_u[inv], d_i_u[inv], uniq


# ---------------------------------------------------------------------------
# slope FDA
# ---------------------------------------------------------------------------

def slope_fda(
    h: np.ndarray, pos: np.ndarray, boundary: np.ndarray,
    d_k: np.ndarray, d_j: np.ndarray, d_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_magnitude_m_per_m, dh_per_axis_dict).

    h        : (N,) float, elevation per cell (in source row-order = monotonic Z7).
    pos      : (N, 6) int, position of each neighbour in cell_ids; -1 for outside-AOI.
    boundary : (N,) bool, True if any column of pos is -1.
    d_k/j/i  : (N,) float, per-cell axis distances in metres.
    """
    # safe indexing — boundary cells will be NaN'd out below
    pos_safe = np.where(pos < 0, 0, pos)
    h_n = h[pos_safe]                # (N, 6)
    dh = h_n - h[:, None]            # (N, 6)

    d_y = 0.5 * (d_j + d_i)
    dh_dx = (dh[:, 0] - dh[:, 5]) / (2.0 * d_k)
    dh_dy = (dh[:, 3] + dh[:, 4] - dh[:, 1] - dh[:, 2]) / (2.0 * np.sqrt(3.0) * d_y)
    slope = np.hypot(dh_dx, dh_dy)
    slope[boundary] = np.nan
    return slope, {"dh_dx": dh_dx, "dh_dy": dh_dy}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. open archive
    ds = z7_zarr.open_dataset(ARCHIVE_PATH, decode=True)
    level = int(ds.attrs["dggs"]["refinement_level"])
    cell_ids = ds.cell_ids.values.astype(np.uint64)
    n = cell_ids.size
    elevation = np.asarray(ds["elevation"].values, dtype=np.float64)
    cls_m = float(RESOLUTION_STATS[level]["cls_m"])

    print(f"archive   : {ARCHIVE_PATH}")
    print(f"            level={level}  N={n}  cls_m={cls_m:.2f}")

    # ---- 2. neighbours and positions
    nbrs = get_neighbours_batch(cell_ids)
    n_invalid = int((nbrs == INVALID).sum())
    assert n_invalid == 0, f"unexpected pentagons in Pori AOI: {n_invalid}"
    pos = neighbour_positions(cell_ids, nbrs)
    boundary = (pos < 0).any(axis=1)
    interior = ~boundary
    print(f"neighbours: pentagons=0  interior={interior.sum()}/{n}  "
          f"boundary={boundary.sum()}/{n}")

    # ---- 3. per-cell axis distances via level-4 parent lookup
    empirical = pd.read_parquet(DIST_LOOKUP_PATH).to_dict("index")
    model = HexGridDistortionModel(empirical)
    d_k, d_j, d_i, parents_uniq = per_cell_axis_distances(cell_ids, level, model)
    print(f"distance  : level-{PARENT_LEVEL} parents covering AOI = {parents_uniq.size}")
    print(f"            d_k = [{d_k.min():.2f}, {d_k.max():.2f}]  "
          f"(mean {d_k.mean():.2f})")
    print(f"            d_j = [{d_j.min():.2f}, {d_j.max():.2f}]  "
          f"(mean {d_j.mean():.2f})")
    print(f"            d_i = [{d_i.min():.2f}, {d_i.max():.2f}]  "
          f"(mean {d_i.mean():.2f})")
    print(f"            cls_m / mean(d_k,d_j,d_i) = "
          f"{cls_m / np.mean([d_k.mean(), d_j.mean(), d_i.mean()]):.4f}")

    # ---- 4. synthetic tilted plane validation
    centers = ds.dggs.cell_centers()
    lon = centers.longitude.values
    lat = centers.latitude.values
    target_sx = 0.0500   # m/m east
    target_sy = 0.0200   # m/m north
    closed = np.hypot(target_sx, target_sy)

    ref_lon, ref_lat = float(lon.mean()), float(lat.mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
    cx = (lon - ref_lon) * m_per_deg_lon
    cy = (lat - ref_lat) * m_per_deg_lat
    h_synth = target_sx * cx + target_sy * cy

    s_synth, _ = slope_fda(h_synth, pos, boundary, d_k, d_j, d_i)
    s_synth_int = s_synth[interior]
    rel_err = (s_synth_int - closed) / closed * 100.0

    print(f"\n{'='*78}")
    print("Synthetic tilted plane — closed-form vs FDA (per-region distances)")
    print(f"{'='*78}")
    print(f"target slope vector  : (sx={target_sx}, sy={target_sy})")
    print(f"closed-form magnitude: {closed:.6f} m/m")
    print(f"interior n           : {s_synth_int.size}")
    print(f"slope mean ± std     : {s_synth_int.mean():.6f} ± {s_synth_int.std():.6f} m/m")
    print(f"rel error mean ± std : {rel_err.mean():+.4f}% ± {rel_err.std():.4f}%")
    print(f"rel error min/max    : {rel_err.min():+.4f}% / {rel_err.max():+.4f}%")

    # ---- 5. real DEM slope
    s_real, _ = slope_fda(elevation, pos, boundary, d_k, d_j, d_i)
    valid = np.isfinite(s_real)
    s_real_deg = np.degrees(np.arctan(s_real))

    print(f"\n{'='*78}")
    print(f"Porijogi r10 DEM slope")
    print(f"{'='*78}")
    print(f"valid n              : {valid.sum()}/{n} (boundary masked)")
    print(f"elevation range      : [{elevation.min():.2f}, {elevation.max():.2f}] m  "
          f"(median {np.median(elevation):.2f})")
    print(f"slope (m/m)          : min/median/mean/max = "
          f"{s_real[valid].min():.5f} / {np.median(s_real[valid]):.5f} / "
          f"{s_real[valid].mean():.5f} / {s_real[valid].max():.5f}")
    print(f"slope (deg)          : min/median/mean/max = "
          f"{s_real_deg[valid].min():.3f}° / {np.median(s_real_deg[valid]):.3f}° / "
          f"{s_real_deg[valid].mean():.3f}° / {s_real_deg[valid].max():.3f}°")
    print(f"percentiles (deg)    : p50={np.percentile(s_real_deg[valid], 50):.3f}°  "
          f"p90={np.percentile(s_real_deg[valid], 90):.3f}°  "
          f"p99={np.percentile(s_real_deg[valid], 99):.3f}°")


if __name__ == "__main__":
    main()
