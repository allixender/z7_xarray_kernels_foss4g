#!/usr/bin/env python
"""Phase 3a — settle the slope FDA math on Pori r10 (~3,101 cells, in-memory).

Diagnostic script. No Numba, no Dask, no kernels — pure scalar Python +
numpy. Resolves PHASE3.md open questions before we commit a slope kernel:

  Q2 (axis labelling across resolution parity)
        — what are the actual bearings of z7py directions d=1..6 at level
          10 (CCW)? Are 1↔4, 2↔5, 3↔6 antipodal?
  Q3 (cls_m vs true geodesic centre-to-centre distance)
        — how does mean(d_i) compare to RESOLUTION_STATS[r]["cls_m"]?
          If the bias is >1% we use per-direction geodesic distances in
          the kernel; otherwise cls_m is fine.

Also checks that the paper's FDA formula reproduces a closed-form slope
on a synthetic tilted plane.

Run:
    pixi run python scripts/phase3a_slope_math.py
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

import z7_xarray_paper.z7_zarr as z7_zarr
from z7py import z7
from z7py.z7 import RESOLUTION_STATS

INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------


def get_neighbours_batch(cell_ids: np.ndarray) -> np.ndarray:
    """Per cell, return its 6 neighbours in z7py canonical d=1..6 order.

    Returns shape (N, 6) uint64 with INVALID where a neighbour is missing
    (pentagons only, none in Pori AOI).
    """
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i, cid in enumerate(cell_ids):
        out[i, :] = z7.get_neighbours(np.uint64(cid))
    return out


def lonlat_to_local_xy(lon, lat, ref_lon, ref_lat):
    """Quick equirectangular-ish projection to local (east_m, north_m).

    Used only for synthesising a tilted-plane elevation field — small AOI,
    near-tangent plane, accurate enough that closed-form slope is a fair
    target for the FDA.
    """
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
    return (lon - ref_lon) * m_per_deg_lon, (lat - ref_lat) * m_per_deg_lat


def circular_mean_std(deg: np.ndarray) -> tuple[float, float]:
    """Circular mean and std (in degrees) of a 1-D angular array."""
    rad = np.radians(deg)
    mc, ms = np.cos(rad).mean(), np.sin(rad).mean()
    R = np.hypot(mc, ms)
    mean = np.degrees(np.arctan2(ms, mc)) % 360.0
    std = np.degrees(np.sqrt(-2.0 * np.log(R))) if R > 0 else float("nan")
    return mean, std


# ---------------------------------------------------------------------------


def main() -> None:
    # ---------------- 1. open + centroids
    ds = z7_zarr.open_dataset("data/working/pori_z7_r10.zarr", decode=True)
    level = int(ds.attrs["dggs"]["refinement_level"])
    cell_ids = ds.cell_ids.values.astype(np.uint64)
    n = cell_ids.size
    centers = ds.dggs.cell_centers()
    lon = centers.longitude.values
    lat = centers.latitude.values
    cls_m = float(RESOLUTION_STATS[level]["cls_m"])

    print(f"Pori archive : level={level} (parity={'even/CCW' if level % 2 == 0 else 'odd/CW'})")
    print(f"               N={n} cells")
    print(f"               lon=[{lon.min():.4f}, {lon.max():.4f}]")
    print(f"               lat=[{lat.min():.4f}, {lat.max():.4f}]")
    print(f"               cls_m = {cls_m:.2f}")

    # ---------------- 2. neighbours
    nbrs = get_neighbours_batch(cell_ids)
    valid_mask = nbrs != INVALID
    n_invalid_total = int((~valid_mask).sum())
    print(f"\nneighbours   : computed for {n} cells × 6 directions")
    print(f"               {n_invalid_total} INVALID slots (pentagons; expected 0 on Pori)")

    # ---------------- 3. neighbour centroids (one DGGRID call for all uniques)
    grid = ds.xindexes["cell_ids"].grid_info
    flat = nbrs[valid_mask]
    unique_nbrs = np.unique(flat).astype(np.uint64)
    print(f"               unique neighbour cells to query: {unique_nbrs.size}")
    nb_lon, nb_lat = grid.cell_ids2geographic(unique_nbrs)
    nb_lookup = {int(z): (float(lo), float(la)) for z, lo, la in zip(unique_nbrs, nb_lon, nb_lat)}

    # ---------------- 4. distance + bearing per direction
    geod = Geod(ellps="WGS84")
    per_dir = {d: {"dist": [], "bearing": []} for d in range(1, 7)}

    for i in range(n):
        clo, cla = lon[i], lat[i]
        for d in range(1, 7):
            nz = int(nbrs[i, d - 1])
            if nz == int(INVALID):
                continue
            nlo, nla = nb_lookup[nz]
            az_fwd, _, dist = geod.inv(clo, cla, nlo, nla)
            per_dir[d]["dist"].append(dist)
            per_dir[d]["bearing"].append(az_fwd)

    # GBT axial labelling (per Phase 3a finding):
    #   d=0     : centre (this cell — not a neighbour)
    #   d=1, 6  : +k / -k axis
    #   d=2, 5  : +j / -j axis
    #   d=3, 4  : -i / +i axis
    # The axis pairing is intrinsic to z7py's labelling and parity-invariant;
    # bearings rotate by ~19.1° between odd and even resolutions but the
    # antipodal structure does not.
    AXIS_OF_DIR = {1: ("k", +1), 6: ("k", -1),
                   2: ("j", +1), 5: ("j", -1),
                   4: ("i", +1), 3: ("i", -1)}

    # ---------------- 5. per-direction summary (with axial labels)
    print(f"\n{'='*82}")
    print(f"Per-direction stats — axial labels (cls_m baseline = {cls_m:.2f} m)")
    print(f"{'='*82}")
    print(f"{'d':>2}  {'axis':>5}  {'count':>6}  {'mean_d':>9}  {'std_d':>7}  "
          f"{'d/cls':>7}  {'min_d':>8}  {'max_d':>8}  "
          f"{'bearing_mean':>13}  {'bearing_std':>11}")
    for d in range(1, 7):
        ds_arr = np.asarray(per_dir[d]["dist"])
        bs_arr = np.asarray(per_dir[d]["bearing"])
        bm, bs = circular_mean_std(bs_arr)
        ax_name, sign = AXIS_OF_DIR[d]
        print(f"{d:>2}  {ax_name+('+' if sign>0 else '-'):>5}  "
              f"{ds_arr.size:>6}  {ds_arr.mean():>9.2f}  "
              f"{ds_arr.std():>7.2f}  {ds_arr.mean()/cls_m:>7.4f}  "
              f"{ds_arr.min():>8.2f}  {ds_arr.max():>8.2f}  "
              f"{bm:>13.2f}°  {bs:>10.2f}°")

    # Q2 antipodality check (per axis)
    print(f"\naxis antipodality check (positive vs negative end, expected 180°):")
    for ax in ("k", "j", "i"):
        d_pos = next(d for d, (a, s) in AXIS_OF_DIR.items() if a == ax and s > 0)
        d_neg = next(d for d, (a, s) in AXIS_OF_DIR.items() if a == ax and s < 0)
        b_p, _ = circular_mean_std(np.asarray(per_dir[d_pos]["bearing"]))
        b_n, _ = circular_mean_std(np.asarray(per_dir[d_neg]["bearing"]))
        diff = (b_p - b_n + 540.0) % 360.0 - 180.0
        print(f"  {ax}-axis  (+{ax}=d{d_pos}, -{ax}=d{d_neg}):  "
              f"{b_p:6.2f}° vs {b_n:6.2f}°   |delta from 180°| = {abs(diff):.4f}°")

    # ---------------- 5b. per-axis (collapsed) distance stats
    # Per-axis distance = mean of the two ends of that axis.
    print(f"\n{'='*82}")
    print(f"Per-axis distances (collapsed over both ends; cls_m = {cls_m:.2f} m)")
    print(f"{'='*82}")
    print(f"{'axis':>5}  {'mean_d':>9}  {'std_d':>7}  {'d/cls':>7}  "
          f"{'min_d':>8}  {'max_d':>8}")
    axis_d = {}
    for ax in ("k", "j", "i"):
        ds_pos = np.asarray(per_dir[next(d for d, (a, s) in AXIS_OF_DIR.items() if a == ax and s > 0)]["dist"])
        ds_neg = np.asarray(per_dir[next(d for d, (a, s) in AXIS_OF_DIR.items() if a == ax and s < 0)]["dist"])
        ds_combined = np.concatenate([ds_pos, ds_neg])
        axis_d[ax] = ds_combined.mean()
        print(f"{ax:>5}  {ds_combined.mean():>9.2f}  {ds_combined.std():>7.2f}  "
              f"{ds_combined.mean()/cls_m:>7.4f}  "
              f"{ds_combined.min():>8.2f}  {ds_combined.max():>8.2f}")

    # Aggregate single-value candidates for "default d per resolution"
    arith_mean = (axis_d["k"] + axis_d["j"] + axis_d["i"]) / 3
    geom_mean  = (axis_d["k"] * axis_d["j"] * axis_d["i"]) ** (1.0 / 3.0)
    print(f"\nsingle-value candidates (vs per-axis ground truth):")
    print(f"  cls_m            : {cls_m:>8.2f} m  (definition: equal-area-circle diameter)")
    print(f"  arithmetic mean  : {arith_mean:>8.2f} m  (avg of k, j, i)")
    print(f"  geometric mean   : {geom_mean:>8.2f} m  (closer to short axis)")
    print(f"  ratio cls_m / d_k: {cls_m/axis_d['k']:>8.4f}  (k-axis is the shortest)")
    print(f"  ratio cls_m / d_j: {cls_m/axis_d['j']:>8.4f}")
    print(f"  ratio cls_m / d_i: {cls_m/axis_d['i']:>8.4f}")

    # ---------------- 6. synthetic tilted plane → closed-form vs FDA
    print(f"\n{'='*82}")
    print("Synthetic tilted plane — closed-form vs FDA")
    print(f"{'='*82}")
    target_sx = 0.0500  # m/m east
    target_sy = 0.0200  # m/m north
    closed = np.hypot(target_sx, target_sy)

    ref_lon = float(lon.mean())
    ref_lat = float(lat.mean())
    cx, cy = lonlat_to_local_xy(lon, lat, ref_lon, ref_lat)
    h_centre = target_sx * cx + target_sy * cy

    nb_x_arr = (np.asarray(nb_lon) - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    nb_y_arr = (np.asarray(nb_lat) - ref_lat) * 111320.0
    nb_h = target_sx * nb_x_arr + target_sy * nb_y_arr
    h_nb_lookup = {int(z): float(h) for z, h in zip(unique_nbrs, nb_h)}

    print(f"target slope vector  : (sx={target_sx}, sy={target_sy})")
    print(f"closed-form magnitude: {closed:.6f} m/m")

    # Pre-compute Δh and per-direction geodesic distances and bearings
    # (cell, dir 0..5) — easier vectorised arithmetic below
    dh_arr   = np.full((n, 6), np.nan)
    d_arr    = np.full((n, 6), np.nan)
    bear_arr = np.full((n, 6), np.nan)
    boundary = np.zeros(n, dtype=bool)
    for i in range(n):
        for k in range(6):
            nz = int(nbrs[i, k])
            if nz == int(INVALID) or nz not in h_nb_lookup:
                boundary[i] = True
                continue
            dh_arr[i, k] = h_nb_lookup[nz] - h_centre[i]
            nlo, nla = nb_lookup[nz]
            az_fwd, _, dist = geod.inv(lon[i], lat[i], nlo, nla)
            d_arr[i, k]    = dist
            bear_arr[i, k] = az_fwd
    interior = ~boundary
    interior_n = int(interior.sum())
    print(f"interior cells (have all 6 neighbours): {interior_n}/{n}")

    def report(label, slopes):
        s = slopes[~np.isnan(slopes)]
        rel = (s - closed) / closed * 100.0
        print(f"\nvariant: {label}")
        print(f"  cells contributing       : {s.size}")
        print(f"  slope mean ± std         : {s.mean():.6f} ± {s.std():.6f} m/m")
        print(f"  relative error mean ± std: {rel.mean():+.4f}% ± {rel.std():.4f}%")
        print(f"  relative error min/max   : {rel.min():+.4f}% / {rel.max():+.4f}%")

    # Variant 1 — paper formula, paper axes, single d=cls_m (BROKEN, expected)
    s = np.full(n, np.nan)
    for i in np.where(interior)[0]:
        dh = dh_arr[i]
        dh_dx = (dh[0] - dh[3]) / (2.0 * cls_m)            # paper (1,4)
        dh_dy = (dh[1] + dh[2] - dh[4] - dh[5]) / (2.0 * np.sqrt(3.0) * cls_m)
        s[i] = np.hypot(dh_dx, dh_dy)
    report("paper axes, single d=cls_m  (expected wrong)", s)

    # Variant 2 — z7py axes, single d=cls_m
    s = np.full(n, np.nan)
    for i in np.where(interior)[0]:
        dh = dh_arr[i]
        dh_dx = (dh[0] - dh[5]) / (2.0 * cls_m)            # z7py (1,6) → +x/-x
        dh_dy = (dh[3] + dh[4] - dh[1] - dh[2]) / (2.0 * np.sqrt(3.0) * cls_m)
        s[i] = np.hypot(dh_dx, dh_dy)
    report("z7py axes, single d=cls_m", s)

    # Variant 3 — z7py axes, per-axis geodesic d (d_x for x, d_y for y)
    s = np.full(n, np.nan)
    for i in np.where(interior)[0]:
        dh = dh_arr[i]
        dx = 0.5 * (d_arr[i, 0] + d_arr[i, 5])             # axis-1 (z7py 1,6)
        dy = 0.25 * (d_arr[i, 1] + d_arr[i, 2] + d_arr[i, 3] + d_arr[i, 4])
        dh_dx = (dh[0] - dh[5]) / (2.0 * dx)
        dh_dy = (dh[3] + dh[4] - dh[1] - dh[2]) / (2.0 * np.sqrt(3.0) * dy)
        s[i] = np.hypot(dh_dx, dh_dy)
    report("z7py axes, per-axis geodesic d", s)

    # Variant 4 — least-squares plane fit (gold-standard sanity check)
    # Solve Δh_k = a * (d_k * sin β_k) + b * (d_k * cos β_k) for (a, b);
    # |slope| = sqrt(a² + b²). Bearing-aware, axis-label-agnostic.
    s = np.full(n, np.nan)
    for i in np.where(interior)[0]:
        dh = dh_arr[i]
        d  = d_arr[i]
        b  = np.radians(bear_arr[i])
        east_offsets  = d * np.sin(b)   # bearing 0=N, so east=sin
        north_offsets = d * np.cos(b)
        A = np.column_stack([east_offsets, north_offsets])
        sx_fit, sy_fit = np.linalg.lstsq(A, dh, rcond=None)[0]
        s[i] = np.hypot(sx_fit, sy_fit)
    report("least-squares plane fit (gold)", s)


if __name__ == "__main__":
    main()
