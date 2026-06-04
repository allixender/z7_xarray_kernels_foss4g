#!/usr/bin/env python
"""Phase 3a — global level-4 anisotropy sweep, aggregated at level-3 parents.

Probes ISEA7H shape distortion in *neighbour-distance space* (vs the
compactness measure in Kmoch et al. 2022). For every level-4 hexagonal
cell globally (~24,000 cells), computes per-axis geodesic distances
(d_k, d_j, d_i) to its 6 neighbours, classifies which axis is squeezed,
and aggregates at the cell's level-3 parent (~3,420 hex parents).

Output files (parquet, in data/output/):
  phase3a_global_l4_anisotropy.parquet
      — one row per level-4 hex cell, with axis distances, bearings,
        squeeze ratio, squeezed-axis label, level-3 parent id.
  phase3a_global_l3_parent_summary.parquet
      — one row per level-3 hex parent, with means/stds/modal squeezed axis.

Stdout: a sequence of summary tables answering the questions:

  - Is the geometric mean of (d_k, d_j, d_i) approximately constant globally?
    (If yes: separate "magnitude" from "shape" in the kernel formulation.)
  - Are squeezed axes evenly distributed across {k, j, i}?
    (5-petal × 12 base cell symmetry should give roughly equal thirds.)
  - Where are the singularities — pentagons + 20 triple-points?
"""

from __future__ import annotations

import os
import shapely
import numpy as np
import pandas as pd
from pyproj import Geod

from dggrid4py import DGGRIDv8, geoseries_to_geodetic

import z7py.z7 as z7
from z7py.z7 import RESOLUTION_STATS

from z7_xarray_paper.config import (
    IGEO7_META,
    DATA_OUTPUT,
    clipper_scale_factor_for,
)

INVALID = np.uint64(0xFFFFFFFFFFFFFFFF)
LEVEL = 5
PARENT_LEVEL = 4


# ---------------------------------------------------------------------------
# Step 1: enumerate all level-4 cells globally
# ---------------------------------------------------------------------------


def enumerate_global_cells(level: int) -> pd.DataFrame:
    """All cells at `level` globally via WHOLE_EARTH mode.

    Applies authalic-to-geodetic conversion to centroids.
    """
    if "DGGRID_PATH" not in os.environ:
        raise SystemExit("DGGRID_PATH not set")
    cache_dir = os.path.expanduser("~/.cache/_z7_global_sweep")
    os.makedirs(cache_dir, exist_ok=True)
    dggrid = DGGRIDv8(os.environ["DGGRID_PATH"], working_dir=cache_dir, silent=True)

    expected = RESOLUTION_STATS[level]["num_cells"]
    print(f"[enum] requesting level={level} via WHOLE_EARTH (expected {expected:,} cells) ...")

    df_all = dggrid.grid_cell_centroids_for_extent(
        "IGEO7", level,
        clip_geom=None,    # triggers WHOLE_EARTH mode
        **IGEO7_META,
    )

    # Apply authalic to geodetic conversion
    df_all.geometry = geoseries_to_geodetic(df_all.geometry)

    df_all["cell_id"] = df_all["name"].apply(int, base=16).astype(np.uint64)
    df_all["lon"] = df_all.geometry.x.astype(np.float64)
    df_all["lat"] = df_all.geometry.y.astype(np.float64)
    df_all = df_all[["cell_id", "lon", "lat"]].drop_duplicates(subset="cell_id").reset_index(drop=True)

    print(f"[enum] got {len(df_all):,} cells")
    if len(df_all) != expected:
        print(f"[enum] WARNING: cell count mismatch (got {len(df_all):,}, expected {expected:,})")
    return df_all


# ---------------------------------------------------------------------------
# Step 2: neighbours via z7py
# ---------------------------------------------------------------------------


def get_neighbours_batch(cell_ids: np.ndarray) -> np.ndarray:
    out = np.empty((cell_ids.size, 6), dtype=np.uint64)
    for i, cid in enumerate(cell_ids):
        out[i, :] = z7.get_neighbours(np.uint64(cid))
    return out


# ---------------------------------------------------------------------------
# Step 3: per-axis geodesic distances
# ---------------------------------------------------------------------------


def compute_per_direction_geodesics(
    df_cells: pd.DataFrame,
    nbrs: np.ndarray,
    centroids: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (dists[N, 6], bearings[N, 6]). NaN where neighbour missing."""
    geod = Geod(ellps="WGS84")
    n = len(df_cells)
    dists = np.full((n, 6), np.nan)
    bearings = np.full((n, 6), np.nan)
    for i in range(n):
        clo = float(df_cells["lon"].iat[i])
        cla = float(df_cells["lat"].iat[i])
        for d in range(6):
            nz = int(nbrs[i, d])
            if nz == int(INVALID) or nz not in centroids:
                continue
            nlo, nla = centroids[nz]
            az_fwd, _, dist = geod.inv(clo, cla, nlo, nla)
            dists[i, d] = dist
            bearings[i, d] = az_fwd
    return dists, bearings


# ---------------------------------------------------------------------------


def main() -> None:
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    # ------ enumerate
    df = enumerate_global_cells(LEVEL)

    # ------ centroid lookup
    centroids = {
        int(r.cell_id): (float(r.lon), float(r.lat)) for r in df.itertuples()
    }

    # ------ neighbours
    cell_ids = df["cell_id"].values.astype(np.uint64)
    print(f"[nbrs] computing 6 neighbours per cell ({len(cell_ids):,}) ...")
    nbrs = get_neighbours_batch(cell_ids)

    # Pentagons have at least one INVALID neighbour
    is_pentagon = (nbrs == INVALID).any(axis=1)
    pent_count = int(is_pentagon.sum())
    print(f"[nbrs] pentagons: {pent_count} (expected 12)")

    # ------ filter to hex
    hex_mask = ~is_pentagon
    df_hex = df[hex_mask].reset_index(drop=True)
    nbrs_hex = nbrs[hex_mask]
    print(f"[nbrs] hex cells to analyse: {len(df_hex):,}")

    # ------ geodesics
    print(f"[geod] computing geodesics for {len(df_hex):,} cells × 6 dirs ...")
    dists, bearings = compute_per_direction_geodesics(df_hex, nbrs_hex, centroids)

    n_missing = int(np.isnan(dists).sum())
    if n_missing:
        print(f"[geod] WARNING: {n_missing} (cell,dir) pairs had missing neighbour centroid")

    # ------ axis distances (z7py axial labelling)
    #   k axis: directions 1, 6 → indices 0, 5
    #   j axis: directions 2, 5 → indices 1, 4
    #   i axis: directions 3, 4 → indices 2, 3
    d_k = np.nanmean(dists[:, [0, 5]], axis=1)
    d_j = np.nanmean(dists[:, [1, 4]], axis=1)
    d_i = np.nanmean(dists[:, [2, 3]], axis=1)

    df_hex["d_k"] = d_k
    df_hex["d_j"] = d_j
    df_hex["d_i"] = d_i
    df_hex["d_geom_mean"] = np.cbrt(d_k * d_j * d_i)
    df_hex["d_arith_mean"] = (d_k + d_j + d_i) / 3.0
    df_hex["d_min"] = np.minimum.reduce([d_k, d_j, d_i])
    df_hex["d_max"] = np.maximum.reduce([d_k, d_j, d_i])
    df_hex["squeeze_ratio"] = df_hex["d_min"] / df_hex["d_max"]
    axis_min_idx = np.argmin(np.column_stack([d_k, d_j, d_i]), axis=1)
    df_hex["squeezed_axis"] = pd.Categorical(
        np.array(["k", "j", "i"])[axis_min_idx], categories=["k", "j", "i"]
    )

    # ------ level-3 parent
    parents3 = np.empty(len(df_hex), dtype=np.uint64)
    cell_ids_hex = df_hex["cell_id"].values.astype(np.uint64)
    for i, c in enumerate(cell_ids_hex):
        parents3[i] = z7.get_parent_at(np.uint64(c), PARENT_LEVEL)
    df_hex["parent3"] = parents3
    df_hex["base_cell"] = (cell_ids_hex >> np.uint64(60)).astype(np.uint8)

    # =======================================================================
    # SUMMARIES
    # =======================================================================
    cls_m = float(RESOLUTION_STATS[LEVEL]["cls_m"])
    print(f"\n{'='*78}")
    print(f"GLOBAL LEVEL-{LEVEL} ANISOTROPY  (cls = {cls_m:.0f} m)")
    print(f"{'='*78}")
    for label, arr in [("d_k", d_k), ("d_j", d_j), ("d_i", d_i),
                       ("d_geom_mean", df_hex["d_geom_mean"].values),
                       ("d_arith_mean", df_hex["d_arith_mean"].values)]:
        a = np.asarray(arr)
        print(f"  {label:>14s}: mean={a.mean():>9.0f}  std={a.std():>7.0f}  "
              f"range=[{a.min():>9.0f}, {a.max():>9.0f}]  "
              f"std/mean={a.std()/a.mean()*100:>5.2f}%")
    print()
    print(f"  cls_m / d_geom_mean(global): {cls_m / df_hex['d_geom_mean'].mean():.4f}")
    sr = df_hex["squeeze_ratio"].values
    print(f"  squeeze_ratio (d_min/d_max): mean={sr.mean():.4f}, "
          f"min={sr.min():.4f}, max={sr.max():.4f}")

    print(f"\n{'='*78}")
    print(f"SQUEEZED-AXIS DISTRIBUTION (global)")
    print(f"{'='*78}")
    counts = df_hex["squeezed_axis"].value_counts().sort_index()
    for ax, c in counts.items():
        print(f"  {ax}: {c:>6,d}  ({100 * c / len(df_hex):>5.1f}%)")

    print(f"\n{'='*78}")
    print(f"PER-BASE-CELL BREAKDOWN  (12 base cells, IDs 0..11)")
    print(f"{'='*78}")
    print(f"  {'bc':>3}  {'n':>5}  {'mean_sr':>7}  {'min_sr':>7}  "
          f"{'%k':>5}  {'%j':>5}  {'%i':>5}  {'mean_geom':>9}")
    for bc in range(12):
        sub = df_hex[df_hex["base_cell"] == bc]
        if len(sub) == 0:
            continue
        n_bc = len(sub)
        sax = sub["squeezed_axis"].value_counts(normalize=True) * 100
        print(f"  {bc:>3}  {n_bc:>5}  "
              f"{sub['squeeze_ratio'].mean():>7.4f}  "
              f"{sub['squeeze_ratio'].min():>7.4f}  "
              f"{sax.get('k', 0):>5.1f}  {sax.get('j', 0):>5.1f}  {sax.get('i', 0):>5.1f}  "
              f"{sub['d_geom_mean'].mean():>9.0f}")

    # ------ per-parent-3 aggregation
    grouped = df_hex.groupby("parent3", observed=True).agg(
        n=("cell_id", "count"),
        mean_d_k=("d_k", "mean"),
        std_d_k=("d_k", "std"),
        mean_d_j=("d_j", "mean"),
        std_d_j=("d_j", "std"),
        mean_d_i=("d_i", "mean"),
        std_d_i=("d_i", "std"),
        mean_d_geom=("d_geom_mean", "mean"),
        mean_squeeze=("squeeze_ratio", "mean"),
        min_squeeze=("squeeze_ratio", "min"),
    ).reset_index()
    # modal squeezed axis per parent (most common among children)
    modal = (
        df_hex.groupby("parent3", observed=True)["squeezed_axis"]
              .agg(lambda s: s.value_counts().idxmax())
              .reset_index(name="modal_squeezed_axis")
    )
    grouped = grouped.merge(modal, on="parent3", how="left")

    print(f"\n{'='*78}")
    print(f"LEVEL-3 PARENT SUMMARY  (n = {len(grouped):,} parent zones)")
    print(f"{'='*78}")
    sr_p = grouped["mean_squeeze"]
    print(f"  per-parent mean_squeeze: mean={sr_p.mean():.4f}, "
          f"std={sr_p.std():.4f}, min={sr_p.min():.4f}, max={sr_p.max():.4f}")
    print(f"  parents with mean_squeeze < 0.95 (strongly squeezed): "
          f"{(sr_p < 0.95).sum():>5d} ({100*(sr_p<0.95).mean():.1f}%)")
    print(f"  parents with mean_squeeze < 0.90 (very squeezed)    : "
          f"{(sr_p < 0.90).sum():>5d} ({100*(sr_p<0.90).mean():.1f}%)")
    print(f"  parents with mean_squeeze > 0.99 (near-isotropic)   : "
          f"{(sr_p > 0.99).sum():>5d} ({100*(sr_p>0.99).mean():.1f}%)")
    print(f"  modal squeezed axis distribution at parent level:")
    for ax, c in grouped["modal_squeezed_axis"].value_counts().sort_index().items():
        print(f"    {ax}: {c:>5d}  ({100*c/len(grouped):>5.1f}%)")

    # ------ outputs
    cell_path = DATA_OUTPUT / f"phase3a_global_l{LEVEL}_anisotropy.parquet"
    parent_path = DATA_OUTPUT / f"phase3a_global_l{PARENT_LEVEL}_parent_summary.parquet"
    df_hex_out = df_hex.copy()
    df_hex_out["squeezed_axis"] = df_hex_out["squeezed_axis"].astype(str)
    grouped_out = grouped.copy()
    grouped_out["modal_squeezed_axis"] = grouped_out["modal_squeezed_axis"].astype(str)
    df_hex_out.to_parquet(cell_path, index=False)
    grouped_out.to_parquet(parent_path, index=False)
    print(f"\n{'='*78}")
    print(f"OUTPUTS")
    print(f"{'='*78}")
    print(f"  per-cell    : {cell_path}  ({cell_path.stat().st_size/1024:.1f} KB)")
    print(f"  per-parent3 : {parent_path}  ({parent_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
