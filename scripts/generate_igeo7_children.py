#!/usr/bin/env python
"""
Script to generate IGEO7 preset children cells at level 7 for given parent cells at level 5.

Uses dggrid4py with DGGRIDv8 class and custom IGEO7 configuration parameters.
Also applies authalic latitude conversion for proper spatial indexing.

Usage:
    pixi run python scripts/generate_igeo7_children.py
"""

import os
import sys
from dggrid4py import DGGRIDv8, dggs_types
from dggrid4py.auxlat import geoseries_to_authalic, geoseries_to_geodetic
import geopandas as gpd
import numpy as np

# Import z7py for Z7 string to uint64 conversion
import z7py as Z7


# IGEO7 custom configuration parameters
IGEO7_META = dict(
    dggs_vert0_lon=11.20,
    dggs_vert0_lat=58.28252559,
    dggs_vert0_azimuth=0.0,
    input_address_type="HIERNDX",
    input_hier_ndx_system="Z7",
    input_hier_ndx_form="INT64",
    output_address_type="HIERNDX",
    output_cell_label_type="OUTPUT_ADDRESS_TYPE",
    output_hier_ndx_system="Z7",
    output_hier_ndx_form="INT64",
)

# Resolution-dependent clipper scale factor
CLIPPER_SCALE_FACTOR_DEFAULT = 10_000_000


def clipper_scale_factor_for(level: int) -> int:
    """Get the appropriate clipper scale factor for a given resolution level."""
    if level <= 8:
        return 1_000_000
    if level <= 11:
        return 10_000_000
    return 100_000_000  # level 12+: high-res, fine clipper required


def main():
    # Parent cell IDs at refinement level 5 as Z7_STRING
    parent_cell_ids_z7string = ['0000003', '0723233']
    parent_resolution = 5
    
    # Convert parent cell IDs to uint64 for HIERNDX/INT64 input
    parent_cell_ids_uint64 = [Z7.z7string_to_index(cid) for cid in parent_cell_ids_z7string]
    print(f"Parent cell IDs (Z7_STRING): {parent_cell_ids_z7string}")
    print(f"Parent cell IDs (uint64): {parent_cell_ids_uint64}")
    
    # Generate parents themselves, children at 1 level lower (6), and children at 2 levels higher (7)
    resolutions_to_generate = [parent_resolution, parent_resolution + 1, parent_resolution + 2]
    # resolutions_to_generate = [5, 6, 7]
    
    print("=" * 60)
    print("IGEO7 Multi-Resolution Cell Generation")
    print("=" * 60)
    print(f"Generating cells at resolutions: {resolutions_to_generate}")
    
    # Initialize DGGRIDv8 instance
    # Use environment variable for dggrid path, or specify default
    dggrid_path = os.environ.get('DGGRID_PATH', '/usr/local/bin/dggrid')
    
    if not os.path.exists(dggrid_path):
        print(f"Error: DGGRID executable not found at {dggrid_path}")
        print("Please set DGGRID_PATH environment variable or install DGGRID")
        sys.exit(1)
    
    working_dir = '/tmp'
    os.makedirs(working_dir, exist_ok=True)
    
    dggrid_instance = DGGRIDv8(
        executable=dggrid_path,
        working_dir=working_dir,
        capture_logs=True,
        silent=False
    )
    
    print(f"\nUsing DGGRID v8 at: {dggrid_path}")
    print(f"Working directory: {working_dir}")
    
    # Generate cells at multiple resolutions for each parent
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    all_cells_gdf = None
    
    for resolution in resolutions_to_generate:
        resolution_name = "parent" if resolution == parent_resolution else f"level_{resolution}"
        
        for parent_id_z7string, parent_id_uint64 in zip(parent_cell_ids_z7string, parent_cell_ids_uint64):
            print(f"\nProcessing {resolution_name} at resolution {resolution} for parent: {parent_id_z7string} (uint64: {parent_id_uint64})")
            
            # Get clipper scale factor for the target resolution
            clipper_scale = clipper_scale_factor_for(resolution)
            print(f"  Using clipper scale factor: {clipper_scale}")
            
            if resolution == parent_resolution:
                # Generate the parent cell itself - use Z7_STRING input to avoid INT64 parsing issues
                cells_gdf = dggrid_instance.grid_cell_polygons_from_cellids(
                    cell_id_list=[parent_id_z7string],
                    dggs_type='IGEO7',
                    resolution=resolution,
                    clip_subset_type='INPUT_ADDRESS_TYPE',
                    input_address_type='Z7_STRING',
                    output_address_type='Z7_STRING',
                )
            else:
                # Generate children cells with custom orientation
                # For children, we need to use the parent as a coarse cell for clipping
                # Use Z7_STRING for the parent ID
                orient_params = {
                    k: v for k, v in IGEO7_META.items()
                    if k.startswith('dggs_vert') or k.startswith('dggs_orient')
                }
                cells_gdf = dggrid_instance.grid_cell_polygons_from_cellids(
                    cell_id_list=[parent_id_z7string],
                    dggs_type='IGEO7',
                    resolution=resolution,
                    clip_subset_type='COARSE_CELLS',
                    clip_cell_res=parent_resolution,
                    clip_cell_agg_method='AVERAGE',
                    clip_cell_agg_samples=5,
                    clipper_scale_factor=clipper_scale,
                    input_address_type='Z7_STRING',
                    output_address_type='Z7_STRING',
                    **orient_params
                )
            
            print(f"  Generated {len(cells_gdf)} cells")
            print(f"  Columns: {list(cells_gdf.columns)}")
            
            # Check the format of the name field
            if 'name' in cells_gdf.columns and len(cells_gdf) > 0:
                sample_name = cells_gdf['name'].iloc[0]
                print(f"  Sample 'name' field value: {sample_name}")
            
            # Apply authalic latitude conversion for accurate spatial indexing
            print(f"  Applying authalic latitude conversion...")
            cells_gdf_authalic = cells_gdf.copy()
            cells_gdf_authalic.geometry = geoseries_to_authalic(cells_gdf.geometry)
            cells_gdf_authalic.crs = None  # Authalic coordinates are on a sphere
            
            # Save results
            parent_filename = parent_id_z7string.replace('/', '_')
            
            # Save in authalic coordinates
            authalic_path = os.path.join(output_dir, f"igeo7_{resolution_name}_{parent_filename}_l{resolution}_authalic.gpkg")
            cells_gdf_authalic.to_file(authalic_path, driver='GPKG')
            print(f"  Saved authalic: {authalic_path}")
            
            # Save in geodetic (WGS84) coordinates
            cells_gdf_geodetic = cells_gdf.copy()
            cells_gdf_geodetic.geometry = geoseries_to_geodetic(cells_gdf.geometry)
            cells_gdf_geodetic.crs = 4326  # WGS84
            
            # Convert the 'name' field from hex to Z7_STRING format
            from dggrid4py import igeo7
            cells_gdf_geodetic['z7_string'] = cells_gdf_geodetic['name'].apply(igeo7.z7hex_to_z7string)
            cells_gdf_geodetic['resolution'] = resolution
            
            geodetic_path = os.path.join(output_dir, f"igeo7_{resolution_name}_{parent_filename}_l{resolution}_wgs84.gpkg")
            cells_gdf_geodetic.to_file(geodetic_path, driver='GPKG')
            print(f"  Saved geodetic (WGS84): {geodetic_path}")
            print(f"  'name' field is in hex format, 'z7_string' field is in Z7_STRING format")
            
            # Print sample of Z7_STRING values
            if 'z7_string' in cells_gdf_geodetic.columns and len(cells_gdf_geodetic) > 0:
                print(f"  Sample z7_string values: {cells_gdf_geodetic['z7_string'].head(3).tolist()}")
            
            # Accumulate all cells
            if all_cells_gdf is None:
                all_cells_gdf = cells_gdf_geodetic
            else:
                all_cells_gdf = gpd.GeoDataFrame(
                    gpd.pd.concat([all_cells_gdf, cells_gdf_geodetic], ignore_index=True)
                )
        
        # Print sample
        print(f"\n  Sample cells (first 3):")
        print(cells_gdf.head(3))
    
    # Save combined results
    if all_cells_gdf is not None:
        combined_path = os.path.join(output_dir, f"igeo7_all_cells_multi_resolution_wgs84.gpkg")
        all_cells_gdf.to_file(combined_path, driver='GPKG')
        print(f"\n  Saved combined result: {combined_path}")
        print(f"  Total cells across all resolutions: {len(all_cells_gdf)}")
        # Print summary by resolution
        if 'resolution' in all_cells_gdf.columns:
            res_summary = all_cells_gdf['resolution'].value_counts().sort_index()
            print(f"  Cells per resolution: {dict(res_summary)}")
    
    print("\n" + "=" * 60)
    print("IGEO7 Multi-Resolution Cell Generation Complete")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
