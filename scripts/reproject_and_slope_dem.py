#!/usr/bin/env python
"""
Demo script for reprojecting the Pori DEM from EPSG:3301 to EPSG:4326,
calculating slope, saving results, and visualizing.

Usage:
    pixi run python scripts/reproject_and_slope_dem.py
"""

import os
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 - adds rio accessor to xarray

# Import xarray-spatial functions
from xrspatial import slope

# For display
import matplotlib.pyplot as plt


def main():
    # Input and output paths
    input_path = "data/input/merit_dem_pori_cog.tif"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    reprojected_path = os.path.join(output_dir, "merit_dem_pori_4326.tif")
    slope_path = os.path.join(output_dir, "merit_dem_pori_slope.tif")
    
    print("=" * 60)
    print("DEM Reprojection and Slope Calculation Demo")
    print("=" * 60)
    
    # Step 1: Read the input DEM (EPSG:3301)
    print("\n[1/5] Reading input DEM from EPSG:3301...")
    dem = xr.open_dataarray(input_path, engine="rasterio")
    # Squeeze to remove band dimension if present (shape goes from (1, H, W) to (H, W))
    if dem.ndim == 3:
        dem = dem.squeeze()
    print(f"    Shape: {dem.shape}")
    print(f"    CRS: {dem.rio.crs}")
    print(f"    Bounds: {dem.rio.bounds()}")
    print(f"    Transform: {dem.rio.transform}")
    
    # Step 2: Reproject to EPSG:4326 (WGS84 lat/lon)
    print("\n[2/5] Reprojecting to EPSG:4326...")
    dem_4326 = dem.rio.reproject("EPSG:4326")
    # Squeeze again in case reprojection adds a dimension
    if dem_4326.ndim == 3:
        dem_4326 = dem_4326.squeeze()
    print(f"    Reprojected shape: {dem_4326.shape}")
    print(f"    New CRS: {dem_4326.rio.crs}")
    print(f"    New bounds: {dem_4326.rio.bounds()}")
    
    # Save reprojected DEM
    print(f"\n[3/5] Saving reprojected DEM to {reprojected_path}...")
    dem_4326.rio.to_raster(reprojected_path)
    print("    Saved successfully!")
    
    # Step 3: Calculate slope using xarray-spatial
    print("\n[4/5] Calculating slope with geodesic method...")
    # For geographic coordinates (lat/lon), use method='geodesic' for accurate results
    slope_result = slope(dem_4326, method='geodesic', boundary='nan')
    print(f"    Slope shape: {slope_result.shape}")
    print(f"    Slope range: {float(slope_result.min()):.2f}° to {float(slope_result.max()):.2f}°")
    print(f"    Slope mean: {float(slope_result.mean()):.2f}°")
    print(f"    Slope std: {float(slope_result.std()):.2f}°")
    
    # Calculate statistics on valid slope values
    slope_values = slope_result.values.flatten()
    valid_slope = slope_values[~np.isnan(slope_values)]
    if len(valid_slope) > 0:
        print(f"    Valid slope pixels: {len(valid_slope)}")
        print(f"    Slope median: {float(np.median(valid_slope)):.2f}°")
        # Percentage of area with different slope ranges
        flat_count = np.sum(valid_slope <= 5)
        gentle_count = np.sum((valid_slope > 5) & (valid_slope <= 15))
        moderate_count = np.sum((valid_slope > 15) & (valid_slope <= 30))
        steep_count = np.sum(valid_slope > 30)
        
        flat_pct = (flat_count / len(valid_slope)) * 100
        gentle_pct = (gentle_count / len(valid_slope)) * 100
        moderate_pct = (moderate_count / len(valid_slope)) * 100
        steep_pct = (steep_count / len(valid_slope)) * 100
        
        print(f"    Terrain breakdown:")
        print(f"      Flat (0-5°): {flat_pct:.1f}%")
        print(f"      Gentle (5-15°): {gentle_pct:.1f}%")
        print(f"      Moderate (15-30°): {moderate_pct:.1f}%")
        print(f"      Steep (>30°): {steep_pct:.1f}%")
    
    # Save slope result
    print(f"\n[5/5] Saving slope result to {slope_path}...")
    slope_result.rio.write_crs("EPSG:4326", inplace=True)
    slope_result.rio.to_raster(slope_path)
    print("    Saved successfully!")
    
    # Step 6: Visualize with matplotlib
    print("\n[Bonus] Creating matplotlib visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Original DEM
    ax1 = plt.subplot(2, 2, 1)
    dem_4326.plot.imshow(
        ax=ax1,
        cmap='terrain',
        add_colorbar=True,
        cbar_kwargs={'label': 'Elevation (m)', 'shrink': 0.8}
    )
    ax1.set_title('Pori DEM (EPSG:4326)', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Subplot 2: Slope
    ax2 = plt.subplot(2, 2, 2)
    slope_result.plot.imshow(
        ax=ax2,
        cmap='YlOrRd',
        add_colorbar=True,
        cbar_kwargs={'label': 'Slope (°)', 'shrink': 0.8},
        vmin=0,
        vmax=90
    )
    ax2.set_title('Slope (degrees) - Geodesic Method', fontsize=14)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Subplot 3: Slope histogram
    ax3 = plt.subplot(2, 2, 3)
    slope_values = slope_result.values.flatten()
    slope_values = slope_values[~np.isnan(slope_values)]
    ax3.hist(slope_values, bins=50, color='orange', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Slope (°)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Slope Distribution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Combined DEM + Slope
    ax4 = plt.subplot(2, 2, 4)
    # Plot DEM in grayscale as base
    dem_4326.plot.imshow(
        ax=ax4,
        cmap='gray',
        add_colorbar=False,
        alpha=0.7
    )
    # Overlay slope with transparency
    slope_norm = slope_result / slope_result.max()
    slope_norm.plot.imshow(
        ax=ax4,
        cmap='hot',
        add_colorbar=False,
        alpha=0.5,
        vmin=0,
        vmax=1
    )
    ax4.set_title('DEM + Slope Overlay', fontsize=14)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dem_and_slope_analysis.png"), dpi=150, bbox_inches='tight')
    print(f"    Figure saved to {output_dir}/dem_and_slope_analysis.png")
    
    # Also save individual plots
    fig2, ax = plt.subplots(figsize=(10, 8))
    dem_4326.plot.imshow(
        ax=ax,
        cmap='terrain',
        add_colorbar=True,
        cbar_kwargs={'label': 'Elevation (m)'}
    )
    ax.set_title('Pori DEM (EPSG:4326)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dem_4326.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    fig3, ax = plt.subplots(figsize=(10, 8))
    slope_result.plot.imshow(
        ax=ax,
        cmap='YlOrRd',
        add_colorbar=True,
        cbar_kwargs={'label': 'Slope (°)'},
        vmin=0,
        vmax=90
    )
    ax.set_title('Slope (degrees)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slope.png"), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - {os.path.basename(reprojected_path)}")
    print(f"  - {os.path.basename(slope_path)}")
    print(f"  - dem_4326.png")
    print(f"  - slope.png")
    print(f"  - dem_and_slope_analysis.png")
    print("\nYou can view the results with:")
    print(f"  pixi run python -c \"import matplotlib.pyplot as plt; import matplotlib.image as mpimg; img=mpimg.imread('{output_dir}/dem_and_slope_analysis.png'); plt.imshow(img); plt.axis('off'); plt.show()\"")


if __name__ == "__main__":
    main()
