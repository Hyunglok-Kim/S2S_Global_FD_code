# -*- coding: utf-8 -*-
"""Groundwater STD File Converter to FDII Standard Format."""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

# ==============================================================================
# CONFIG & PATHS
# ==============================================================================

INPUT_DIR = "/home/venkateshb/cpudata/personal_data/yusang/ground_water/std"
OUTPUT_DIR = (
    "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/STD/GW_STD"
)


# ==============================================================================
# PROCESSING FUNCTIONS
# ==============================================================================


def process_groundwater_file(
    file_path: str, output_dir: str, target_lat: np.ndarray
) -> None:
    """Load, regrid, reformat, and save a groundwater standardized NetCDF file."""
    print(f"Processing: {os.path.basename(file_path)}")

    with xr.open_dataset(file_path) as ds:
        # Sort latitude
        ds = ds.sortby("lat")

        # Convert and sort longitude (-180 to 180)
        lon_new = ((ds.lon + 180) % 360) - 180
        ds = ds.assign_coords(lon=lon_new).sortby("lon")

        # Interpolate to target latitude grid
        ds = ds.interp(lat=target_lat, method="nearest")

        # Convert step values to timedelta
        step_days = ds.step.values * 5
        step_timedelta = pd.to_timedelta(step_days, unit="D")
        ds = ds.assign_coords(step=("step", step_timedelta))

        # Enforce float64 data types for coordinates
        ds["lat"] = ds["lat"].astype(np.float64)
        ds["lon"] = ds["lon"].astype(np.float64)

        # Update global attributes
        ds.attrs.update(
            {
                "description": (
                    "Standardized groundwater pentad forecast and "
                    "observation for FDII framework"
                ),
                "standardization_method": (
                    "LOYA (Leave-One-Year-Out across 2016-2025)"
                ),
            }
        )

        # Set output file path and compression encodings
        out_file = os.path.join(output_dir, os.path.basename(file_path))
        encoding = {
            var: {"zlib": True, "complevel": 5, "dtype": "float32"}
            for var in ds.data_vars
        }

        # Save processed NetCDF
        ds.to_netcdf(out_file, encoding=encoding)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nc")))
    total_files = len(files)
    print(f"\nTotal Files Found : {total_files}")

    target_lat = np.arange(-90, 90.25, 0.25)

    for i, file_path in enumerate(files, 1):
        print("\n" + "=" * 50)
        print(f"Processing File {i} / {total_files}")
        print("=" * 50)

        process_groundwater_file(file_path, OUTPUT_DIR, target_lat)

    print("\n" + "=" * 50)
    print("ALL FILES COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
