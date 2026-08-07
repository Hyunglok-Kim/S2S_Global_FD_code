# -*- coding: utf-8 -*-

import glob
import os
import warnings
from typing import Dict, List

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==============================================================================
# AUTO BUILD VAR MAP
# ==============================================================================


def build_var_map(era_folder: str) -> Dict[str, str]:
    prefix_map = {
        "sm": "sm",
        "t2m": "t2m",
        "td": "d2m",  # dewpoint mapping
        "lhf": "lhf",
        "tp": "tp",
    }

    var_map = {}
    for s2s_var, era_var in prefix_map.items():
        era_path = os.path.join(era_folder, f"{era_var}_era5_pentad.nc")
        var_map[f"{s2s_var}_s2s"] = era_path
        var_map[f"{s2s_var}_obs"] = era_path

    return var_map


# ==============================================================================
# LOAD ERA CACHE (Lazy Dask)
# ==============================================================================


def load_era_cache(
    var_map: Dict[str, str], lon_size: int
) -> Dict[str, xr.Dataset]:
    era_cache = {}
    for var, path in tqdm(var_map.items(), desc="Opening ERA files (Dask)"):
        # Use chunks to trigger Dask lazy loading
        ds = xr.open_dataset(
            path,
            chunks={"year": 1, "pentad": -1, "lat": "auto", "lon": "auto"},
        )
        ds = ds.isel(lon=slice(0, lon_size))
        ds = ds.sortby("lat")
        era_cache[var] = ds
    return era_cache


# ==============================================================================
# FAST STANDARDIZER
# ==============================================================================


class FastS2SStandardizer:
    def __init__(
        self,
        s2s_path: str,
        var_map: Dict[str, str],
        era_cache: Dict[str, xr.Dataset],
        output_dir: str,
    ):
        self.s2s_path = s2s_path
        self.var_map = var_map
        self.era_cache = era_cache
        self.output_dir = output_dir

        self.ds: xr.Dataset = None
        self.years: np.ndarray = None
        self.pentad: xr.DataArray = None

        os.makedirs(output_dir, exist_ok=True)

    def load_s2s(self) -> None:
        # Open S2S with Dask chunks
        ds = xr.open_dataset(
            self.s2s_path, chunks={"step": 1, "lat": "auto", "lon": "auto"}
        )

        # Remove duplicate longitudes
        _, idx = np.unique(ds["lon"].values, return_index=True)
        ds = ds.isel(lon=np.sort(idx))

        self.ds = ds
        self.years = ds["valid_time"].dt.year.values

        # Pentad calculation
        doy = ds["valid_time"].dt.dayofyear
        pentad = (doy - 1) // 5 + 1
        self.pentad = ((pentad - 1) % 73) + 1

    def process_var(self, var: str) -> xr.DataArray:
        ds_era = self.era_cache[var]
        era_var = list(ds_era.data_vars)[0]

        mean_list: List[xr.DataArray] = []
        std_list: List[xr.DataArray] = []

        for i in range(len(self.pentad)):
            # Dask handles filtering and lazy computation
            tmp = ds_era[era_var].where(
                ds_era.year != self.years[i], drop=True
            ).isel(pentad=int(self.pentad[i] - 1))

            mean = tmp.mean(dim="year")
            std = tmp.std(dim="year")

            mean_list.append(mean.expand_dims(step=[i]))
            std_list.append(std.expand_dims(step=[i]))

        mean_all = xr.concat(mean_list, dim="step").assign_coords(
            step=self.ds["step"]
        )
        std_all = xr.concat(std_list, dim="step").assign_coords(
            step=self.ds["step"]
        )

        mean_all = mean_all.reindex_like(self.ds, method="nearest")
        std_all = std_all.reindex_like(self.ds, method="nearest")

        safe_std = xr.where(std_all < 1e-3, np.nan, std_all)
        res = (self.ds[var] - mean_all) / safe_std

        return res.where((res >= -5) & (res <= 5))

    def run(self) -> None:
        out_path = os.path.join(
            self.output_dir,
            os.path.basename(self.s2s_path).replace(".nc", "_std.nc"),
        )

        # Skip if already calculated
        if os.path.exists(out_path):
            print(f"Skipped (exists): {out_path}")
            return

        print(f"\nProcessing: {os.path.basename(self.s2s_path)}")
        self.load_s2s()

        ds_out = xr.Dataset(coords=self.ds.coords)
        vars_list = list(self.var_map.keys())

        for v in vars_list:
            ds_out[f"{v}_std"] = self.process_var(v)

        encoding = {
            v: {"zlib": True, "complevel": 5} for v in ds_out.data_vars
        }

        ds_out.to_netcdf(out_path, encoding=encoding)
        print("Saved:", out_path)


# ==============================================================================
# MULTI FILE PROCESSOR
# ==============================================================================


def fast_process_all(
    s2s_folder: str, era_folder: str, output_folder: str
) -> None:
    # Initialize Dask Cluster
    cluster = LocalCluster(
        n_workers=16, threads_per_worker=1, memory_limit="16GB"
    )
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    var_map = build_var_map(era_folder)
    files = sorted(
        glob.glob(os.path.join(s2s_folder, "S2S_ERA5_pentad_init_*.nc"))
    )
    print(f"\nFound {len(files)} S2S files")

    # Get longitude length for cache building
    with xr.open_dataset(files[0]) as temp:
        _, idx = np.unique(temp["lon"].values, return_index=True)
        lon_size = len(idx)

    era_cache = load_era_cache(var_map, lon_size)

    for f in tqdm(files, desc="Processing S2S files"):
        FastS2SStandardizer(f, var_map, era_cache, output_folder).run()

    client.close()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    fast_process_all(
        s2s_folder="/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/S2S_ERA5_Pentad",
        era_folder="/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/ERA5_Pentad/Pentads",
        output_folder="/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/STD",
    )
