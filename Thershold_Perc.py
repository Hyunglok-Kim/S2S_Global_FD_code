# -*- coding: utf-8 -*-
"""ERA5 Quantile Threshold Computation Pipeline for FDII."""

import logging
import warnings
from typing import Dict

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

# ==============================================================================
# 0. WARNINGS & LOGGING CONFIG
# ==============================================================================

warnings.filterwarnings("ignore")
logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("xarray").setLevel(logging.ERROR)

# ==============================================================================
# CONFIG & PATHS
# ==============================================================================

BASE_PATH = (
    "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/"
    "ERA5_Pentad/Pentads/"
)

FILE_MAP = {
    "SM": f"{BASE_PATH}sm_era5_pentad.nc",
    "TP": f"{BASE_PATH}tp_era5_pentad.nc",
    "T2M": f"{BASE_PATH}t2m_era5_pentad.nc",
    "TD": f"{BASE_PATH}d2m_era5_pentad.nc",
    "ET": f"{BASE_PATH}lhf_era5_pentad.nc",
}

OUT_FILE = (
    "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/"
    "Threshold/ERA5_thresholds_loya.nc"
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def load_lazy(path: str) -> xr.DataArray:
    """Lazy load NetCDF variables using optimized Dask chunks."""
    ds = xr.open_dataset(
        path, chunks={"year": -1, "pentad": 1, "lat": 100, "lon": 100}
    )
    return ds[list(ds.data_vars)[0]]


def calc_vpd(t_air: xr.DataArray, t_dew: xr.DataArray) -> xr.DataArray:
    """Calculate Vapor Pressure Deficit (VPD) from air and dewpoint temperatures."""
    es = 0.6108 * np.exp((17.27 * t_air) / (t_air + 237.3))
    ea = 0.6108 * np.exp((17.27 * t_dew) / (t_dew + 237.3))
    return es - ea


def get_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    """Standardize input DataArray and extract the quantile threshold along 'year'."""
    clim_mean = da.mean(dim="year")
    clim_std = da.std(dim="year")
    std_var = (da - clim_mean) / clim_std

    res = std_var.quantile(q, dim="year").drop_vars("quantile", errors="ignore")
    return res


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================


def run_simulation() -> None:
    """Run parallel threshold extraction using Dask distributed."""
    cluster = LocalCluster(
        n_workers=32, threads_per_worker=2, memory_limit="16GB"
    )
    client = Client(cluster)
    print(f"Dask Dashboard link: {client.dashboard_link}")

    try:
        print("Loading datasets lazily...")
        data: Dict[str, xr.DataArray] = {
            var: load_lazy(path) for var, path in FILE_MAP.items()
        }

        # Calculate VPD
        vpd = calc_vpd(data["T2M"], data["TD"])

        print("Building computation graph...")
        th_sm = get_threshold(data["SM"], 0.2)
        th_tp = get_threshold(data["TP"], 0.2)
        th_et = get_threshold(data["ET"], 0.2)
        th_vpd = get_threshold(vpd, 0.8)

        # Merge results into output Dataset
        out_ds = xr.Dataset(
            {
                "TH_SM": th_sm,
                "TH_TP": th_tp,
                "TH_ET": th_et,
                "TH_VPD": th_vpd,
            }
        )

        out_ds = out_ds.transpose("pentad", "lat", "lon")

        print(f"Starting parallel run. Saving to: {OUT_FILE}")
        with ProgressBar():
            out_ds.to_netcdf(OUT_FILE)

        print("\n✅ Run finished without warnings.")

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    run_simulation()
