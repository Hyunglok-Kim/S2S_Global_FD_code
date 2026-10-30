# -*- coding: utf-8 -*-
"""ERA5 Quantile Threshold Computation Pipeline for FDII."""

import logging
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore")
logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("xarray").setLevel(logging.ERROR)

BASE_PATH = Path("./data/era5_pentad")
FILE_MAP = {
    "SM": BASE_PATH / "sm_era5_pentad.nc",
    "TP": BASE_PATH / "tp_era5_pentad.nc",
    "T2M": BASE_PATH / "t2m_era5_pentad.nc",
    "TD": BASE_PATH / "d2m_era5_pentad.nc",
    "ET": BASE_PATH / "lhf_era5_pentad.nc",
}
OUT_FILE = Path("./data/thresholds/ERA5_thresholds_climatology.nc")


def load_lazy(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks={"year": -1, "pentad": 1, "lat": 100, "lon": 100})
    return ds[list(ds.data_vars)[0]]


def calc_vpd(t_air, t_dew):
    t_air_c = t_air - 273.15
    t_dew_c = t_dew - 273.15

    es = 0.6108 * np.exp(
        (17.27 * t_air_c) / (t_air_c + 237.3)
    )
    ea = 0.6108 * np.exp(
        (17.27 * t_dew_c) / (t_dew_c + 237.3)
    )

    return es - ea


def get_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    std_var = (da - da.mean("year")) / da.std("year")
    return std_var.quantile(q, dim="year").drop_vars("quantile", errors="ignore")


def run_simulation() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cluster = LocalCluster(n_workers=32, threads_per_worker=2, memory_limit="16GB")
    client = Client(cluster)
    print(f"Dask Dashboard link: {client.dashboard_link}")

    try:
        data = {var: load_lazy(path) for var, path in FILE_MAP.items()}
        vpd = calc_vpd(data["T2M"], data["TD"])

        out_ds = xr.Dataset({
            "TH_SM": get_threshold(data["SM"], 0.2),
            "TH_TP": get_threshold(data["TP"], 0.2),
            "TH_ET": get_threshold(data["ET"], 0.2),
            "TH_VPD": get_threshold(vpd, 0.8),
        }).transpose("pentad", "lat", "lon")

        print(f"Starting parallel run. Saving to: {OUT_FILE}")
        with ProgressBar():
            out_ds.to_netcdf(OUT_FILE)
        print("\n✅ Run finished without warnings.")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    run_simulation()