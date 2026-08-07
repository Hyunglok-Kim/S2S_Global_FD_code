# ==============================================================================
# MULTIVARIATE FDII PIPELINE
# ==============================================================================

import glob
import os
import numpy as np
import xarray as xr

# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_DIR = "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/STD"
GW_DIR = "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/STD/GW_STD"
THRES_FILE = "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/Threshold/ERA5_thresholds_loya.nc"
OUT_DIR = "/home/venkateshb/cpudata/personal_data/Venky/S2S_FD/Processed/FDII/FDII_MFDI_GW"

os.makedirs(OUT_DIR, exist_ok=True)
INPUT_FILES = sorted(glob.glob(os.path.join(INPUT_DIR, "*_std.nc")))


# ==============================================================================
# UTILITIES
# ==============================================================================


def print_stats(name: str, da: xr.DataArray) -> None:
    try:
        print(f"\n🔹 {name}")
        print(f"   min  : {float(da.min())}")
        print(f"   max  : {float(da.max())}")
        print(f"   mean : {float(da.mean())}")
        print(f"   std  : {float(da.std())}")
    except Exception:
        pass


def normalize(x: xr.DataArray) -> xr.DataArray:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


# ==============================================================================
# VARIABLE EXTRACTION
# ==============================================================================


def get_variables(ds: xr.Dataset, prefix: str) -> dict[str, xr.DataArray]:
    return {
        "SM": ds[f"sm_{prefix}_std"],
        "TP": ds[f"tp_{prefix}_std"],
        "T2M": ds[f"t2m_{prefix}_std"],
        "TD": ds[f"td_{prefix}_std"],
        "ET": ds[f"lhf_{prefix}_std"],
    }


def get_gw(ds_gw: xr.Dataset, prefix: str) -> xr.DataArray:
    possible = [
        f"gw_{prefix}_std",
        f"gws_{prefix}_std",
        f"tws_{prefix}_std",
    ]

    for v in possible:
        if v in ds_gw.data_vars:
            print(f"\n✅ GW VARIABLE FOUND : {v}")
            return ds_gw[v]

    raise ValueError("❌ No GW variable found in dataset.")


# ==============================================================================
# CALCULATION HELPERS
# ==============================================================================


def compute_vpd(t2m: xr.DataArray, td: xr.DataArray) -> xr.DataArray:
    return (t2m - td) / np.sqrt(2)


def compute_fd_int(da: xr.DataArray, sign: int) -> xr.DataArray:
    dx = sign * (da - da.shift(step=1))
    dx = dx.rolling(step=3, min_periods=3).max()
    return dx.where(dx > 0, 0)


def get_thresholds(
    ds: xr.Dataset, ds_th: xr.Dataset
) -> dict[str, xr.DataArray]:
    doy = ds["valid_time"].dt.dayofyear
    pentad = ((doy - 1) // 5 + 1).clip(max=73)

    th_sm = ds_th["TH_SM"].sel(pentad=pentad)
    th_tp = ds_th["TH_TP"].sel(pentad=pentad)
    th_et = ds_th["TH_ET"].sel(pentad=pentad)
    th_vpd = ds_th["TH_VPD"].sel(pentad=pentad)

    # Clean pentad coordinate artifacts
    thresholds = {}
    for key, da in zip(
        ["SM", "TP", "ET", "VPD"], [th_sm, th_tp, th_et, th_vpd]
    ):
        if "pentad" in da.coords:
            da = da.drop_vars("pentad")
        thresholds[key] = da.drop_vars("pentad", errors="ignore")

    return thresholds


def drought_mask(
    vars_dict: dict[str, xr.DataArray], th: dict[str, xr.DataArray]
) -> dict[str, xr.DataArray]:
    return {
        "SM": vars_dict["SM"] < th["SM"],
        "TP": vars_dict["TP"] < th["TP"],
        "VPD": vars_dict["VPD"] > th["VPD"],
        "ET": vars_dict["ET"] < th["ET"],
    }


def persistence(
    mask: dict[str, xr.DataArray]
) -> dict[str, xr.DataArray]:
    return {
        k: v.rolling(step=3, min_periods=3).sum() >= 3 for k, v in mask.items()
    }


def severity(
    vars_dict: dict[str, xr.DataArray], mask: dict[str, xr.DataArray]
) -> dict[str, xr.DataArray]:
    return {
        k: np.abs(vars_dict[k].where(mask[k], 0))
        .rolling(step=3, min_periods=3)
        .mean()
        for k in ["SM", "TP", "VPD", "ET"]
    }


def compute_gw_modifier(gw: xr.DataArray) -> xr.DataArray:
    gw_def = normalize(-gw)
    gw_trend = normalize(-(gw - gw.shift(step=1)))
    return 1 + 0.3 * (0.7 * gw_def + 0.3 * gw_trend)


def dro_sev(
    sev: dict[str, xr.DataArray],
    pers: dict[str, xr.DataArray],
    gw_mod: xr.DataArray,
) -> dict[str, xr.DataArray]:
    print("STEP 9 : COMPUTING DRO-SEV")
    dro = {k: ((1 + sev[k]) * gw_mod).where(pers[k], 0) for k in sev}

    for k, v in dro.items():
        print_stats(f"{k} DRO-SEV", v)

    return dro


def compute_fdii(
    fd_int: dict[str, xr.DataArray], dro: dict[str, xr.DataArray]
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    fdii_sm = (fd_int["SM"] * dro["SM"]).where((fd_int["SM"] * dro["SM"]) > 0, 0)
    fdii_tp = (fd_int["TP"] * dro["TP"]).where((fd_int["TP"] * dro["TP"]) > 0, 0)
    fdii_et = (fd_int["ET"] * dro["ET"]).where((fd_int["ET"] * dro["ET"]) > 0, 0)
    fdii_vpd = (fd_int["VPD"] * dro["VPD"]).where(
        (fd_int["VPD"] * dro["VPD"]) > 0, 0
    )

    fdii_sum = fdii_sm + fdii_tp + fdii_et + fdii_vpd

    active = (
        (fdii_sm > 0).astype(int)
        + (fdii_tp > 0).astype(int)
        + (fdii_et > 0).astype(int)
        + (fdii_vpd > 0).astype(int)
    )

    fdii_comb = xr.where(active > 0, fdii_sum / active, 0)
    fdii_comb = fdii_comb.where(active >= 2, 0)

    fd_flag = (fdii_comb > 0).astype(int)

    return fdii_sm, fdii_tp, fdii_et, fdii_vpd, fdii_comb, fd_flag, active


def extract_events(fd_flag: xr.DataArray) -> xr.DataArray:
    start = (fd_flag == 1) & (fd_flag.shift(step=1, fill_value=0) == 0)
    event_id = start.cumsum(dim="step")
    return event_id.where(fd_flag == 1)


def compute_metrics(
    event_id: xr.DataArray,
    fd_flag: xr.DataArray,
    fdii: xr.DataArray,
    fd_int: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    frequency = event_id.max("step")
    severity_total = fdii.where(fd_flag == 1).sum("step")

    fd_int_comb = xr.concat(list(fd_int.values()), dim="var").max("var")
    intensity = fd_int_comb.where(fd_flag == 1).max("step")

    mfdi = (
        normalize(frequency) + normalize(severity_total) + normalize(intensity)
    ) / 3

    return frequency, severity_total, intensity, mfdi


# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================


def run_pipeline(
    ds: xr.Dataset,
    ds_gw: xr.Dataset,
    ds_th: xr.Dataset,
    prefix: str,
    tag: str,
) -> xr.Dataset:
    vars_dict = get_variables(ds, prefix)
    vars_dict["VPD"] = compute_vpd(vars_dict["T2M"], vars_dict["TD"])

    gw = get_gw(ds_gw, prefix)
    print_stats(f"{tag} GW", gw)

    fd_int = {
        "SM": compute_fd_int(vars_dict["SM"], -1),
        "TP": compute_fd_int(vars_dict["TP"], -1),
        "VPD": compute_fd_int(vars_dict["VPD"], 1),
        "ET": compute_fd_int(vars_dict["ET"], -1),
    }

    th = get_thresholds(ds, ds_th)
    mask = drought_mask(vars_dict, th)
    pers = persistence(mask)
    sev = severity(vars_dict, mask)

    gw_mod = compute_gw_modifier(gw)
    print_stats(f"{tag} GW MODIFIER", gw_mod)

    dro = dro_sev(sev, pers, gw_mod)

    (
        fdii_sm,
        fdii_tp,
        fdii_et,
        fdii_vpd,
        fdii_comb,
        fd_flag,
        active,
    ) = compute_fdii(fd_int, dro)

    event_id = extract_events(fd_flag)
    frequency, severity_total, intensity, mfdi = compute_metrics(
        event_id, fd_flag, fdii_comb, fd_int
    )

    print_stats(f"{tag} FDII", fdii_comb)

    return xr.Dataset(
        {
            # Final Flash Drought Products
            f"{tag}_FDII_SM": fdii_sm,
            f"{tag}_FDII_TP": fdii_tp,
            f"{tag}_FDII_ET": fdii_et,
            f"{tag}_FDII_VPD": fdii_vpd,
            f"{tag}_FDII_COMB": fdii_comb,
            f"{tag}_FD_FLAG": fd_flag.astype(np.int8),
            f"{tag}_MFDI": mfdi,
            # Event Information
            f"{tag}_EVENT_ID": event_id,
            # Final Metrics
            f"{tag}_FREQUENCY": frequency,
            f"{tag}_SEVERITY_TOTAL": severity_total,
            f"{tag}_INTENSITY": intensity,
            # FD Intensification Components
            f"{tag}_FD_INT_SM": fd_int["SM"],
            f"{tag}_FD_INT_TP": fd_int["TP"],
            f"{tag}_FD_INT_ET": fd_int["ET"],
            f"{tag}_FD_INT_VPD": fd_int["VPD"],
            # Final Drought Severity Weight
            f"{tag}_DRO_SM": dro["SM"],
            f"{tag}_DRO_TP": dro["TP"],
            f"{tag}_DRO_ET": dro["ET"],
            f"{tag}_DRO_VPD": dro["VPD"],
            # Multivariate Agreement
            f"{tag}_ACTIVE_COUNT": active.astype(np.int8),
        }
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Main processing loop for S2S GLDAS datasets."""
    ds_th = xr.open_dataset(THRES_FILE)

    # Match threshold grid using reference file
    with xr.open_dataset(INPUT_FILES[0]) as sample_ds:
        ds_th = ds_th.interp(lat=sample_ds.lat, lon=sample_ds.lon)

    for input_file in INPUT_FILES:
        print("\n" + "=" * 50)
        print(f"INITIALIZATION : {os.path.basename(input_file)}")
        print("=" * 50)

        init_date = os.path.basename(input_file).split("_")[-2]
        gw_pattern = os.path.join(GW_DIR, f"S2S_GLDAS_*_{init_date}_std.nc")
        gw_match = glob.glob(gw_pattern)

        if not gw_match:
            print(f"❌ NO GW FILE FOUND FOR {init_date}")
            continue

        gw_file = gw_match[0]

        print("\nSTEP 1 : LOADING DATASETS")
        ds = xr.open_dataset(input_file)
        ds_gw = xr.open_dataset(gw_file)

        print(f"✅ INPUT LOADED: {os.path.basename(input_file)}")
        print(f"✅ GW LOADED:    {os.path.basename(gw_file)}")

        # Outlier Cleaning
        print("\n⚠️ CLEANING GW OUTLIERS")
        for var in ds_gw.data_vars:
            before_min, before_max = float(ds_gw[var].min()), float(
                ds_gw[var].max()
            )
            ds_gw[var] = ds_gw[var].where((ds_gw[var] > -5) & (ds_gw[var] < 5))
            after_min, after_max = float(ds_gw[var].min()), float(
                ds_gw[var].max()
            )
            bad_pixels = int(np.isnan(ds_gw[var]).sum().values)

            print(
                f"🔹 {var:10} | Range: [{before_min:7.4f}, {before_max:7.4f}] -> "
                f"[{after_min:7.4f}, {after_max:7.4f}] | Cleaned: {bad_pixels}"
            )

        # Run pipelines (S2S & ERA)
        out_s2s = run_pipeline(ds, ds_gw, ds_th, "s2s", "S2S")
        out_era = run_pipeline(ds, ds_gw, ds_th, "obs", "ERA")

        out = xr.merge([out_s2s, out_era], compat="override")

        # Save Output
        out_file = os.path.join(
            OUT_DIR, os.path.basename(input_file).replace("_std", "_FDII_GW")
        )

        encoding = {
            var: dict(zlib=True, complevel=6) for var in out.data_vars
        }
        out.to_netcdf(out_file, encoding=encoding)
        print(f"\n✅ SAVED : {out_file}")

        # Cleanup
        ds.close()
        ds_gw.close()
        out.close()

    ds_th.close()


if __name__ == "__main__":
    main()
