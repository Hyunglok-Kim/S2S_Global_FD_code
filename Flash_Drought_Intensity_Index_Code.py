"""Flash-drought intensity and MFDI processing.

This annotated version preserves the supplied calculation logic. REVIEW comments
identify assumptions or potential issues that should be checked before production.

Expected inputs:
  - Standardized S2S/ERA5 land-atmosphere files containing *_s2s_std and *_obs_std variables.
  - Standardized groundwater files containing gw/gws/tws s2s and obs variables.
  - Pentad-specific threshold NetCDF with TH_SM, TH_TP, TH_ET, and TH_VPD.

Important: This script currently selects only the first N_CALIBRATION files.
It does not automatically process a separate validation period.
"""

import glob
import os
import numpy as np
import xarray as xr


# =========================================================
# CONFIG
# =========================================================
from pathlib import Path

# Define the project root directory based on the location of this script.
# Assumes this script is stored inside the fdii_generation/ folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directory containing standardized S2S and ERA5 land-atmosphere NetCDF files.
INPUT_DIR = PROJECT_DIR / "data" / "s2s_standardized"

# Directory containing standardized groundwater (GLDAS/MLP) NetCDF files.
GW_DIR = PROJECT_DIR / "data" / "groundwater_standardized"

# ERA5 pentad-specific climatological thresholds for SM, TP, ET and VPD.
THRES_FILE = (
    PROJECT_DIR / "data" / "thresholds" / "ERA5_thresholds_climatology.nc"
)

# Output directory for generated FDII and MFDI NetCDF products.
OUT_DIR = PROJECT_DIR / "outputs" / "FDII_MFDI"

# Create the output directory automatically if it does not already exist.
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Identify all standardized input NetCDF files and sort them by filename.
# Ensure the filename structure supports chronological sorting.
ALL_INPUT_FILES = sorted(glob.glob(os.path.join(INPUT_DIR, "*_std.nc")))

# Number of initialization files selected for the calibration period.
N_CALIBRATION = 208

# Select the first 208 sorted files for calibration-period processing.
# Set N_CALIBRATION to the desired number of files for another experiment.
INPUT_FILES = ALL_INPUT_FILES[:N_CALIBRATION]

# IMPORTANT:
# This configuration processes only the first 208 sorted input files.
# It does not automatically process a separate validation period.
# Verify that filename sorting corresponds to the intended calibration
# period and that all selected files have matching groundwater datasets.

# =========================================================
# GROUNDWATER MODIFIER PARAMETERS
# =========================================================

ALPHA_GW = 0.2  # Groundwater influence coefficient controlling the
                # maximum relative contribution of groundwater conditions
                # to the drought-severity modifier.

WSD = 0.6       # Weight assigned to groundwater storage deficit (GW_SD),
                # representing the groundwater deficit contribution.

WTR = 0.4       # Weight assigned to groundwater depletion trend (GW_DR),
                # representing the contribution of groundwater decline.

# NOTE:
# WSD + WTR = 1.0, ensuring that the two groundwater components
# contribute through a normalized weighted combination.
#
# The groundwater modifier is calculated as:
# GW_MOD = 1 + ALPHA_GW * (WSD * GW_SD + WTR * GW_DR)
#
# Since GW_SD and GW_DR are independently normalized to [0, 1],
# the modifier theoretically ranges from 1.0 to 1.2.
# Thus, groundwater conditions amplify drought severity but do not
# reduce it below the baseline severity.
#
# IMPORTANT: These parameters should be documented with their
# calibration or sensitivity-analysis rationale in the manuscript.

# =========================================================
# STATISTICS
# =========================================================
def print_stats(name, da):
    try:
        finite = da.where(np.isfinite(da))
        print(f"\n🔹 {name}")
        print(f"    min  : {float(finite.min(skipna=True)):.6f}")
        print(f"    max  : {float(finite.max(skipna=True)):.6f}")
        print(f"    mean : {float(finite.mean(skipna=True)):.6f}")
        print(f"    std  : {float(finite.std(skipna=True)):.6f}")
        print(f"    NaN  : {int(da.isnull().sum())}")
    except Exception as e:
        print(f"    ⚠️ Statistics failed: {e}")


def print_mask_stats(name, da):
    try:
        total = da.size
        active = int(da.sum(skipna=True))
        pct = 100 * active / total if total else 0
        print(f"    {name:12} : {active:,}/{total:,} ({pct:.4f}%)")
    except Exception:
        pass


# =========================================================
# NORMALIZATION
# =========================================================
def normalize(x):
    xmin = x.min(skipna=True)
    xmax = x.max(skipna=True)
    return (x - xmin) / (xmax - xmin + 1e-6)


# =========================================================
# VARIABLES
# =========================================================


def get_variables(ds, prefix):
    """
    Extract standardized hydroclimatic variables.

    VPD must be calculated from physical T2M and TD
    before standardization during preprocessing.
    """

    return {
        "SM":  ds[f"sm_{prefix}_std"],
        "TP":  ds[f"tp_{prefix}_std"],
        "ET":  ds[f"lhf_{prefix}_std"],
        "VPD": ds[f"vpd_{prefix}_std"],
    }

# =========================================================
# GROUNDWATER VARIABLES
# =========================================================
# Identify the available standardized groundwater variable.
# Supports groundwater storage variables with GW, GWS or TWS prefixes.
def get_gw(ds_gw, prefix):
    possible = [f"gw_{prefix}_std", f"gws_{prefix}_std", f"tws_{prefix}_std"]
    for v in possible:
        if v in ds_gw.data_vars:
            print(f"\n✅ GW VARIABLE FOUND : {v}")
            return ds_gw[v]
    raise ValueError(f"❌ No GW variable found for prefix={prefix}")




# =========================================================
# FLASH-DROUGHT INTENSITY
# =========================================================
def compute_fd_int(X, sign):
    dX = sign * (X - X.shift(step=1))
    dX = dX.rolling(step=3, min_periods=3).max()
    return dX.where(dX > 0, 0)


# =========================================================
# THRESHOLDS
# =========================================================
def get_thresholds(ds, ds_th):
    doy = ds["valid_time"].dt.dayofyear
    pentad = ((doy - 1) // 5 + 1).clip(max=73)
    th_sm = ds_th["TH_SM"].sel(pentad=pentad)
    th_tp = ds_th["TH_TP"].sel(pentad=pentad)
    th_et = ds_th["TH_ET"].sel(pentad=pentad)
    th_vpd = ds_th["TH_VPD"].sel(pentad=pentad)
    return {
        "SM": th_sm.drop_vars("pentad", errors="ignore"),
        "TP": th_tp.drop_vars("pentad", errors="ignore"),
        "ET": th_et.drop_vars("pentad", errors="ignore"),
        "VPD": th_vpd.drop_vars("pentad", errors="ignore"),
    }


# =========================================================
# DROUGHT MASK
# =========================================================
def drought_mask(vars, th):
    mask = {
        "SM": vars["SM"] < th["SM"],
        "TP": vars["TP"] < th["TP"],
        "VPD": vars["VPD"] > th["VPD"],
        "ET": vars["ET"] < th["ET"],
    }
    print("\n📊 DROUGHT MASK COVERAGE")
    for k, v in mask.items():
        print_mask_stats(k, v)
    return mask


# =========================================================
# PERSISTENCE
# =========================================================
def persistence(mask):
    pers = {k: v.rolling(step=3, min_periods=3).sum() >= 3 for k, v in mask.items()}
    print("\n📊 3-PENTAD PERSISTENCE")
    for k, v in pers.items():
        print_mask_stats(k, v)
    return pers


# =========================================================
# SEVERITY
# =========================================================
def severity(vars, mask):
    sev = {
        k: np.abs(vars[k].where(mask[k], 0)).rolling(step=3, min_periods=3).mean()
        for k in ["SM", "TP", "VPD", "ET"]
    }
    print("\n📊 SEVERITY")
    for k, v in sev.items():
        print_stats(f"{k} SEVERITY", v)
    return sev


# =========================================================
# GROUNDWATER MODIFIER
# =========================================================
def compute_gw_modifier(GW, fd_step):
    print("\n📊 GW MODIFIER")
    GW = GW.transpose("step", "lat", "lon")
    GW = GW.assign_coords(step=fd_step.values)
    GW_def = normalize(-GW).fillna(0)
    GW_change = GW.diff("step")
    GW_trend_raw = -GW_change
    first_trend = GW_trend_raw.isel(step=0)
    GW_trend_raw = xr.concat([first_trend, GW_trend_raw], dim="step")
    GW_trend_raw = GW_trend_raw.assign_coords(step=fd_step.values).transpose("step", "lat", "lon")
    GW_trend = normalize(GW_trend_raw).fillna(0)
    GW_mod = 1 + ALPHA_GW * (WSD * GW_def + WTR * GW_trend)
    GW_mod = GW_mod.where(np.isfinite(GW_mod), 1.0).transpose("step", "lat", "lon")
    
    print("   GW original :", GW.dims, GW.shape)
    print("   GW_def      :", GW_def.dims, GW_def.shape)
    print("   GW_trend    :", GW_trend.dims, GW_trend.shape)
    print("   GW_mod      :", GW_mod.dims, GW_mod.shape)
    print("   GW steps    :", GW.step.values)
    print("   FDII steps  :", fd_step.values)
    print("\n📊 GW MODIFIER AFTER FDII ALIGNMENT")
    print_stats("GW DEFICIT", GW_def)
    print_stats("GW TREND", GW_trend)
    print_stats("GW MODIFIER", GW_mod)
    return GW_mod, GW_def, GW_trend


# =========================================================
# DROUGHT SEVERITY WITH GW
# =========================================================
def dro_sev(sev, pers, GW_mod):
    print("STEP 9 : COMPUTING DRO-SEV")
    dro = {
        k: ((1 + sev[k]) * GW_mod).where(pers[k], 0)
        for k in sev
    }
    for k, v in dro.items():
        print_stats(f"{k} DRO-SEV", v)
    return dro


# =========================================================
# FDII
# =========================================================
def compute_fdii(fd_int, dro):
    FDII_SM = fd_int["SM"] * dro["SM"]
    FDII_TP = fd_int["TP"] * dro["TP"]
    FDII_ET = fd_int["ET"] * dro["ET"]
    FDII_VPD = fd_int["VPD"] * dro["VPD"]

    FDII_SM = FDII_SM.where(FDII_SM > 0, 0)
    FDII_TP = FDII_TP.where(FDII_TP > 0, 0)
    FDII_ET = FDII_ET.where(FDII_ET > 0, 0)
    FDII_VPD = FDII_VPD.where(FDII_VPD > 0, 0)

    active = (
        (FDII_SM > 0).astype(np.int8) +
        (FDII_TP > 0).astype(np.int8) +
        (FDII_ET > 0).astype(np.int8) +
        (FDII_VPD > 0).astype(np.int8)
    )

    W_SM = W_TP = W_ET = W_VPD = 0.25

    FDII_COMB = W_SM * FDII_SM + W_TP * FDII_TP + W_ET * FDII_ET + W_VPD * FDII_VPD
    FDII_COMB = FDII_COMB.where(active >= 2, 0)
    FD_flag = (FDII_COMB > 0).astype(np.int8)

    print("\n📊 ACTIVE VARIABLE COUNT")
    for n in range(5):
        count = int((active == n).sum())
        print(f"    {n} variables : {count:,}")

    print("\n📊 FDII ACTIVE")
    print_mask_stats("FDII", FD_flag)

    return FDII_SM, FDII_TP, FDII_ET, FDII_VPD, FDII_COMB, FD_flag, active


# =========================================================
# EVENT EXTRACTION
# =========================================================
def extract_events(FD_flag):
    start = (FD_flag == 1) & (FD_flag.shift(step=1, fill_value=0) == 0)
    return start.cumsum(dim="step").where(FD_flag == 1)


# =========================================================
# MFDI METRICS
# =========================================================
def compute_metrics(event_id, FD_flag, FDII, fd_int):
    frequency = event_id.max("step", skipna=True)
    severity_total = FDII.where(FD_flag == 1).sum("step", skipna=True)

    FD_INT_comb = xr.concat(list(fd_int.values()), dim="var").max("var", skipna=True)
    intensity = FD_INT_comb.where(FD_flag == 1).max("step", skipna=True)

    frequency = frequency.fillna(0)
    severity_total = severity_total.fillna(0)
    intensity = intensity.fillna(0)

    MFDI = (normalize(frequency) + normalize(severity_total) + normalize(intensity)) / 3
    MFDI = MFDI.fillna(0)

    return frequency, severity_total, intensity, MFDI


# =========================================================
# PIPELINE
# =========================================================
def run_pipeline(ds, ds_gw, ds_th, prefix, tag):
    vars = get_variables(ds, prefix)
    
    GW = get_gw(ds_gw, prefix)
    print_stats(f"{tag} GW", GW)
    
    fd_int = {
        "SM": compute_fd_int(vars["SM"], -1),
        "TP": compute_fd_int(vars["TP"], -1),
        "VPD": compute_fd_int(vars["VPD"], 1),
        "ET": compute_fd_int(vars["ET"], -1),
    }

    th = get_thresholds(ds, ds_th)
    mask = drought_mask(vars, th)
    pers = persistence(mask)
    sev = severity(vars, mask)

    fd_step = fd_int["SM"].step
    GW_mod, GW_def, GW_trend = compute_gw_modifier(GW, fd_step)

    print_stats(f"{tag} GW MODIFIER", GW_mod)
    dro = dro_sev(sev, pers, GW_mod)

    (FDII_SM, FDII_TP, FDII_ET, FDII_VPD, FDII_COMB,
     FD_flag, active) = compute_fdii(fd_int, dro)

    event_id = extract_events(FD_flag)

    frequency, severity_total, intensity, mfdi = compute_metrics(
        event_id,
        FD_flag,
        FDII_COMB,
        fd_int
    )

    print_stats(f"{tag} FDII", FDII_COMB)

    out = xr.Dataset({
        # FINAL FLASH DROUGHT PRODUCTS
        f"{tag}_FDII_SM": FDII_SM,
        f"{tag}_FDII_TP": FDII_TP,
        f"{tag}_FDII_ET": FDII_ET,
        f"{tag}_FDII_VPD": FDII_VPD,
        f"{tag}_FDII_COMB": FDII_COMB,
        f"{tag}_FD_FLAG": FD_flag.astype(np.int8),
        f"{tag}_MFDI": mfdi,

        # EVENT INFORMATION
        f"{tag}_EVENT_ID": event_id,

        # FINAL METRICS
        f"{tag}_FREQUENCY": frequency,
        f"{tag}_SEVERITY_TOTAL": severity_total,
        f"{tag}_INTENSITY": intensity,

        # FD INTENSIFICATION COMPONENTS
        f"{tag}_FD_INT_SM": fd_int["SM"],
        f"{tag}_FD_INT_TP": fd_int["TP"],
        f"{tag}_FD_INT_ET": fd_int["ET"],
        f"{tag}_FD_INT_VPD": fd_int["VPD"],

        # FINAL DROUGHT SEVERITY WEIGHT
        f"{tag}_DRO_SM": dro["SM"],
        f"{tag}_DRO_TP": dro["TP"],
        f"{tag}_DRO_ET": dro["ET"],
        f"{tag}_DRO_VPD": dro["VPD"],

        # MULTIVARIATE AGREEMENT
        f"{tag}_ACTIVE_COUNT": active.astype(np.int8),
    })

    return out


# =========================================================
# MAIN EXECUTION PIPELINE
# =========================================================
def main():
    """Main processing loop for S2S GLDAS datasets. 

    Handles threshold interpolation, outlier removal, and pipeline execution.
    """
    # 1. LOAD & PREPARE THRESHOLDS
    ds_th = xr.open_dataset(THRES_FILE)
    
    # Match threshold grid once using the first input file as a reference
    with xr.open_dataset(INPUT_FILES[0]) as sample_ds:
        ds_th = ds_th.interp(
            lat=sample_ds.lat, 
            lon=sample_ds.lon
        )

    # 2. ITERATE THROUGH INITIALIZATIONS
    for INPUT_FILE in INPUT_FILES:
        print("\n" + "=" * 50)
        print(f"INITIALIZATION : {os.path.basename(INPUT_FILE)}")
        print("=" * 50)

        # Extract init date from filename
        init_date = os.path.basename(INPUT_FILE).split('_')[-2]

        # Locate corresponding GW file
        gw_pattern = os.path.join(GW_DIR, f"S2S_GLDAS_*_{init_date}_std.nc")
        gw_match = glob.glob(gw_pattern)

        if not gw_match:
            print(f"❌ NO GW FILE FOUND FOR {init_date}")
            continue

        GW_FILE = gw_match[0]

        # 3. LOAD DATASETS
        print("\nSTEP 1 : LOADING DATASETS")
        ds = xr.open_dataset(INPUT_FILE)
        ds_gw = xr.open_dataset(GW_FILE)

        print(f"✅ INPUT LOADED: {os.path.basename(INPUT_FILE)}")
        print(f"✅ GW LOADED:    {os.path.basename(GW_FILE)}")
        print("✅ THRESHOLDS INTERPOLATED")

        # 4. CLEAN GW OUTLIERS
        print("\n⚠️  CLEANING GW OUTLIERS")
        for var in ds_gw.data_vars:
            before_min, before_max = float(ds_gw[var].min()), float(ds_gw[var].max())

            # Apply threshold mask (-5 to 5)
            ds_gw[var] = ds_gw[var].where((ds_gw[var] > -5) & (ds_gw[var] < 5))

            after_min, after_max = float(ds_gw[var].min()), float(ds_gw[var].max())
            bad_pixels = np.isnan(ds_gw[var]).sum().values

            print(f"🔹 {var:10} | Range: [{before_min:7.4f}, {before_max:7.4f}] -> [{after_min:7.4f}, {after_max:7.4f}] | Cleaned: {bad_pixels}")

        # 5. RUN PIPELINES (S2S & ERA)
        out_s2s = run_pipeline(ds, ds_gw, ds_th, "s2s", "S2S")
        out_era = run_pipeline(ds, ds_gw, ds_th, "obs", "ERA")

        # Merge results
        out = xr.merge([out_s2s, out_era], compat="override")

        # 6. EXPORT & CLOSE
        out_file = os.path.join(
            OUT_DIR, 
            os.path.basename(INPUT_FILE).replace("_std", "_FDII_GW")
        )

        encoding = {var: dict(zlib=True, complevel=6) for var in out.data_vars}
        out.to_netcdf(out_file, encoding=encoding)
        print(f"\n✅ SAVED : {out_file}")

        # Cleanup memory
        ds.close()
        ds_gw.close()
        out.close()

    ds_th.close()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()