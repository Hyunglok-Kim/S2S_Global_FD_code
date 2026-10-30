# Global Flash Drought Intensity Index (FDII) and Multivariate Flash Drought Index (MFDI)

A Python framework for generating ERA5-based pentad climatological
thresholds, identifying flash drought conditions, quantifying flash
drought intensity, and incorporating groundwater conditions into a
multivariate flash drought assessment.

The framework uses four hydroclimatic variables:

-   **Soil moisture (SM)**
-   **Total precipitation (TP)**
-   **Evapotranspiration (ET)**, represented by latent heat flux (LHF)
-   **Vapor pressure deficit (VPD)**

Standardized groundwater storage and depletion trends are incorporated
through a groundwater modifier.

------------------------------------------------------------------------

## 1. Repository structure

Organize the repository as follows:

``` text
FDII-MFDI/
├── fdii_generation/
│   └── Flash_Drought_Intensity_Index_Code.py
├── threshold_generation/
│   └── Threshold_Percentiles_ERA5_Climatology.py
├── data/
│   ├── era5_pentad/
│   │   ├── sm_era5_pentad.nc
│   │   ├── tp_era5_pentad.nc
│   │   ├── t2m_era5_pentad.nc
│   │   ├── d2m_era5_pentad.nc
│   │   └── lhf_era5_pentad.nc
│   ├── s2s_standardized/
│   │   └── *_std.nc
│   ├── groundwater_standardized/
│   │   └── S2S_GLDAS_*_YYYYMMDD_std.nc
│   └── thresholds/
│       └── ERA5_thresholds_climatology.nc
├── outputs/
│   └── FDII_MFDI/
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` and `outputs/` directories contain local input and generated
NetCDF files. Large datasets should generally be stored outside the
public Git repository.

**Path configuration:** The FDII script derives the project root from
its own location and assumes it is stored in `fdii_generation/`. The
threshold-generation script currently uses paths relative to the working
directory; run it from the repository root.

------------------------------------------------------------------------

## 2. Software requirements

-   Python 3.10 or later
-   NumPy
-   Xarray
-   Dask
-   Dask Distributed
-   NetCDF4

### Create an environment

``` bash
conda create -n fdii_env python=3.10 -y
conda activate fdii_env
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

``` text
numpy
xarray
dask[distributed]
netCDF4
```

For reproducible production runs, record and pin the package versions
used in the experiment.

------------------------------------------------------------------------

## 3. Input data requirements

The workflow requires three categories of input data.

### 3.1 ERA5 pentad datasets

Place the following files in `data/era5_pentad/`:

  Variable   Expected filename      Description
  ---------- ---------------------- ---------------------------
  SM         `sm_era5_pentad.nc`    Soil moisture
  TP         `tp_era5_pentad.nc`    Total precipitation
  T2M        `t2m_era5_pentad.nc`   2-m air temperature
  TD         `d2m_era5_pentad.nc`   2-m dew-point temperature
  ET         `lhf_era5_pentad.nc`   Latent heat flux

The threshold-generation script expects compatible datasets with
dimensions:

``` text
year × pentad × lat × lon
```

T2M and TD must contain physical temperatures in Kelvin because the
script calculates physical VPD from these variables.

### 3.2 Standardized S2S and observation datasets

Place the standardized forecast and observation NetCDF files in:

``` text
data/s2s_standardized/
```

Each input file must contain these variables:

  Variable                 S2S field       Observation field
  ------------------------ --------------- -------------------
  Soil moisture            `sm_s2s_std`    `sm_obs_std`
  Total precipitation      `tp_s2s_std`    `tp_obs_std`
  Latent heat flux         `lhf_s2s_std`   `lhf_obs_std`
  Vapor pressure deficit   `vpd_s2s_std`   `vpd_obs_std`

The FDII script also expects the relevant temporal and spatial
coordinates, including:

``` text
step
valid_time
lat
lon
```

**VPD preprocessing requirement:** Calculate physical VPD from the
original, non-standardized T2M and TD values before standardizing the
variables. Include `vpd_s2s_std` and `vpd_obs_std` in the standardized
input files. The FDII script reads these fields directly; it does not
calculate VPD from standardized temperature variables.

The standardization applied to the S2S/observation variables must be
statistically compatible with the ERA5-derived threshold reference.
Document the reference period and standardization method used.

### 3.3 Standardized groundwater datasets

Place groundwater NetCDF files in:

``` text
data/groundwater_standardized/
```

The FDII script searches for files matching:

``` text
S2S_GLDAS_*_YYYYMMDD_std.nc
```

The groundwater file must contain standardized groundwater variables for
both forecast and observation processing. The script supports one of the
following variable-name pairs:

``` text
gw_s2s_std     gw_obs_std
gws_s2s_std    gws_obs_std
tws_s2s_std    tws_obs_std
```

The groundwater data must have compatible `step`, `lat`, and `lon`
dimensions with the corresponding hydroclimatic input data. Confirm that
the groundwater and FDII forecast leads correspond before processing.

------------------------------------------------------------------------

## 4. Processing workflow

The framework has two sequential stages.

### Stage 1 --- Generate ERA5 climatological thresholds

**Script:**
`threshold_generation/Threshold_Percentiles_ERA5_Climatology.py`

The threshold-generation script:

1.  Loads ERA5 pentad datasets.
2.  Calculates physical VPD from T2M and TD.
3.  Standardizes each variable using the full-period climatology across
    years for each pentad and grid cell.
4.  Calculates variable-specific percentile thresholds.
5.  Saves the threshold dataset as a NetCDF file.

The standardization is:

\[ Z\_{y,p} = `\frac{X_{y,p}-\mu_p}{\sigma_p}`{=tex} \]

where (X\_{y,p}) is the variable for year (y) and pentad (p), and
(`\mu`{=tex}\_p) and (`\sigma`{=tex}\_p) are the full-period mean and
standard deviation for that pentad and grid cell.

The threshold definitions are:

  Variable                Percentile Output field
  --------------------- ------------ --------------
  Soil moisture                 20th `TH_SM`
  Total precipitation           20th `TH_TP`
  Latent heat flux              20th `TH_ET`
  VPD                           80th `TH_VPD`

The output is written to:

``` text
data/thresholds/ERA5_thresholds_climatology.nc
```

### Stage 2 --- Generate FDII and MFDI products

**Script:** `fdii_generation/Flash_Drought_Intensity_Index_Code.py`

The FDII script:

1.  Loads standardized S2S and observation variables.
2.  Loads the corresponding standardized groundwater dataset.
3.  Interpolates the ERA5 thresholds to the spatial grid of the first
    input file.
4.  Identifies drought conditions using the variable-specific
    thresholds.
5.  Evaluates three-pentad persistence.
6.  Calculates flash drought intensification and severity components.
7.  Calculates groundwater storage deficit and depletion trend.
8.  Applies the groundwater modifier.
9.  Generates individual and combined FDII components.
10. Extracts flash drought events and calculates frequency, severity,
    intensity, and MFDI.
11. Writes the results to NetCDF files.

### Groundwater modifier

The implemented groundwater modifier is:

\[ GW\_{`\mathrm{MOD}`{=tex}} = 1+`\alpha`{=tex}*{`\mathrm{GW}`{=tex}}
`\left`{=tex}( W*{`\mathrm{SD}`{=tex}}GW\_{`\mathrm{SD}`{=tex}}+
W\_{`\mathrm{TR}`{=tex}}GW\_{`\mathrm{TR}`{=tex}} `\right`{=tex}) \]

where:

-   (GW\_{`\mathrm{SD}`{=tex}}) is the normalized groundwater storage
    deficit.
-   (GW\_{`\mathrm{TR}`{=tex}}) is the normalized groundwater depletion
    trend.
-   (`\alpha`{=tex}\_{`\mathrm{GW}`{=tex}}) is the groundwater influence
    coefficient.
-   (W\_{`\mathrm{SD}`{=tex}}) is the groundwater storage deficit
    weight.
-   (W\_{`\mathrm{TR}`{=tex}}) is the groundwater trend weight.

Default values in the script:

  Parameter      Value
  ------------ -------
  `ALPHA_GW`       0.2
  `WSD`            0.6
  `WTR`            0.4

### Combined FDII

The script combines the four FDII components using equal weights:

\[ FDII\_{`\mathrm{COMB}`{=tex}} = 0.25FDII\_{`\mathrm{SM}`{=tex}}+
0.25FDII\_{`\mathrm{TP}`{=tex}}+ 0.25FDII\_{`\mathrm{ET}`{=tex}}+
0.25FDII\_{`\mathrm{VPD}`{=tex}} \]

The combined FDII is retained when at least two individual components
are active.

### MFDI

The implemented Multivariate Flash Drought Index combines normalized
event frequency, cumulative severity, and intensity:

\[ MFDI = `\frac{N(F)+N(S)+N(I)}{3}`{=tex} \]

where (N) denotes the min--max normalization implemented in the script.

------------------------------------------------------------------------

## 5. Running the framework

Run the commands below from the repository root.

### Step 1 --- Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FDII-MFDI
```

### Step 2 --- Activate the environment and install dependencies

``` bash
conda activate fdii_env
pip install -r requirements.txt
```

### Step 3 --- Prepare the input datasets

Place the ERA5 pentad, standardized S2S/observation, and standardized
groundwater NetCDF files in the directories described in Section 3.

Check that filenames, variable names, dimensions, and coordinates match
the requirements.

### Step 4 --- Generate ERA5 thresholds

``` bash
python threshold_generation/Threshold_Percentiles_ERA5_Climatology.py
```

Expected output:

``` text
data/thresholds/ERA5_thresholds_climatology.nc
```

### Step 5 --- Generate FDII and MFDI products

``` bash
python fdii_generation/Flash_Drought_Intensity_Index_Code.py
```

The script writes one output NetCDF file per processed initialization
to:

``` text
outputs/FDII_MFDI/
```

------------------------------------------------------------------------

## 6. Configuration

The main parameters are defined directly in the Python scripts.

### Threshold-generation script

  ----------------------------------------------------------------------------------------------------
  Parameter               Current setting                                      Purpose
  ----------------------- ---------------------------------------------------- -----------------------
  `BASE_PATH`             `./data/era5_pentad`                                 ERA5 input directory

  `FILE_MAP`              Defined in script                                    Input NetCDF filenames

  `OUT_FILE`              `./data/thresholds/ERA5_thresholds_climatology.nc`   Threshold output

  Dask workers            32                                                   Number of workers

  Threads per worker      2                                                    Threads assigned to
                                                                               each worker

  Worker memory limit     16 GB                                                Per-worker memory limit
  ----------------------------------------------------------------------------------------------------

Adjust Dask resources to match the available machine. The configured
worker count and memory limit can require substantial resources.

### FDII-generation script

  --------------------------------------------------------------------------------------------------
  Parameter               Current setting                                    Purpose
  ----------------------- -------------------------------------------------- -----------------------
  `INPUT_DIR`             `data/s2s_standardized`                            Standardized
                                                                             hydroclimatic inputs

  `GW_DIR`                `data/groundwater_standardized`                    Standardized
                                                                             groundwater inputs

  `THRES_FILE`            `data/thresholds/ERA5_thresholds_climatology.nc`   ERA5 thresholds

  `OUT_DIR`               `outputs/FDII_MFDI`                                Output directory

  `N_CALIBRATION`         208                                                Number of selected
                                                                             initialization files

  `ALPHA_GW`              0.2                                                Groundwater influence
                                                                             coefficient

  `WSD`                   0.6                                                Groundwater deficit
                                                                             weight

  `WTR`                   0.4                                                Groundwater trend
                                                                             weight
  --------------------------------------------------------------------------------------------------

**Initialization selection:** The current script sorts the input
filenames and selects the first 208 files. Confirm that filename sorting
corresponds to the intended calibration period. The script does not
automatically process a separate validation period.

------------------------------------------------------------------------

## 7. Output products

The FDII-generation script produces one NetCDF file per processed
initialization. The output filename is generated by replacing `_std` in
the input filename with `_FDII_GW`.

The output dataset contains products for both S2S forecasts and
observations. Variable names use the `S2S` and `ERA` prefixes,
respectively.

  Output field suffix   Description
  --------------------- ----------------------------------------
  `FDII_SM`             Soil moisture FDII component
  `FDII_TP`             Precipitation FDII component
  `FDII_ET`             Evapotranspiration FDII component
  `FDII_VPD`            VPD FDII component
  `FDII_COMB`           Combined flash drought intensity index
  `FD_FLAG`             Flash drought occurrence flag
  `EVENT_ID`            Flash drought event identifier
  `FREQUENCY`           Flash drought event frequency
  `SEVERITY_TOTAL`      Cumulative flash drought severity
  `INTENSITY`           Maximum flash drought intensification
  `MFDI`                Multivariate flash drought index
  `FD_INT_SM`           Soil moisture intensification
  `FD_INT_TP`           Precipitation intensification
  `FD_INT_ET`           Evapotranspiration intensification
  `FD_INT_VPD`          VPD intensification
  `DRO_SM`              Groundwater-adjusted SM severity
  `DRO_TP`              Groundwater-adjusted TP severity
  `DRO_ET`              Groundwater-adjusted ET severity
  `DRO_VPD`             Groundwater-adjusted VPD severity
  `ACTIVE_COUNT`        Number of active FDII components

For example, `S2S_FDII_COMB` represents the combined forecast FDII,
while `ERA_FDII_COMB` represents the observation/reference product.

------------------------------------------------------------------------

## 8. Data and computational considerations

1.  **VPD consistency:** Calculate physical VPD from Kelvin T2M and TD
    before standardization. Supply the resulting standardized VPD fields
    to the FDII script.
2.  **Threshold consistency:** Use the ERA5 threshold file generated by
    Stage 1. The standardized input variables and thresholds must use
    statistically compatible reference conventions.
3.  **Spatial grid:** The FDII script interpolates thresholds to the
    grid of the first input file. Ensure subsequent input files use the
    same grid.
4.  **Temporal alignment:** Verify that groundwater and hydroclimatic
    `step` coordinates represent matching forecast leads. The current
    implementation assigns the groundwater step labels to the FDII
    steps.
5.  **Input matching:** Ensure each S2S initialization has a
    corresponding groundwater file matching the expected filename
    pattern.
6.  **Missing data:** Inspect input files for missing values and valid
    data coverage before processing.
7.  **Computational resources:** Adjust Dask workers and memory limits
    to the resources available on the machine.
8.  **Calibration and validation:** The current FDII script processes
    the first 208 sorted input files only. Modify the selection logic
    when processing other periods.

------------------------------------------------------------------------

## 9. Troubleshooting

  -----------------------------------------------------------------------
  Issue                   Possible cause          Recommended check
  ----------------------- ----------------------- -----------------------
  Threshold file not      Incorrect path or       Verify `OUT_FILE` and
  found                   working directory       run from the repository
                                                  root

  Missing VPD variable    VPD was not generated   Check `vpd_s2s_std` and
                          during preprocessing    `vpd_obs_std`

  Groundwater file not    Filename does not match Check initialization
  found                   the search pattern      date and groundwater
                                                  filenames

  Missing variable error  NetCDF variable name    Inspect the input data
                          differs from the        variables
                          expected name           

  Dimension mismatch      Incompatible `step`,    Check dimensions and
                          `lat`, or `lon`         coordinate alignment
                          coordinates             

  Dask memory error       Insufficient available  Reduce worker count or
                          memory                  memory allocation

  No output files         No matching input or    Verify input
                          groundwater files       directories and
                                                  filename patterns
  -----------------------------------------------------------------------

To inspect a NetCDF file:

``` python
import xarray as xr

ds = xr.open_dataset("path/to/input_file.nc")

print(ds)
print(ds.data_vars)
print(ds.dims)
print(ds.coords)

ds.close()
```

------------------------------------------------------------------------

## 10. Reproducibility

For reproducible experiments, record:

-   Input dataset versions and temporal coverage.
-   ERA5 climatological reference period.
-   Standardization methodology.
-   Threshold percentile definitions.
-   Groundwater modifier parameters.
-   Calibration and validation periods.
-   Python and package versions.
-   Processing configuration and output filenames.

The current implementation uses full-period climatological
standardization and pentad-specific percentile thresholds. Changes to
the reference period, standardization procedure, or groundwater
parameters may affect the resulting FDII and MFDI products.

------------------------------------------------------------------------

## 11. Citation and license

If you use this framework in a scientific publication, cite the
associated research article or repository release once the formal
citation information is available.

Add the selected open-source license to the repository before public
distribution. For questions, bug reports, and contributions, use the
GitHub Issues and Pull Requests sections.
