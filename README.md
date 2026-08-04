# SPICE_SCHMIDT_CRUISE_DATA

This project is a dedicated investigation into how salt finger-driven double-diffusive mixing influences nutrient supply and ecosystem productivity in the western equatorial North Atlantic. This region is a known hotspot for "thermohaline staircases"-oceanic structures formed by salt fingering where warm, salty subtropical waters overlie cooler, fresher Antarctic Intermediate Waters. While mechanical turbulence is typically considered the primary driver of nutrient upwelling in the open ocean, double-diffusive processes like salt fingering can transfer dissolved constituents more efficiently, potentially supplying a significant portion of "new" nitrogen to the surface. Accordingly, the science party is referring to this cruise as Project SPICE (Salt finger Processes Influence on Carbon and Ecosystem dynamics).

This repo holds the satellite and platform-tracking figure pipeline supporting the ru29 glider mission and the R/V Falkor (too) / Schmidt Ocean Institute cruise for this project - the core Python logic only. Cron wrappers, absolute deployment paths, and generated output (figures, downloaded NetCDFs, logs) live on the production server and are not tracked here.

## Scripts

- **`cmems_download.py`** — pulls gridded satellite products from Copernicus Marine (SSH/SLA, SST, chlorophyll, sea surface salinity + density, Sargassum floating algae index) into `cmems_data/<product>/`. Shared by the two plotting scripts below.
- **`SPICE_CMEMS_SAT.py`** — generates one map per variable/day, overlaying every enabled platform's track (glider, ship) from `PLATFORMS` in the script. Copies output to the web folder configured in `config.py`. `NFAI_BINARY_MODE` toggles the Sargassum index between a continuous gradient and a simple detected/not-detected view.
- **`cmems_sla_adt.py`** — generates a KMZ (SSH/SLA) for viewing in Google Earth.
- **`ru29_staircase.py`** — pulls ru29 glider profiles from the Rutgers glider ERDDAP, detects thermohaline staircases, and writes both the staircase figures and the glider's position track (`ru29_latest_track.csv`) that `SPICE_CMEMS_SAT.py` overlays. Plots against longitude (not distance-along-track) while ru29 is within its fixed zonal survey latitude band.
- **`gliders_staircase.py`** — generalized version of `ru29_staircase.py` for any other glider deployment (e.g. the VOTO glider): takes a `deployment` ID the same way the real-time plotting scripts below do, reusing the same staircase-detection/hovmoller pipeline. Use `ru29_staircase.py` specifically for ru29 (it has ru29's own fixed zonal-survey plotting variant); use this one for everything else.
- **`get_falkor_position.py`** — fetches R/V Falkor (too)'s position from FSU/COAPS SAMOS (public THREDDS/OPeNDAP feed, no API key required) and writes a rolling track (`falkor_track.csv`).
- **`eddy_trajectory_download.py`** — fetches AVISO+ near-real-time mesoscale eddy trajectories (anticyclonic + cyclonic; the atlas covers the whole globe back to 2018, subset here to the SPICE bbox), plus a matching-date SLA snapshot for background context. **Requires AVISO+ credentials in `~/.netrc`** (see Setup below) - this data is not part of Copernicus Marine and needs its own login. Writes `eddy_<polarity>_latest.nc` and `sla_background.nc` into `cmems_data/eddy_trajectory/`. Decoupled from plotting and meant to run once/day (not the 3-hourly cycle) - AVISO's files are the entire growing atlas (~1.6-1.7GB each), not a rolling window, so re-fetching more often wastes real bandwidth for no new data.
- **`eddy_trajectory_plot.py`** — plots eddy positions and recent tracks (last `TAIL_DAYS`), reading back whatever `eddy_trajectory_download.py` last wrote (no network access itself). `SLA_BACKGROUND_MODE` toggles between a plain view (unique color per eddy) and an SLA-background view (eddies colored by whether they're still active or already ended within the tail window, shape still distinguishing anticyclonic/cyclonic). Overlays platform tracks the same way as `SPICE_CMEMS_SAT.py`.
- **`eddy_12N_forecast.py`** — summarizes currently-active eddies (also reading back whatever `eddy_trajectory_download.py` last wrote, no network access itself) into a plain-text metrics table: radius, translation speed, forecasted arrival time at the 12N survey line, and both a real-time and a map-matched (date-lagged) distance to each enabled platform. Which bbox counts as "active," and which columns appear in the table, are entirely config-driven via `configs/eddy_12N_forecast.yml` - see that file's comments for the full set of available fields. Writes into the same dated `eddy_trajectory/` output folder as `eddy_trajectory_plot.py`'s map, overwritten in place on every 3-hourly cron run (a new file only appears when the eddy data's own date changes, not every run).
- **`plot_glider_xsection_rt.py`** — pulls a real-time glider deployment from ERDDAP and plots time/depth cross-sections (scatter, colored by variable) for each variable in the config, for the synoptic record or the last 24/48 hours.
- **`plot_glider_profiles_rt.py`** — same real-time glider data source, plotted instead as depth profiles colored by profile time, for the synoptic record or the last 24/48 hours.
- **`plot_glider_ts_rt.py`** — same real-time glider data source, generates a T-S diagram colored by depth, for the synoptic record or the last 24/48 hours.

Both glider plotting scripts take a `deployment` ID (e.g. `ru29-20260623T2102-profile-sci-rt`), read per-variable colormap/title/axis-limit settings from `configs/plot_vars_glider.yml` (or a deployment-specific override `configs/plot_vars_glider_<deployment>.yml`), and share fetch/plot helpers in `functions/common.py` and `functions/plotting.py`.

## Setup

**1. Environment**
```bash
conda env create -f environment.yml -n spice_data
conda activate spice_data
```

**2. Local configuration**

Server-specific paths (currently just the web output folder) are kept out of the tracked scripts:
```bash
cp config.py.example config.py
# edit config.py with your actual WEB_FOLDER path
```
`config.py` is gitignored — never commit it.

**3. EEZ boundary shapefile**

`SPICE_CMEMS_SAT.py` and `eddy_trajectory_plot.py` overlay gray EEZ (Exclusive Economic Zone) boundary lines on every map. The shapefile itself isn't tracked in this repo (not a `.py` file). Download the World EEZ v11 boundaries dataset from [marineregions.org](https://www.marineregions.org/), and place the extracted folder as `World_EEZ_v11_20191118/` in the same directory as the `.py` scripts, so `eez_boundaries_v11.shp` (plus its `.dbf`/`.shx`/`.prj`/`.cpg` sidecar files) resolves at:
```
World_EEZ_v11_20191118/eez_boundaries_v11.shp
```
If the folder or shapefile is missing, both scripts print a warning and skip the overlay rather than failing the run.

**4. Copernicus Marine credentials**

Run once, interactively:
```bash
python -c "import copernicusmarine; copernicusmarine.login()"
```
Credentials are cached locally by the `copernicusmarine` package itself; nothing gets stored in this repo.

**5. AVISO+ credentials (for eddy trajectory data)**

`eddy_trajectory_download.py` pulls from AVISO+, not Copernicus Marine, and needs its own free account (register at aviso.altimetry.fr) with credentials in `~/.netrc`:
```
machine tds-odatis.aviso.altimetry.fr
    login <your AVISO+ username>
    password <your AVISO+ password>
```
```bash
chmod 600 ~/.netrc
```
Most OPeNDAP/netCDF backends (and the plain HTTP client this script uses) refuse to use a `.netrc` file with looser permissions.

## Adding or toggling a platform

`SPICE_CMEMS_SAT.py`'s `PLATFORMS` list controls what gets overlaid on the maps:
```python
PLATFORMS = [
    {"name": "ru29", "csv": "ru29_latest_track.csv", "marker": "*", "color": "gold", "markersize": 10, "enabled": True},
    {"name": "Falkor (too)", "csv": "falkor_track.csv", "marker": "^", "color": "magenta", "markersize": 8, "enabled": False},
]
```
Each entry needs a `time,lat,lon` CSV written by its own fetch script. Flip `"enabled"` to turn a platform's overlay on/off without removing it; add a new dict (with a distinct `marker`/`color`) for additional platforms.

`eddy_trajectory_plot.py` and `eddy_12N_forecast.py` each keep their own copy of this list (not imported - neither of the other two scripts is safely importable as a module) with just `name`/`csv`/`enabled`, no marker/color. Keep `"enabled"` in sync by hand across all three when toggling a platform - e.g. flipping Falkor (too) on for the real cruise means updating it in `SPICE_CMEMS_SAT.py`, `eddy_trajectory_plot.py`, and `eddy_12N_forecast.py`. In `eddy_12N_forecast.py` specifically, a disabled platform's distance columns simply aren't available to select in `configs/eddy_12N_forecast.yml` - naming one anyway is skipped with a run-time warning rather than an error, and starts working the moment it's re-enabled.
