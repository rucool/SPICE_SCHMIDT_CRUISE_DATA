# SPICE_SCHMIDT_CRUISE_DATA

This project is a dedicated investigation into how salt finger-driven double-diffusive mixing influences nutrient supply and ecosystem productivity in the western equatorial North Atlantic. This region is a known hotspot for "thermohaline staircases"-oceanic structures formed by salt fingering where warm, salty subtropical waters overlie cooler, fresher Antarctic Intermediate Waters. While mechanical turbulence is typically considered the primary driver of nutrient upwelling in the open ocean, double-diffusive processes like salt fingering can transfer dissolved constituents more efficiently, potentially supplying a significant portion of "new" nitrogen to the surface. Accordingly, the science party is referring to this cruise as Project SPICE (Salt finger Processes Influence on Carbon and Ecosystem dynamics).

This repo holds the satellite and platform-tracking figure pipeline supporting the ru29 glider mission and the R/V Falkor (too) / Schmidt Ocean Institute cruise for this project - the core Python logic only. Cron wrappers, absolute deployment paths, and generated output (figures, downloaded NetCDFs, logs) live on the production server and are not tracked here.

## Scripts

- **`cmems_download.py`** — pulls gridded satellite products from Copernicus Marine (SSH/SLA, SST, chlorophyll, sea surface salinity + density) into `cmems_data/<product>/`. Shared by the two plotting scripts below.
- **`SPICE_CMEMS_SAT.py`** — generates one map per variable/day, overlaying every enabled platform's track (glider, ship) from `PLATFORMS` in the script. Copies output to the web folder configured in `config.py`.
- **`cmems_sla_adt.py`** — generates a KMZ (SSH/SLA) for viewing in Google Earth.
- **`ru29_staircase.py`** — pulls ru29 glider profiles from the Rutgers glider ERDDAP, detects thermohaline staircases, and writes both the staircase figures and the glider's position track (`ru29_latest_track.csv`) that `SPICE_CMEMS_SAT.py` overlays.
- **`get_falkor_position.py`** — fetches R/V Falkor (too)'s position from FSU/COAPS SAMOS (public THREDDS/OPeNDAP feed, no API key required) and writes a rolling track (`falkor_track.csv`).

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

**3. Copernicus Marine credentials**

Run once, interactively:
```bash
python -c "import copernicusmarine; copernicusmarine.login()"
```
Credentials are cached locally by the `copernicusmarine` package itself; nothing gets stored in this repo.

## Adding or toggling a platform

`SPICE_CMEMS_SAT.py`'s `PLATFORMS` list controls what gets overlaid on the maps:
```python
PLATFORMS = [
    {"name": "ru29", "csv": "ru29_latest_track.csv", "marker": "*", "color": "gold", "markersize": 10, "enabled": True},
    {"name": "Falkor (too)", "csv": "falkor_track.csv", "marker": "^", "color": "magenta", "markersize": 8, "enabled": False},
]
```
Each entry needs a `time,lat,lon` CSV written by its own fetch script. Flip `"enabled"` to turn a platform's overlay on/off without removing it; add a new dict (with a distinct `marker`/`color`) for additional platforms.
