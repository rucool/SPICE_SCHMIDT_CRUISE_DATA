import os
import pandas as pd
import xarray as xr
import copernicusmarine


# Shared bounding box for every figure/kmz script: [lon_min, lon_max, lat_min, lat_max]
TROP_WTRN_ATL_EXTENT = [-63, -40.75, 4, 19]

CMEMS_BASE_DIR = "./cmems_data"

# product registry: everything pulled via Copernicus Marine now.
# AOML's AVISO SSH mirror died 2026-01-22, and NOAA CoastWatch (sst/ocean_color/ssh
# alternatives) is blocked by their WAF - Copernicus also auto-clamps out-of-range
# date requests (with a warning) instead of 404ing, which sidesteps NRT latency gaps.
SATELLITE_PRODUCTS = {
    "aviso_ssh": {
        "source": "copernicus",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        "variables": ["adt", "sla", "ugos", "vgos", "ugosa", "vgosa"],
        "plot_vars": ["sla"],
    },
    # OSTIA Global L4 SST analysis - gap-filled (no cloud holes), GHRSST L4
    "sst": {
        "source": "copernicus",
        "dataset_id": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "variables": ["analysed_sst", "analysis_error", "mask"],
        "plot_vars": ["analysed_sst"],
    },
    # Copernicus-GlobColour multi-sensor chlorophyll, L4 gap-free daily (interpolated)
    "ocean_color": {
        "source": "copernicus",
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
        "variables": ["CHL"],
        "plot_vars": ["CHL"],
    },
    # Multi-obs Sea Surface Salinity + density, L4 NRT
    # (product MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013). Per the CMEMS catalogue
    # this dataset only updates weekly (Tuesdays 16:00 UTC), not daily like the
    # other products - so most daily cron runs will just re-plot the same
    # underlying day until the next Tuesday update lands. max_lookback_days is
    # generous (>2 weekly cycles) so one delayed/missed update doesn't hard-fail.
    "sss": {
        "source": "copernicus",
        "dataset_id": "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
        "variables": ["sos", "dos"],
        "plot_vars": ["sos", "dos"],
        "max_lookback_days": 18,
    },
    # TODO: PACE OCI - not on ERDDAP or CMEMS, would need NASA Earthdata/earthaccess
    "ocean_color_pace": {
        "source": "todo",
        "plot_vars": [],
    },
}


# copernicusmarine.login()
# ONLY NEED TO RUN THIS ONCE - credentials get cached locally after first login.
# Every later copernicusmarine.subset()/open_dataset() call (including from cron)
# reuses that cache automatically.


def fetch_copernicus(dataset_id, variables, date, bbox, output_dir, n_days=1, max_lookback_days=5):
    """Pull n_days of gridded data ending at `date` (or the latest available day
    at/before it) from a Copernicus Marine dataset.

    date: anything pd.Timestamp() accepts (e.g. a datetime, or "2026-06-25").
    bbox: [lon_min, lon_max, lat_min, lat_max], e.g. TROP_WTRN_ATL_EXTENT.
    n_days: how many days of history to pull, ending at the latest available day.
    NRT products lag "today" by varying amounts, and copernicusmarine only
    auto-clamps when the request partially overlaps available data - a fully
    out-of-range request (e.g. "today" on a dataset that ends yesterday) hard
    errors instead - so this walks back a day at a time until something works.
    """
    date = pd.Timestamp(date)
    lon_min, lon_max, lat_min, lat_max = bbox

    last_error = None
    for days_back in range(max_lookback_days + 1):
        end_date = date - pd.Timedelta(days=days_back)
        start_date = end_date - pd.Timedelta(days=n_days - 1)
        out_name = f"{dataset_id}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=end_date.strftime("%Y-%m-%dT23:59:59"),
                output_filename=out_name,
                output_directory=output_dir,
                overwrite=True,
            )
            ds = xr.open_dataset(os.path.join(output_dir, f"{out_name}.nc"))
            ds.attrs["fetch_date"] = end_date.strftime("%Y-%m-%d")
            return ds
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No data available for {dataset_id} within {max_lookback_days} days of {date:%Y-%m-%d}"
    ) from last_error


# pull the last N_DAYS_BACK days (ending at today, or the latest available day)
# for every registered product. Override via env for one-off backfills
# (e.g. cleanup_and_rerun.sh widens this to cover a multi-day rebuild)
# without changing the normal daily cron window.
plot_date = pd.Timestamp.now(tz="UTC").normalize()
N_DAYS_BACK = int(os.environ.get("N_DAYS_BACK", 5))

daily_data = {}
for product_name, info in SATELLITE_PRODUCTS.items():
    if info["source"] == "todo":
        continue
    daily_data[product_name] = fetch_copernicus(
        info["dataset_id"],
        info["variables"],
        plot_date,
        TROP_WTRN_ATL_EXTENT,
        output_dir=os.path.join(CMEMS_BASE_DIR, product_name),
        n_days=N_DAYS_BACK,
        max_lookback_days=info.get("max_lookback_days", 5),
    )
