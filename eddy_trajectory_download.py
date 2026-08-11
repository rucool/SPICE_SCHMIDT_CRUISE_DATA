#!/usr/bin/env python
"""
Fetches AVISO+ near-real-time eddy trajectory data (anticyclonic + cyclonic)
for the SPICE survey bbox and a recent rolling time window, and saves the
subset locally for eddy_trajectory_plot.py to read.

Auth/access: AVISO+ requires HTTP Basic Auth (confirmed - an unauthenticated
request gets a 401 Unauthorized from the THREDDS server). This reads the
same ~/.netrc entry curl/OPeNDAP clients use (standard, no separate secret
needed here or on the command line):

    machine tds-odatis.aviso.altimetry.fr
        login <your AVISO+ username>
        password <your AVISO+ password>

chmod 600 ~/.netrc - most OPeNDAP/netCDF backends refuse to use a .netrc
file with looser permissions.

Deliberately NOT using OPeNDAP/dodsC (xr.open_dataset on a remote dodsC
URL): netCDF-C's DAP2 client generates a malformed constraint expression
with a stray unencoded space (e.g. "...time[0] ") when probing the time
variable while opening this dataset, which this server's Tomcat 10.1
frontend rejects outright as an RFC 7230 request-line violation - confirmed
via direct testing (curl -n against dodsC succeeds, netCDF4/xarray against
the same URL does not). Adding a .dodsrc HTTP.NETRC entry fixes auth but
not this - the constraint-expression bug is inside netCDF-C itself. Rather
than add a new dependency (e.g. pydap) to work around a compiled library
bug, this instead downloads the whole file via THREDDS's plain HTTP
"fileServer" service (no DAP, no constraint expressions, so the bug can't
trigger) using only the stdlib (netrc + urllib), then opens that local file
normally.
"""
import argparse
import base64
import glob
import json
import netrc
import os
import re
import shutil
import urllib.error
import urllib.request

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr

# Shared bounding box - same as the rest of the pipeline (SPICE_CMEMS_SAT.py,
# cmems_download.py)
TROP_WTRN_ATL_EXTENT = [-63, -40.75, 4, 19]

AVISO_HOST = "tds-odatis.aviso.altimetry.fr"
FILESERVER_BASE = f"https://{AVISO_HOST}/thredds/fileServer/dataset-duacs-nrt-value-added-eddy-trajectory"
CATALOG_URL = f"https://{AVISO_HOST}/thredds/catalog/dataset-duacs-nrt-value-added-eddy-trajectory/catalog.xml"

# Same dataset cmems_download.py's aviso_ssh product uses - duplicated here
# (not imported - cmems_download.py runs its own full multi-product
# download loop as an import side effect) so the eddy plot can show the
# actual SLA field the eddy positions were derived from, not just whatever
# SLA happens to be freshest today.
SLA_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"

ROLLING_WINDOW_DAYS = 21  # how much eddy-track history to keep per fetch


def discover_eddy_filenames(catalog_url=CATALOG_URL):
    """AVISO republishes these files with a rolling end-date baked into the
    filename (e.g. ..._20180101_20260713.nc, extended as new data lands), so
    the exact name can't be hardcoded - look it up from the public THREDDS
    catalog (no auth needed, unlike the actual data access) each run.
    """
    with urllib.request.urlopen(catalog_url, timeout=30) as resp:
        catalog_xml = resp.read().decode("utf-8")

    filenames = {}
    for polarity in ("anticyclonic", "cyclonic"):
        # The catalog can list more than one entry per polarity, e.g. during
        # AVISO's nightly file-rotation window if an old entry is still
        # listed alongside (or instead of) the new one - picking the first
        # regex match isn't safe since catalog content during a glitch isn't
        # guaranteed. On 2026-08-11 this picked a stale ..._20220512.nc file
        # instead of the current ..._20260727.nc one, which got written
        # straight into eddy_<polarity>_latest.nc before the run crashed
        # later on an unrelated SLA date-range check - exact mechanism
        # (wrong entry picked out of several vs. only the stale one served)
        # unconfirmed, but picking by end-date is a free improvement either
        # way. The real backstop for a same-single-entry glitch, where this
        # fix alone wouldn't help, is the overwrite guard in
        # fetch_eddy_subset below.
        matches = re.findall(
            rf'name="(Eddy_trajectory_nrt_3\.2exp_{polarity}_\d{{8}}_(\d{{8}}))\.nc"',
            catalog_xml,
        )
        if not matches:
            raise RuntimeError(f"Could not find current {polarity} eddy trajectory filename in AVISO catalog")
        best_stem, _ = max(matches, key=lambda m: m[1])
        filenames[polarity] = f"{best_stem}.nc"
    return filenames


def _basic_auth_header(host):
    try:
        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError) as e:
        raise RuntimeError(f"Could not read ~/.netrc for {host}: {e}") from e
    if auth is None:
        raise RuntimeError(f"No ~/.netrc entry found for machine {host}")
    login, _, password = auth
    token = base64.b64encode(f"{login}:{password}".encode()).decode()
    return f"Basic {token}"


def _cleanup_stale(polarity, filename, raw_dir):
    """Removes any local copy left over from a previous (differently
    end-dated) AVISO filename for this polarity, so disk usage doesn't grow
    every time AVISO rolls the date."""
    prefix = f"Eddy_trajectory_nrt_3.2exp_{polarity}_"
    for old_file in glob.glob(os.path.join(raw_dir, f"{prefix}*.nc")):
        if os.path.basename(old_file) != filename:
            os.remove(old_file)
            old_stamp = old_file + ".last_modified"
            if os.path.exists(old_stamp):
                os.remove(old_stamp)


def _read_stamp(stamp_path):
    """Returns {"last_modified": ..., "size": ...} for a verified-complete
    previous download, or None if there is no stamp, it is unreadable, or
    it predates this size-tracking format (e.g. a plain Last-Modified
    string written by an older version of this script) - treated the same
    as "no stamp" so a stale/corrupt cache always falls back to a full
    fresh download rather than being trusted blindly."""
    try:
        with open(stamp_path) as f:
            data = json.load(f)
        if "last_modified" in data and "size" in data:
            return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return None


def download_eddy_file(polarity, filename, raw_dir, max_attempts=5):
    """Downloads filename via THREDDS's plain HTTP fileServer service (see
    module docstring for why, instead of OPeNDAP).

    Downloads to a <filename>.part temp file and only atomically moves it
    into place once its size is verified against the server's reported
    size - a run that got cut short partway through (seen in practice: a
    ~21min connection cutoff, likely a university proxy/firewall duration
    limit, well short of these ~1.6-1.7GB files finishing) must never
    silently overwrite a good cached copy with a truncated one.

    Before downloading anything, does a cheap HEAD request and compares its
    Content-Length against the size already verified on disk - if they
    match, skips the download entirely. This is a client-side check, not
    conditional GET: confirmed via direct testing (curl -n with a correct
    If-Modified-Since still got a plain 200, never 304) that THREDDS's
    fileServer ignores that header outright. The originally-written version
    of this function relied on If-Modified-Since/304 and silently
    re-downloaded the full ~1.6-1.7GB file on every single run regardless
    of whether AVISO's copy had actually changed, since the 304 branch
    could never fire against this server.

    If a transfer comes up short, retries using an HTTP Range request to
    resume from the byte reached rather than restarting the full transfer
    (confirmed via the HEAD response's Accept-Ranges: bytes that this is
    supported).
    """
    os.makedirs(raw_dir, exist_ok=True)
    local_path = os.path.join(raw_dir, filename)
    stamp_path = local_path + ".last_modified"
    tmp_path = local_path + ".part"
    url = f"{FILESERVER_BASE}/{filename}"
    auth_header = _basic_auth_header(AVISO_HOST)

    stamp = _read_stamp(stamp_path)
    local_verified = (
        stamp is not None
        and os.path.exists(local_path)
        and os.path.getsize(local_path) == stamp["size"]
    )

    if local_verified and not os.path.exists(tmp_path):
        try:
            head_req = urllib.request.Request(url, headers={"Authorization": auth_header}, method="HEAD")
            with urllib.request.urlopen(head_req, timeout=60) as resp:
                remote_size = resp.headers.get("Content-Length")
            if remote_size is not None and int(remote_size) == stamp["size"]:
                print(f"{polarity}: {filename} unchanged (Content-Length matches cached copy), reusing local copy")
                _cleanup_stale(polarity, filename, raw_dir)
                return local_path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            print(f"{polarity}: HEAD check failed ({e}), falling back to full download")

    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    for attempt in range(1, max_attempts + 1):
        headers = {"Authorization": auth_header}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"{polarity}: connection failed before transfer started ({e}), "
                  f"retrying (attempt {attempt}/{max_attempts})")
            continue

        try:
            with resp:
                got_range = resp.status == 206
                mode = "ab" if (resume_from and got_range) else "wb"
                if mode == "wb":
                    resume_from = 0
                with open(tmp_path, mode) as out:
                    shutil.copyfileobj(resp, out)
                last_modified = resp.headers.get("Last-Modified")
                content_range = resp.headers.get("Content-Range")
                content_length_header = resp.headers.get("Content-Length")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            print(f"{polarity}: download interrupted ({e}) at byte {resume_from}, "
                  f"retrying (attempt {attempt}/{max_attempts})")
            continue

        actual_size = os.path.getsize(tmp_path)
        if content_range and "/" in content_range:
            expected_size = int(content_range.rsplit("/", 1)[1])
        elif not resume_from and content_length_header:
            expected_size = int(content_length_header)
        else:
            expected_size = None

        if expected_size is not None and actual_size != expected_size:
            resume_from = actual_size
            print(f"{polarity}: incomplete download ({actual_size}/{expected_size} bytes), "
                  f"retrying (attempt {attempt}/{max_attempts})")
            continue

        os.replace(tmp_path, local_path)
        if last_modified:
            with open(stamp_path, "w") as f:
                json.dump({"last_modified": last_modified, "size": actual_size}, f)
        print(f"{polarity}: downloaded {filename} ({actual_size} bytes)")
        break
    else:
        raise RuntimeError(f"{polarity}: failed to fully download {filename} after {max_attempts} attempts")

    _cleanup_stale(polarity, filename, raw_dir)
    return local_path


arg_parser = argparse.ArgumentParser(description='Fetch AVISO+ eddy trajectory data for the SPICE bbox',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-s', '--save_dir',
                        dest='save_dir',
                        type=str,
                        default='./cmems_data/eddy_trajectory',
                        help='Directory to write the subset NetCDF files to')
args = arg_parser.parse_args()


def fetch_eddy_subset(polarity, filename, bbox, window_days, output_dir):
    lon_min, lon_max, lat_min, lat_max = bbox
    raw_dir = os.path.join(output_dir, "_raw")
    local_path = download_eddy_file(polarity, filename, raw_dir)

    ds = xr.open_dataset(local_path)

    # Eddy trajectory files are indexed by observation ("obs"), not a
    # dataset-wide time dimension - each obs is one eddy's position on one
    # day. lon is 0-360 in this product; convert to -180/180 to match the
    # rest of the pipeline's bbox convention before filtering.
    lon = ds["longitude"].values.copy()
    lon[lon > 180] -= 360

    time_vals = pd.to_datetime(ds["time"].values)
    # Cutoff is relative to the latest observation actually present in this
    # file, not wall-clock "now" - AVISO's NRT eddy atlas lags real-time by
    # ~2 weeks in practice (confirmed: the file's own filename end-date is
    # consistently ~14 days behind the day it's fetched), so a wall-clock-
    # relative window can silently return zero eddies once that lag reaches
    # window_days - exactly what happened on first test (both polarities
    # came back "no observations in bbox"). Same class of bug already fixed
    # for platform tracks elsewhere in this pipeline (see
    # get_platform_track's docstring in SPICE_CMEMS_SAT.py) - windowing
    # relative to the data's own latest timestamp instead of the wall clock.
    latest_obs = time_vals.max()
    cutoff = latest_obs - pd.Timedelta(days=window_days)

    mask = (
        (lon >= lon_min) & (lon <= lon_max) &
        (ds["latitude"].values >= lat_min) & (ds["latitude"].values <= lat_max) &
        (time_vals >= cutoff)
    )

    if not mask.any():
        print(f"{polarity}: no observations in bbox within the last {window_days} days")
        return None, latest_obs

    subset = ds.isel(obs=np.where(mask)[0])
    # Clear encoding inherited from the giant source file (e.g. packed
    # int16/int32 + scale_factor) before saving - xarray warned several
    # variables (including time) would be written as an integer dtype with
    # no _FillValue, which can't represent NaN and would silently corrupt
    # any missing values on round-trip. This is just a small derived subset
    # file, so there's no need to preserve the source's storage-optimized
    # encoding; let xarray pick a natural one for the actual in-memory dtype.
    for var in subset.variables:
        subset[var].encoding = {}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eddy_{polarity}_latest.nc")

    # Never let a bad fetch silently overwrite a good previous output with
    # older data - a wrong/stale AVISO catalog pick did exactly this on
    # 2026-08-11 (see discover_eddy_filenames), writing ~2022 data over the
    # previous day's real eddies before the run crashed later on an
    # unrelated SLA-background step, so the bad overwrite went unnoticed
    # until the log was read the next morning. .load()+.close() avoids a
    # confirmed sequential-dataset-open cross-contamination bug in this
    # environment (see project memory - same fix used in cmems_sla_adt.py).
    if os.path.exists(out_path):
        existing = xr.open_dataset(out_path)
        existing_latest = pd.to_datetime(existing["time"].load().values).max()
        existing.close()
        if latest_obs < existing_latest:
            raise RuntimeError(
                f"{polarity}: new data's latest observation ({latest_obs:%Y-%m-%d}) is older than "
                f"the existing {out_path}'s ({existing_latest:%Y-%m-%d}) - refusing to overwrite with "
                f"stale data"
            )

    subset.to_netcdf(out_path)
    print(f"{polarity}: {mask.sum()} observations, {len(np.unique(subset['track'].values))} unique eddies, "
          f"wrote {out_path}")
    return out_path, latest_obs


def fetch_sla_background(target_date, bbox, output_dir, max_lookback_days=3):
    """Fetches the SLA snapshot for target_date (the eddy data's own latest
    observation date, not wall-clock "now") so eddy_trajectory_plot.py can
    overlay eddy positions on the actual SSH anomaly field they were
    derived from, rather than an unrelated, much more recent SLA snapshot.
    Always overwrites the same sla_background.nc - only the current plot's
    background is needed, not a history of them.

    target_date is itself a real day AVISO already reported eddy
    observations for, so unlike cmems_download.py's NRT walk-back (which
    exists to handle "how far behind is today's data" uncertainty), a
    small lookback here is just defensive in case that exact day happens to
    have a gap in SLA coverage.
    """
    target_date = pd.Timestamp(target_date)
    lon_min, lon_max, lat_min, lat_max = bbox
    os.makedirs(output_dir, exist_ok=True)
    out_name = "sla_background"
    out_path = os.path.join(output_dir, f"{out_name}.nc")

    last_error = None
    for days_back in range(max_lookback_days + 1):
        day = target_date - pd.Timedelta(days=days_back)
        try:
            copernicusmarine.subset(
                dataset_id=SLA_DATASET_ID,
                variables=["sla"],
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=day.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=day.strftime("%Y-%m-%dT23:59:59"),
                output_filename=f"{out_name}.nc",
                output_directory=output_dir,
                overwrite=True,
            )
            print(f"SLA background: fetched {day:%Y-%m-%d} (target was {target_date:%Y-%m-%d})")
            return out_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not fetch SLA background within {max_lookback_days} days of {target_date:%Y-%m-%d}"
    ) from last_error


if __name__ == "__main__":
    eddy_datasets = discover_eddy_filenames()
    latest_obs_all = []
    for polarity, filename in eddy_datasets.items():
        _, latest_obs = fetch_eddy_subset(polarity, filename, TROP_WTRN_ATL_EXTENT, ROLLING_WINDOW_DAYS, args.save_dir)
        latest_obs_all.append(latest_obs)
    fetch_sla_background(max(latest_obs_all), TROP_WTRN_ATL_EXTENT, args.save_dir)