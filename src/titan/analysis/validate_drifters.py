"""Milestone 7 — real-world validation against NOAA GPS drifter buoys.

This is the honesty capstone: instead of testing our model against a physics simulator, we
test it against REAL objects tracked in the real ocean. For each real drifter in the NW
Mediterranean during July 2022 (a window covered by both the Copernicus currents and the
Global Drifter Program), we release a virtual particle at the drifter's starting position and
advect it with the same Copernicus surface currents our whole system uses, then measure the
separation between our virtual track and the drifter's true GPS track over time.

If our virtual particles stay close to real drifters, the ocean forcing that underlies the
entire project is validated against reality. Separation will grow with lead time (forecasts
degrade); we report the growth honestly.

Note: standard drifters are drogued (follow the ~15 m current), so we compare against pure
surface-current advection (no windage). This validates the transport backbone; it is not a
test of the plastic-specific windage parameterisation.

Run (after val_currents_2022-07.nc and val_wind_2022-07.nc are downloaded):
    python -m titan.analysis.validate_drifters
"""
from __future__ import annotations

import csv
import datetime
import io
import urllib.request

import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM, DATA_RAW, DATA_PROCESSED

BOX = dict(lon_min=3, lon_max=11, lat_min=41, lat_max=44.5)
WINDOW = ("2022-07-01", "2022-07-31")
MAX_DAYS = 14           # forecast horizon per drifter
DEG_M = 111_320.0
ERDDAP = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_hourly_qc.csv"


def build_val_cube() -> str:
    """Minimal validation cube (uo, vo, land_mask, u10, v10) from the 2022 files."""
    cur = xr.open_dataset(DATA_RAW / "val_currents_2022-07.nc")
    if "depth" in cur.dims:
        cur = cur.isel(depth=0, drop=True)
    u, v = cur["uo"], cur["vo"]
    land = u.isel(time=0).isnull().rename("land_mask")

    w = xr.open_dataset(DATA_RAW / "val_wind_2022-07.nc").rename(
        {"eastward_wind": "u10", "northward_wind": "v10"})
    w = w[["u10", "v10"]].resample(time="1D").mean().interp(
        latitude=cur["latitude"], longitude=cur["longitude"], time=cur["time"],
        method="linear", kwargs={"fill_value": None})
    w = w.ffill("longitude").bfill("longitude").ffill("latitude").bfill("latitude")

    cube = xr.Dataset({"uo": u, "vo": v, "land_mask": land, "u10": w["u10"], "v10": w["v10"]})
    out = DATA_INTERIM / "val_cube.zarr"
    cube.chunk({"time": 31}).to_zarr(out, mode="w")
    print(f"[val] built {out}  dims={dict(cube.sizes)}")
    return str(out)


def fetch_drifters() -> dict:
    """Download hourly drifter tracks in the box+window. Returns {id: (times, lon, lat)}."""
    q = (f"?ID%2Ctime%2Clongitude%2Clatitude"
         f"&longitude%3E={BOX['lon_min']}&longitude%3C={BOX['lon_max']}"
         f"&latitude%3E={BOX['lat_min']}&latitude%3C={BOX['lat_max']}"
         f"&time%3E={WINDOW[0]}T00:00:00Z&time%3C={WINDOW[1]}T23:59:59Z")
    req = urllib.request.Request(ERDDAP + q, headers={"User-Agent": "titan"})
    data = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(data)))[2:]
    tracks: dict = {}
    for r in rows:
        if not r or len(r) < 4:
            continue
        did, t, lo, la = r[0], r[1], r[2], r[3]
        try:
            tracks.setdefault(did, []).append((np.datetime64(t[:19]), float(lo), float(la)))
        except ValueError:
            continue
    out = {}
    for did, pts in tracks.items():
        pts.sort()
        ts = np.array([p[0] for p in pts])
        lo = np.array([p[1] for p in pts]); la = np.array([p[2] for p in pts])
        if ts.size >= 24 * 3:  # at least ~3 days of hourly data
            out[did] = (ts, lo, la)
    print(f"[val] usable drifters (>=3 days in window): {len(out)}")
    return out


def _build_fieldset():
    from parcels import FieldSet
    cube = xr.open_zarr(DATA_INTERIM / "val_cube.zarr")
    variables = {"U": "uo", "V": "vo", "land_mask": "land_mask"}
    dimensions = {"U": {"lon": "longitude", "lat": "latitude", "time": "time"},
                  "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
                  "land_mask": {"lon": "longitude", "lat": "latitude"}}
    return FieldSet.from_xarray_dataset(cube, variables=variables, dimensions=dimensions,
                                        mesh="spherical", allow_time_extrapolation=True)


def _sep_km(lo1, la1, lo2, la2):
    dx = (lo1 - lo2) * DEG_M * np.cos(np.deg2rad(la2)) / 1000.0
    dy = (la1 - la2) * DEG_M / 1000.0
    return np.sqrt(dx ** 2 + dy ** 2)


def validate():
    from parcels import AdvectionRK4, JITParticle
    from titan.physics import kernels as K

    cube = xr.open_zarr(DATA_INTERIM / "val_cube.zarr")
    t0_cube = cube["time"].values[0]
    fieldset = _build_fieldset()
    drifters = fetch_drifters()

    step_h = 6
    nsteps = MAX_DAYS * 24 // step_h
    all_sep = []          # (n_drifters, nsteps)
    examples = []         # (real_lon, real_lat, virt_lon, virt_lat) for a few

    for k, (did, (ts, lo, la)) in enumerate(drifters.items()):
        start = max(ts[0], t0_cube)
        if start >= ts[-1]:
            continue
        i0 = int(np.argmin(np.abs(ts - start)))
        pset = __import__("parcels").ParticleSet(fieldset=fieldset, pclass=JITParticle,
                                                 lon=[lo[i0]], lat=[la[i0]],
                                                 time=[start.astype("datetime64[s]").astype(object)])
        virt_lon, virt_lat, sep = [], [], []
        for s in range(nsteps):
            tnow = start + np.timedelta64(step_h * s, "h")
            if tnow > ts[-1]:
                break
            j = int(np.argmin(np.abs(ts - tnow)))
            vlon = float(pset.lon[0]) if pset.size else np.nan
            vlat = float(pset.lat[0]) if pset.size else np.nan
            virt_lon.append(vlon); virt_lat.append(vlat)
            sep.append(_sep_km(vlon, vlat, lo[j], la[j]) if pset.size else np.nan)
            if pset.size:
                kern = pset.Kernel(AdvectionRK4) + pset.Kernel(K.delete_oob)
                pset.execute(kern, runtime=datetime.timedelta(hours=step_h),
                             dt=datetime.timedelta(minutes=20), verbose_progress=False)
        row = np.full(nsteps, np.nan); row[:len(sep)] = sep
        all_sep.append(row)
        # save every drifter's real + virtual track so the figure can pick a good example
        rl = lo[i0:i0 + len(sep) * step_h]; ra = la[i0:i0 + len(sep) * step_h]
        examples.append((rl, ra, np.array(virt_lon), np.array(virt_lat)))

    all_sep = np.array(all_sep)
    hours = np.arange(1, nsteps + 1) * step_h
    med = np.nanmedian(all_sep, 0)
    print("\n=== M7 REAL-WORLD VALIDATION (virtual particles vs real drifters) ===")
    print(f"  drifters validated: {all_sep.shape[0]}")
    for d in (1, 3, 7, 14):
        idx = d * 24 // step_h - 1
        if idx < len(med):
            print(f"  day {d:2d}: median separation = {med[idx]:.1f} km")
    np.savez(DATA_PROCESSED / "drifter_validation.npz", sep=all_sep, hours=hours,
             examples=np.array(examples, dtype=object))
    return all_sep, hours


if __name__ == "__main__":
    build_val_cube()
    validate()
