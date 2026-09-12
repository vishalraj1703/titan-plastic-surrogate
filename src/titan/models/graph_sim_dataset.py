"""Milestone 5b (Route A) — neighbourhood dataset for the GNN learned-simulator.

For every particle-step we extract the 3x3 block of ocean cells around the particle plus the
particle's own properties, and the residual-velocity target. This lets us compare fairly:

  * MLP : uses only the CENTRE cell (index 4)          -> the Paper 1 pointwise model
  * GNN : attends over all 9 cells (centre + 8 neighbours) -> the learned simulator

Evaluation is always on the INDEPENDENT held-out particle set (tag='test', a different random
seed), so there is no particle overlap and no spatial/temporal leakage.

Run:
    python -m titan.models.graph_sim_dataset
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM, DATA_PROCESSED

OCEAN_FIELDS = ["uo", "vo", "vorticity", "okubo_weiss", "u10", "v10", "wind_speed"]
PARTICLE_FIELDS = ["age", "windage", "is_macro", "size_class"]
DEG_M = 111_320.0
DT_SEC = 3 * 3600.0


def _build(tag: str, out_name: str) -> str:
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    lat = cube["latitude"].values
    lon = cube["longitude"].values
    ctime = cube["time"].values
    nlat, nlon, T = lat.size, lon.size, ctime.size
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    t0 = ctime[0]

    # stack ocean fields into (F, T, nlat, nlon), NaN->0
    F = np.stack([np.nan_to_num(cube[f].values) for f in OCEAN_FIELDS])  # (7,T,nlat,nlon)

    suffix = f"_{tag}" if tag else ""
    neigh_parts, p_parts, y_parts, macro_parts = [], [], [], []
    for klass in ("micro", "macro"):
        tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}{suffix}.zarr")
        plon = tr["lon"].values; plat = tr["lat"].values
        ptime = tr["time"].values
        age = tr["age"].values; windage = tr["windage"].values
        is_macro = tr["is_macro"].values; size = tr["size_class"].values

        lon0, lat0 = plon[:, :-1], plat[:, :-1]
        lon1, lat1 = plon[:, 1:], plat[:, 1:]
        t_here = ptime[:, :-1]
        valid = (np.isfinite(lon0) & np.isfinite(lon1) & np.isfinite(lat0) &
                 np.isfinite(lat1) & np.isfinite(t_here.astype("float64")))
        vi = valid.ravel()

        s_lon0 = lon0.ravel()[vi]; s_lat0 = lat0.ravel()[vi]
        s_lon1 = lon1.ravel()[vi]; s_lat1 = lat1.ravel()[vi]
        s_t = t_here.ravel()[vi]

        # nearest grid indices + time index
        ci = np.clip(np.round((s_lat0 - lat[0]) / dlat).astype(int), 0, nlat - 1)
        cj = np.clip(np.round((s_lon0 - lon[0]) / dlon).astype(int), 0, nlon - 1)
        ti = np.clip(np.round((s_t - t0) / np.timedelta64(1, "D")).astype(int), 0, T - 1)

        # gather 3x3 neighbourhood -> (n, 9, 7)
        n = ci.size
        neigh = np.empty((n, 9, len(OCEAN_FIELDS)), np.float32)
        k = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii = np.clip(ci + di, 0, nlat - 1)
                jj = np.clip(cj + dj, 0, nlon - 1)
                neigh[:, k, :] = F[:, ti, ii, jj].T  # (7, n) -> (n, 7)
                k += 1

        # actual residual velocity (m/s), using the CENTRE-cell current as the base
        cos_lat = np.cos(np.deg2rad(s_lat0))
        u_act = (s_lon1 - s_lon0) * DEG_M * cos_lat / DT_SEC
        v_act = (s_lat1 - s_lat0) * DEG_M / DT_SEC
        centre = neigh[:, 4, :]  # 3x3 centre
        res_u = u_act - centre[:, 0]  # uo at centre
        res_v = v_act - centre[:, 1]  # vo at centre
        y = np.column_stack([res_u, res_v]).astype(np.float32)

        s_age = age[:, :-1].ravel()[vi]; s_wind = windage[:, :-1].ravel()[vi]
        s_macro = is_macro[:, :-1].ravel()[vi]; s_size = size[:, :-1].ravel()[vi]
        pfeat = np.column_stack([s_age, s_wind, s_macro, s_size]).astype(np.float32)

        neigh_parts.append(neigh); p_parts.append(pfeat); y_parts.append(y)
        macro_parts.append(s_macro.astype(np.int32))
        print(f"[graph-ds] {klass}{suffix}: {n} samples")

    neigh = np.concatenate(neigh_parts); pfeat = np.concatenate(p_parts)
    y = np.concatenate(y_parts); macro = np.concatenate(macro_parts)

    out = DATA_PROCESSED / out_name
    np.savez(out, neigh=neigh, pfeat=pfeat, y=y, macro=macro,
             ocean_fields=np.array(OCEAN_FIELDS), particle_fields=np.array(PARTICLE_FIELDS))
    print(f"[graph-ds] wrote {out}  neigh={neigh.shape} pfeat={pfeat.shape} y={y.shape}")
    return str(out)


if __name__ == "__main__":
    _build("", "graphsim_train.npz")
    _build("test", "graphsim_test.npz")
