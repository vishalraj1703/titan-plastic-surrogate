"""Milestone 3 — turn OceanParcels trajectories into residual-velocity training pairs.

The learning problem (blueprint Phase 4, Head 1):
    Given the local ocean + wind + particle state at time t, predict the RESIDUAL velocity
    = (actual particle velocity over [t, t+3h]) - (local ocean current at t).

Why residual (not next position): the ocean current is known and dominant; the model only
has to learn the *correction* on top of it (windage, shear effects). This keeps the surrogate
anchored to physics and avoids accumulating drift error. A naive "just follow the current"
baseline predicts residual = 0 — so any real skill on the residual is the GNN earning its keep.

Features per sample X:
    [uo, vo, vorticity, okubo_weiss, u10, v10, wind_speed, age, windage, is_macro, size_class]
Target y:
    [residual_u, residual_v]   (m/s)

Run (after trajectories exist):
    python -m titan.models.dataset
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM, DATA_PROCESSED

DEG_M = 111_320.0
DT_SEC = 3 * 3600.0  # 3-hour output cadence

FEATURES = [
    "uo", "vo", "vorticity", "okubo_weiss",
    "u10", "v10", "wind_speed",
    "age", "windage", "is_macro", "size_class",
]
CUBE_FIELDS = ["uo", "vo", "vorticity", "okubo_weiss", "u10", "v10", "wind_speed"]


def _sample_cube(cube: xr.Dataset, t, lat, lon) -> dict[str, np.ndarray]:
    """Vectorized linear interpolation of cube fields at scattered (t, lat, lon) points."""
    pts = {
        "time": xr.DataArray(t, dims="s"),
        "latitude": xr.DataArray(lat, dims="s"),
        "longitude": xr.DataArray(lon, dims="s"),
    }
    out = {}
    for f in CUBE_FIELDS:
        out[f] = cube[f].interp(**pts, method="linear").values
    return out


def build_training_pairs(tag: str = "", out_name: str = "residual_dataset.npz") -> str:
    """Assemble (X, y) residual-velocity pairs from macro + micro trajectories.

    `tag` selects the trajectory set: "" -> training (traj_micro.zarr), "test" ->
    the independent held-out set (traj_micro_test.zarr). A per-sample `group` array
    (particle id, offset per class) is also saved so evaluation can, if desired, split
    strictly by trajectory rather than by timestep.
    """
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    suffix = f"_{tag}" if tag else ""

    X_parts, y_parts, g_parts = [], [], []
    gid_offset = 0
    for klass in ("micro", "macro"):
        tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}{suffix}.zarr")
        lon = tr["lon"].values      # (traj, obs)
        lat = tr["lat"].values
        tim = tr["time"].values     # datetime64 (traj, obs)
        age = tr["age"].values
        windage = tr["windage"].values
        is_macro = tr["is_macro"].values
        size_class = tr["size_class"].values

        n_traj, n_obs = lon.shape
        # consecutive obs pairs i -> i+1
        lon0, lon1 = lon[:, :-1], lon[:, 1:]
        lat0, lat1 = lat[:, :-1], lat[:, 1:]
        t0 = tim[:, :-1]
        # valid where both endpoints are finite (particle still active)
        valid = np.isfinite(lon0) & np.isfinite(lon1) & np.isfinite(lat0) & np.isfinite(lat1)
        valid &= np.isfinite(t0.astype("float64"))

        # actual velocity over the step (m/s)
        cos_lat = np.cos(np.deg2rad(lat0))
        u_act = (lon1 - lon0) * DEG_M * cos_lat / DT_SEC
        v_act = (lat1 - lat0) * DEG_M / DT_SEC

        vi = valid.ravel()
        s_lon = lon0.ravel()[vi]
        s_lat = lat0.ravel()[vi]
        s_t = t0.ravel()[vi]
        s_uact = u_act.ravel()[vi]
        s_vact = v_act.ravel()[vi]

        # Parcels writes every particle Variable at each obs, so these are (traj, obs).
        s_age = age[:, :-1].ravel()[vi]
        s_wind = windage[:, :-1].ravel()[vi]
        s_macro = is_macro[:, :-1].ravel()[vi]
        s_size = size_class[:, :-1].ravel()[vi]
        # particle id per sample (globally unique across classes) for trajectory-level splits
        pid = np.repeat(np.arange(valid.shape[0]), valid.shape[1])[vi]

        samp = _sample_cube(cube, s_t, s_lat, s_lon)
        # A sample is valid if it sits on real ocean (finite current + wind). vorticity /
        # okubo_weiss can be NaN at coast cells whose derivative touched land — those are
        # physically "no resolvable rotation/strain", so fill with 0 rather than discard.
        good = np.isfinite(samp["uo"]) & np.isfinite(samp["u10"])
        for k in samp:
            samp[k] = np.nan_to_num(samp[k][good], nan=0.0, posinf=0.0, neginf=0.0)

        res_u = s_uact[good] - samp["uo"]
        res_v = s_vact[good] - samp["vo"]

        X = np.column_stack([
            samp["uo"], samp["vo"], samp["vorticity"], samp["okubo_weiss"],
            samp["u10"], samp["v10"], samp["wind_speed"],
            s_age[good], s_wind[good], s_macro[good], s_size[good],
        ]).astype(np.float32)
        y = np.column_stack([res_u, res_v]).astype(np.float32)

        X_parts.append(X)
        y_parts.append(y)
        g_parts.append(pid[good] + gid_offset)
        gid_offset += valid.shape[0]
        print(f"[dataset] {klass}{suffix}: {X.shape[0]} samples")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    group = np.concatenate(g_parts).astype(np.int64)

    # standardization stats (features only; targets kept in physical m/s)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8

    out = DATA_PROCESSED / out_name
    np.savez(
        out, X=X, y=y, mu=mu, sd=sd, group=group,
        features=np.array(FEATURES),
    )
    print(f"[dataset] wrote {out}  X={X.shape} y={y.shape}")
    print(f"[dataset] residual speed (m/s): micro+macro combined")
    print(f"          |res| mean={np.sqrt((y**2).sum(1)).mean():.3f}  max={np.sqrt((y**2).sum(1)).max():.3f}")
    return str(out)


if __name__ == "__main__":
    build_training_pairs()  # training set
    build_training_pairs(tag="test", out_name="residual_dataset_test.npz")  # held-out set
