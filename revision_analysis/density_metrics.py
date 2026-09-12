"""Reviewer major comment #5: quantitative validation of the cleanup-box / density claim.

Runs the surrogate and the pure-advection baseline forward for the FULL 10-day horizon (the
same horizon used to build the original density-field figure), bins particle residence over
the second half of the run into the same grid as OceanParcels truth, and reports:
  - Pearson spatial correlation between surrogate density and OceanParcels truth density
  - Top-decile overlap (Jaccard / IoU) of the highest-density 10% of ocean cells
  - Same two metrics for the pure-advection baseline, as a null-model comparison

This directly tests whether the surrogate's Lagrangian density field -- the thing the cleanup
boxes are extracted from -- actually agrees with the physics ground truth, rather than only
inspecting it visually.
"""
import numpy as np
import torch
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.models.dataset import CUBE_FIELDS, DT_SEC
from titan.models.surrogate import ResidualMLP

DEG_M = 111_320.0
HORIZON_H = 240  # 10 days, matches the original density-field / cleanup-box analysis
STEPS = HORIZON_H // 3


def _sample(cube, t, lat, lon):
    pts = {"time": xr.DataArray(t, dims="s"), "latitude": xr.DataArray(lat, dims="s"),
           "longitude": xr.DataArray(lon, dims="s")}
    return {f: np.nan_to_num(cube[f].interp(**pts, method="linear").values, nan=0.0) for f in CUBE_FIELDS}


def rollout_positions(klass, model, mu, sd, mode):
    """Roll a predictor forward for the full horizon; return (lon, lat) history (P, STEPS+1)."""
    tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}_test.zarr")
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    tlon, tlat, ttime = tr["lon"].values, tr["lat"].values, tr["time"].values
    windage = tr["windage"].values[:, 0]
    is_macro = tr["is_macro"].values[:, 0]
    size_class = tr["size_class"].values[:, 0]

    alive0 = np.isfinite(tlon[:, 0]) & np.isfinite(tlat[:, 0])
    P = int(alive0.sum())
    lon = tlon[alive0, 0].astype(np.float64).copy()
    lat = tlat[alive0, 0].astype(np.float64).copy()
    age = np.zeros(P, np.float64)
    windage, is_macro, size_class = windage[alive0], is_macro[alive0], size_class[alive0]
    t0 = ttime[alive0, 0]

    lon_hist = np.full((P, STEPS + 1), np.nan)
    lat_hist = np.full((P, STEPS + 1), np.nan)
    lon_hist[:, 0], lat_hist[:, 0] = lon, lat
    alive = np.ones(P, bool)

    for k in range(STEPS):
        t = t0 + np.timedelta64(3 * k, "h")
        s = _sample(cube, t, lat, lon)
        cur_u, cur_v = s["uo"], s["vo"]
        if mode == "surrogate":
            X = np.column_stack([s["uo"], s["vo"], s["vorticity"], s["okubo_weiss"], s["u10"], s["v10"],
                                  s["wind_speed"], age, windage, is_macro, size_class]).astype(np.float32)
            with torch.no_grad():
                res = model(torch.tensor((X - mu) / sd)).numpy()
            ru, rv = res[:, 0], res[:, 1]
        elif mode == "windage":
            ru, rv = windage * s["u10"], windage * s["v10"]
        else:
            ru = rv = 0.0
        vel_u, vel_v = cur_u + ru, cur_v + rv
        coslat = np.cos(np.deg2rad(lat))
        lon = lon + vel_u * DT_SEC / (DEG_M * coslat)
        lat = lat + vel_v * DT_SEC / DEG_M
        age += DT_SEC
        # stop advancing particles that leave the domain (mirror OceanParcels delete_oob)
        out_of_domain = (lon < 3.0) | (lon > 11.0) | (lat < 41.0) | (lat > 44.5)
        alive &= ~out_of_domain
        lon = np.where(alive, lon, np.nan)
        lat = np.where(alive, lat, np.nan)
        lon_hist[:, k + 1], lat_hist[:, k + 1] = lon, lat
    return lon_hist, lat_hist


def truth_positions(klass):
    tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}_test.zarr")
    return tr["lon"].values, tr["lat"].values  # (P, obs) already at 3h cadence


def density_field(lon_hist, lat_hist, half_start, lon_edges, lat_edges):
    lon_flat = lon_hist[:, half_start:].ravel()
    lat_flat = lat_hist[:, half_start:].ravel()
    m = np.isfinite(lon_flat) & np.isfinite(lat_flat)
    H, _, _ = np.histogram2d(lon_flat[m], lat_flat[m], bins=[lon_edges, lat_edges])
    return H.T


def compare(name, truth_d, pred_d, top_frac=0.10):
    t = truth_d.ravel(); p = pred_d.ravel()
    corr = np.corrcoef(t, p)[0, 1]
    kt = max(1, int(len(t) * top_frac))
    top_t = set(np.argsort(t)[-kt:])
    top_p = set(np.argsort(p)[-kt:])
    iou = len(top_t & top_p) / len(top_t | top_p)
    print(f"  {name:20s}: spatial correlation = {corr:.3f}   top-10% cell IoU = {iou:.3f}")
    return corr, iou


def main():
    ck = torch.load(DATA_PROCESSED / "surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats)); model.load_state_dict(ck["state_dict"]); model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)

    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    lon_c, lat_c = cube["longitude"].values, cube["latitude"].values
    dlon = np.abs(np.diff(lon_c)).mean(); dlat = np.abs(np.diff(lat_c)).mean()
    lon_edges = np.append(lon_c - dlon / 2, lon_c[-1] + dlon / 2)
    lat_edges = np.append(lat_c - dlat / 2, lat_c[-1] + dlat / 2)

    half = STEPS // 2  # second half of the 10-day run, matching the original density analysis

    for klass in ("micro", "macro"):
        print(f"\n=== [{klass}] density-field agreement (surrogate / baseline vs OceanParcels truth) ===")
        tlon, tlat = truth_positions(klass)
        truth_d = density_field(tlon, tlat, half, lon_edges, lat_edges)

        for mode in ("baseline", "windage", "surrogate"):
            lon_h, lat_h = rollout_positions(klass, model, mu, sd, mode)
            pred_d = density_field(lon_h, lat_h, half, lon_edges, lat_edges)
            compare(mode, truth_d, pred_d)


if __name__ == "__main__":
    main()
