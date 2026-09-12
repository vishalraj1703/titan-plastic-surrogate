"""72h rollout with the analytical windage baseline added as a THIRD predictor, alongside
pure advection and the neural surrogate (reviewer major comment #3 remedy: 'Add a direct
analytical windage baseline ... compare the neural surrogate against this baseline').
"""
import numpy as np
import torch
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.models.dataset import CUBE_FIELDS, DT_SEC
from titan.models.surrogate import ResidualMLP

DATA_INTERIM = r"E:\oceanography\data\interim"
DATA_PROCESSED = r"E:\oceanography\data\processed"
DEG_M = 111_320.0
HORIZON_H = 72
STEPS = HORIZON_H // 3


def _sample(cube, t, lat, lon):
    pts = {"time": xr.DataArray(t, dims="s"), "latitude": xr.DataArray(lat, dims="s"),
           "longitude": xr.DataArray(lon, dims="s")}
    return {f: np.nan_to_num(cube[f].interp(**pts, method="linear").values, nan=0.0) for f in CUBE_FIELDS}


def _load_surrogate():
    ck = torch.load(f"{DATA_PROCESSED}/surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)


def rollout_class(klass, model, mu, sd, mode):
    """mode in {'baseline', 'windage', 'surrogate'}"""
    tr = xr.open_zarr(f"{DATA_PROCESSED}/traj_{klass}_test.zarr")
    cube = xr.open_zarr(f"{DATA_INTERIM}/cube.zarr")

    tlon, tlat, ttime = tr["lon"].values, tr["lat"].values, tr["time"].values
    windage = tr["windage"].values[:, 0]
    is_macro = tr["is_macro"].values[:, 0]
    size_class = tr["size_class"].values[:, 0]

    alive = np.isfinite(tlon[:, :STEPS + 1]).all(1) & np.isfinite(tlat[:, :STEPS + 1]).all(1)
    P = int(alive.sum())
    lon = tlon[alive, 0].astype(np.float64).copy()
    lat = tlat[alive, 0].astype(np.float64).copy()
    age = np.zeros(P, np.float64)
    windage, is_macro, size_class = windage[alive], is_macro[alive], size_class[alive]
    t0 = ttime[alive, 0]

    sep = np.full((P, STEPS), np.nan)
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
        tk_lon, tk_lat = tlon[alive, k + 1], tlat[alive, k + 1]
        dx = (lon - tk_lon) * DEG_M * np.cos(np.deg2rad(tk_lat))
        dy = (lat - tk_lat) * DEG_M
        sep[:, k] = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
    return sep


def main():
    model, mu, sd = _load_surrogate()
    print("=== 72h rollout: pure advection vs analytical windage-law vs neural surrogate ===\n")
    for klass in ("macro", "micro"):
        print(f"[{klass}]")
        for mode in ("baseline", "windage", "surrogate"):
            sep = rollout_class(klass, model, mu, sd, mode)
            final = sep[:, -1]
            med = np.nanmedian(final)
            pct15 = float(np.mean(final <= 15)) * 100
            lo, hi = np.percentile(
                [np.nanmedian(final[np.random.default_rng(i).integers(0, len(final), len(final))])
                 for i in range(500)], [2.5, 97.5])
            print(f"  {mode:10s}: median={med:6.1f} km [95% CI {lo:.1f},{hi:.1f}]   {pct15:5.1f}% within 15km")
        print()


if __name__ == "__main__":
    main()
