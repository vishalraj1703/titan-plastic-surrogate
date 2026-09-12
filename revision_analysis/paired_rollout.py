"""Paired, trajectory-level comparison of surrogate vs analytical-windage baseline in the
72h rollout (reviewer major comment #2, round 2): median separation difference and
within-15km proportion difference, with bootstrap CIs on the paired difference (each particle
contributes exactly one final-separation value, so per-particle resampling here IS already
trajectory-clustered -- unlike the one-step residual metric, this one needs no correction,
only the paired-difference framing the reviewer asked for).
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


def rollout_final_sep(klass, model, mu, sd, mode):
    tr = xr.open_zarr(f"{DATA_PROCESSED}/traj_{klass}_test.zarr")
    cube = xr.open_zarr(f"{DATA_INTERIM}/cube.zarr")
    tlon, tlat, ttime = tr["lon"].values, tr["lat"].values, tr["time"].values
    windage = tr["windage"].values[:, 0]
    is_macro = tr["is_macro"].values[:, 0]
    size_class = tr["size_class"].values[:, 0]

    alive = np.isfinite(tlon[:, :STEPS + 1]).all(1) & np.isfinite(tlat[:, :STEPS + 1]).all(1)
    alive_idx = np.where(alive)[0]
    P = int(alive.sum())
    lon = tlon[alive, 0].astype(np.float64).copy()
    lat = tlat[alive, 0].astype(np.float64).copy()
    age = np.zeros(P, np.float64)
    windage, is_macro, size_class = windage[alive], is_macro[alive], size_class[alive]
    t0 = ttime[alive, 0]

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

    tk_lon, tk_lat = tlon[alive, STEPS], tlat[alive, STEPS]
    dx = (lon - tk_lon) * DEG_M * np.cos(np.deg2rad(tk_lat))
    dy = (lat - tk_lat) * DEG_M
    sep = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
    return alive_idx, sep


def paired_bootstrap(a, b, stat_fn, n_boot=2000, seed=0):
    n = len(a)
    rng = np.random.default_rng(seed)
    point = stat_fn(a) - stat_fn(b)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = stat_fn(a[idx]) - stat_fn(b[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def main():
    ck = torch.load(f"{DATA_PROCESSED}/surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats)); model.load_state_dict(ck["state_dict"]); model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)

    print("=== Paired rollout comparison: surrogate MINUS analytical windage baseline ===")
    print("(same particles for both predictors; 2000 paired bootstrap resamples)\n")
    for klass in ("macro", "micro"):
        idx_s, sep_s = rollout_final_sep(klass, model, mu, sd, "surrogate")
        idx_w, sep_w = rollout_final_sep(klass, model, mu, sd, "windage")
        assert np.array_equal(idx_s, idx_w), "particle sets must match for pairing"
        n = len(sep_s)

        med_diff, lo, hi = paired_bootstrap(sep_s, sep_w, np.median)
        print(f"[{klass}] n={n} particles")
        print(f"  median separation diff (surrogate - windage) = {med_diff:+.2f} km  95% CI [{lo:+.2f}, {hi:+.2f}]")

        pct_fn = lambda x: float(np.mean(x <= 15)) * 100
        pct_diff, plo, phi = paired_bootstrap(sep_s, sep_w, pct_fn)
        print(f"  within-15km %% diff (surrogate - windage)   = {pct_diff:+.1f} pp 95% CI [{plo:+.1f}, {phi:+.1f}]\n")


if __name__ == "__main__":
    main()
