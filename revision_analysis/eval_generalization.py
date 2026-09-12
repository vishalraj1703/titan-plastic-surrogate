"""Reviewer major comment #1: evaluate the ALREADY-TRAINED surrogate (trained only on
January 2024) on residual-velocity prediction under JULY 2022 forcing -- a held-out season
and year, same domain. This is the genuine environmental-generalization test the reviewer
asked for, as distinct from the in-distribution independent test set (same month, different
particle seed).
"""
import numpy as np
import torch
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.models.dataset import CUBE_FIELDS, DT_SEC, FEATURES
from titan.models.surrogate import ResidualMLP


def sample_cube(cube, t, lat, lon):
    pts = {"time": xr.DataArray(t, dims="s"), "latitude": xr.DataArray(lat, dims="s"),
           "longitude": xr.DataArray(lon, dims="s")}
    return {f: cube[f].interp(**pts, method="linear").values for f in CUBE_FIELDS}


def build_pairs(klass, cube):
    DEG_M = 111_320.0
    tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}_val2022.zarr")
    lon, lat, tim = tr["lon"].values, tr["lat"].values, tr["time"].values
    age, windage = tr["age"].values, tr["windage"].values
    is_macro, size_class = tr["is_macro"].values, tr["size_class"].values

    lon0, lon1 = lon[:, :-1], lon[:, 1:]
    lat0, lat1 = lat[:, :-1], lat[:, 1:]
    t0 = tim[:, :-1]
    valid = np.isfinite(lon0) & np.isfinite(lon1) & np.isfinite(lat0) & np.isfinite(lat1) & np.isfinite(t0.astype("float64"))
    cos_lat = np.cos(np.deg2rad(lat0))
    u_act = (lon1 - lon0) * DEG_M * cos_lat / DT_SEC
    v_act = (lat1 - lat0) * DEG_M / DT_SEC

    vi = valid.ravel()
    s_lon, s_lat, s_t = lon0.ravel()[vi], lat0.ravel()[vi], t0.ravel()[vi]
    s_uact, s_vact = u_act.ravel()[vi], v_act.ravel()[vi]
    s_age = age[:, :-1].ravel()[vi]
    s_wind = windage[:, :-1].ravel()[vi]
    s_macro = is_macro[:, :-1].ravel()[vi]
    s_size = size_class[:, :-1].ravel()[vi]

    samp = sample_cube(cube, s_t, s_lat, s_lon)
    good = np.isfinite(samp["uo"]) & np.isfinite(samp["u10"])
    for k in samp:
        samp[k] = np.nan_to_num(samp[k][good], nan=0.0, posinf=0.0, neginf=0.0)

    res_u = s_uact[good] - samp["uo"]
    res_v = s_vact[good] - samp["vo"]
    X = np.column_stack([samp["uo"], samp["vo"], samp["vorticity"], samp["okubo_weiss"],
                          samp["u10"], samp["v10"], samp["wind_speed"],
                          s_age[good], s_wind[good], s_macro[good], s_size[good]]).astype(np.float32)
    y = np.column_stack([res_u, res_v]).astype(np.float32)
    return X, y


def skill(pred, true):
    mse_model = ((pred - true) ** 2).sum(1).mean()
    mse_base = (true ** 2).sum(1).mean()
    return 1.0 - mse_model / mse_base


def bootstrap_ci(pred, true, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = skill(pred[idx], true[idx])
    return np.percentile(vals, [2.5, 97.5])


def main():
    ck = torch.load(DATA_PROCESSED / "surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    assert feats == FEATURES
    model = ResidualMLP(len(feats)); model.load_state_dict(ck["state_dict"]); model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)
    macro_i = feats.index("is_macro")
    u10_i, v10_i, wind_i = feats.index("u10"), feats.index("v10"), feats.index("windage")

    cube = xr.open_zarr(DATA_INTERIM / "val_cube_full.zarr")

    print("=== UNSEEN-FORCING GENERALIZATION: trained on Jan 2024, evaluated on Jul 2022 ===\n")
    X_all, y_all, macro_all = [], [], []
    for klass in ("micro", "macro"):
        X, y = build_pairs(klass, cube)
        X_all.append(X); y_all.append(y); macro_all.append(X[:, macro_i])
    X = np.concatenate(X_all); y = np.concatenate(y_all); macro_mask = np.concatenate(macro_all) == 1

    with torch.no_grad():
        pred = model(torch.tensor((X - mu) / sd)).numpy()
    pred_wind = np.column_stack([X[:, wind_i] * X[:, u10_i], X[:, wind_i] * X[:, v10_i]])

    for name, m in [("macro", macro_mask), ("micro", ~macro_mask), ("overall", np.ones(len(X), bool))]:
        s_net = skill(pred[m], y[m]); lo, hi = bootstrap_ci(pred[m], y[m])
        s_wind = skill(pred_wind[m], y[m])
        rmse_net = np.sqrt(((pred[m] - y[m]) ** 2).sum(1).mean())
        rmse_base = np.sqrt((y[m] ** 2).sum(1).mean())
        print(f"  {name:8s}  n={m.sum():6d}   surrogate skill={s_net:+.3f} [{lo:.3f},{hi:.3f}]   "
              f"analytical-windage skill={s_wind:+.3f}   surrogate RMSE={rmse_net:.3f}  baseline RMSE={rmse_base:.3f}")


if __name__ == "__main__":
    main()
