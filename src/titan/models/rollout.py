"""Milestone 4 — autoregressive rollout evaluation (blueprint Phase 8 honesty protocol).

Establishes the real trajectory-error metric and the bar the GNN must beat. For each
held-out test particle we roll the surrogate forward in an autoregressive loop:

    next_pos = pos + (local_current + predicted_residual) * dt

comparing three predictors against the OceanParcels truth trajectory:
  * baseline : residual = 0                (pure advection — ignores windage)
  * surrogate: residual = MLP(features)    (our trained model)

Headline metric = Separation-Distance CDF at 72 h: the fraction of particles whose predicted
position is within D km of the OceanParcels truth. Reported per behavioural class, because
macro (deterministic windage) and micro (irreducible diffusion) have very different ceilings.

Run (after a held-out test set + trained surrogate exist):
    python -m titan.models.rollout
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.models.dataset import CUBE_FIELDS, DT_SEC
from titan.models.surrogate import ResidualMLP

DEG_M = 111_320.0
HORIZON_H = 72          # forecast horizon (hours)
STEPS = HORIZON_H // 3  # 3-hourly steps


def _sample(cube, t, lat, lon):
    pts = {
        "time": xr.DataArray(t, dims="s"),
        "latitude": xr.DataArray(lat, dims="s"),
        "longitude": xr.DataArray(lon, dims="s"),
    }
    return {f: np.nan_to_num(cube[f].interp(**pts, method="linear").values, nan=0.0)
            for f in CUBE_FIELDS}


def _load_surrogate():
    ck = torch.load(DATA_PROCESSED / "surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["mu"].astype(np.float32), ck["sd"].astype(np.float32), feats


def rollout_class(klass: str, model, mu, sd, use_model: bool):
    """Roll a predictor forward for one class; return separation distance (km) [P, STEPS]."""
    tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}_test.zarr")
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")

    tlon = tr["lon"].values      # (P, obs) truth
    tlat = tr["lat"].values
    ttime = tr["time"].values
    windage = tr["windage"].values[:, 0]
    is_macro = tr["is_macro"].values[:, 0]
    size_class = tr["size_class"].values[:, 0]

    # keep only particles alive through the whole horizon (finite truth positions)
    alive = np.isfinite(tlon[:, : STEPS + 1]).all(1) & np.isfinite(tlat[:, : STEPS + 1]).all(1)
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
        if use_model:
            X = np.column_stack([
                s["uo"], s["vo"], s["vorticity"], s["okubo_weiss"],
                s["u10"], s["v10"], s["wind_speed"], age, windage, is_macro, size_class,
            ]).astype(np.float32)
            with torch.no_grad():
                res = model(torch.tensor((X - mu) / sd)).numpy()
            ru, rv = res[:, 0], res[:, 1]
        else:
            ru = rv = 0.0
        vel_u, vel_v = cur_u + ru, cur_v + rv
        coslat = np.cos(np.deg2rad(lat))
        lon = lon + vel_u * DT_SEC / (DEG_M * coslat)
        lat = lat + vel_v * DT_SEC / DEG_M
        age += DT_SEC
        # separation vs truth at step k+1
        tk_lon, tk_lat = tlon[alive, k + 1], tlat[alive, k + 1]
        dx = (lon - tk_lon) * DEG_M * np.cos(np.deg2rad(tk_lat))
        dy = (lat - tk_lat) * DEG_M
        sep[:, k] = np.sqrt(dx**2 + dy**2) / 1000.0  # km
    return sep


def main():
    model, mu, sd, feats = _load_surrogate()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    hours = np.arange(1, STEPS + 1) * 3
    summary = {}
    for klass in ("micro", "macro"):
        for use_model, label, c in [(False, "baseline (advection)", "gray"),
                                    (True, "surrogate (MLP)", "tab:blue")]:
            sep = rollout_class(klass, model, mu, sd, use_model)
            med = np.nanmedian(sep, 0)
            ax_i = ax[0] if klass == "micro" else ax[1]
            ax_i.plot(hours, med, color=c, lw=2, label=label)
            ax_i.fill_between(hours, np.nanpercentile(sep, 25, 0), np.nanpercentile(sep, 75, 0),
                              color=c, alpha=0.15)
            final = sep[:, -1]
            summary[(klass, use_model)] = (np.nanmedian(final),
                                           float(np.mean(final <= 15)) * 100)
        ax_i = ax[0] if klass == "micro" else ax[1]
        ax_i.set_title(f"{klass}: separation vs OceanParcels truth")
        ax_i.set_xlabel("forecast hour"); ax_i.set_ylabel("separation (km)")
        ax_i.legend()
    fig.suptitle("Milestone 4: 72h autoregressive rollout — surrogate vs pure advection")
    fig.tight_layout()
    out = DATA_INTERIM / "figs" / "rollout_eval.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")

    print("\n=== 72h ROLLOUT RESULT (median separation / % within 15km) ===")
    for klass in ("micro", "macro"):
        b_med, b_pct = summary[(klass, False)]
        m_med, m_pct = summary[(klass, True)]
        print(f"  {klass:6s}  baseline: {b_med:6.1f} km ({b_pct:4.1f}% <15km)   "
              f"surrogate: {m_med:6.1f} km ({m_pct:4.1f}% <15km)")
    print(f"\n[rollout] wrote {out}")


if __name__ == "__main__":
    main()
