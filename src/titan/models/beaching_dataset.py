"""Milestone 5b (Route A) — beaching-prediction dataset (blueprint Head 4).

Task: for a particle currently in the ocean, will it beach within the next 24 hours?
This is inherently SPATIAL: beaching depends on the coastline geometry around the particle.
We give each cell node a land-mask feature plus the local flow. A pointwise model sees only
its own (ocean) cell, where land-mask is always 0 -> it is blind to the coast. A GNN sees the
7x7 neighbourhood (~+/-25 km) and therefore the shape of the nearby shore.

A particle is 'beached' at the first obs where status==1 (particles that merely leave the
domain keep status==0 and are NOT counted as beaching). Label = 1 for the up-to-8 steps (24 h)
before a beaching event, else 0. Negatives are sub-sampled (4:1) for balance; evaluation uses
ROC-AUC, which is insensitive to that ratio.

Run:
    python -m titan.models.beaching_dataset
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM, DATA_PROCESSED

FLOW_FIELDS = ["uo", "vo", "u10", "v10"]
PARTICLE_FIELDS = ["windage", "is_macro", "size_class", "age"]
RADIUS = 3            # 7x7 neighbourhood
H_OBS = 8             # 24 h horizon (3-hourly steps)
NEG_RATIO = 4


def _build(tag: str, out_name: str, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    lat = cube["latitude"].values; lon = cube["longitude"].values
    ctime = cube["time"].values
    nlat, nlon, T = lat.size, lon.size, ctime.size
    dlat = lat[1] - lat[0]; dlon = lon[1] - lon[0]; t0 = ctime[0]
    land = cube["land_mask"].values.astype(np.float32)                     # (nlat,nlon)
    Fflow = np.stack([np.nan_to_num(cube[f].values) for f in FLOW_FIELDS])  # (4,T,nlat,nlon)

    side = 2 * RADIUS + 1
    ncell = side * side
    suffix = f"_{tag}" if tag else ""

    neigh_all, pfeat_all, y_all, macro_all = [], [], [], []
    for klass in ("micro", "macro"):
        tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}{suffix}.zarr")
        plon = tr["lon"].values; plat = tr["lat"].values
        ptime = tr["time"].values; status = tr["status"].values
        windage = tr["windage"].values; is_macro = tr["is_macro"].values
        size = tr["size_class"].values; age = tr["age"].values
        ntraj, nobs = plon.shape

        # first beaching obs per particle (or -1)
        beach_obs = np.full(ntraj, -1)
        for p in range(ntraj):
            hit = np.where(status[p] == 1)[0]
            if hit.size:
                beach_obs[p] = hit[0]

        rows_p, rows_k = [], []
        labels = []
        for p in range(ntraj):
            bk = beach_obs[p]
            last = bk if bk >= 0 else nobs
            for k in range(last):
                if not np.isfinite(plon[p, k]):
                    break
                lab = 1 if (bk >= 0 and 1 <= (bk - k) <= H_OBS) else 0
                rows_p.append(p); rows_k.append(k); labels.append(lab)
        rows_p = np.array(rows_p); rows_k = np.array(rows_k); labels = np.array(labels)

        # sub-sample negatives 4:1
        pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
        keep_neg = rng.choice(neg, size=min(neg.size, NEG_RATIO * pos.size), replace=False)
        sel = np.concatenate([pos, keep_neg]); rng.shuffle(sel)
        rows_p, rows_k, labels = rows_p[sel], rows_k[sel], labels[sel]

        s_lon = plon[rows_p, rows_k]; s_lat = plat[rows_p, rows_k]
        s_t = ptime[rows_p, rows_k]
        ci = np.clip(np.round((s_lat - lat[0]) / dlat).astype(int), 0, nlat - 1)
        cj = np.clip(np.round((s_lon - lon[0]) / dlon).astype(int), 0, nlon - 1)
        ti = np.clip(np.round((s_t - t0) / np.timedelta64(1, "D")).astype(int), 0, T - 1)

        n = ci.size
        neigh = np.empty((n, ncell, 1 + len(FLOW_FIELDS)), np.float32)
        c = 0
        for di in range(-RADIUS, RADIUS + 1):
            for dj in range(-RADIUS, RADIUS + 1):
                ii = np.clip(ci + di, 0, nlat - 1); jj = np.clip(cj + dj, 0, nlon - 1)
                neigh[:, c, 0] = land[ii, jj]
                neigh[:, c, 1:] = Fflow[:, ti, ii, jj].T
                c += 1

        pf = np.column_stack([windage[rows_p, rows_k], is_macro[rows_p, rows_k],
                              size[rows_p, rows_k], age[rows_p, rows_k]]).astype(np.float32)
        neigh_all.append(neigh); pfeat_all.append(pf)
        y_all.append(labels.astype(np.float32)); macro_all.append(is_macro[rows_p, rows_k].astype(np.int32))
        print(f"[beach-ds] {klass}{suffix}: {n} samples ({int(labels.sum())} positive)")

    neigh = np.concatenate(neigh_all); pfeat = np.concatenate(pfeat_all)
    y = np.concatenate(y_all); macro = np.concatenate(macro_all)
    out = DATA_PROCESSED / out_name
    np.savez(out, neigh=neigh, pfeat=pfeat, y=y, macro=macro, radius=RADIUS)
    print(f"[beach-ds] wrote {out}  neigh={neigh.shape} positives={int(y.sum())}/{y.size}")
    return str(out)


if __name__ == "__main__":
    _build("", "beach_train.npz")
    _build("test", "beach_test.npz")
