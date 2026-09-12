"""Milestone 5b — flow-intrinsic accumulation target for the GNN vs MLP comparison.

We want a per-cell label that says "does the flow trap plastic here?" that is a property of
the FLOW, not of one particular release source. Method: seed particles uniformly over every
ocean cell, advect them through the time-mean current + macro windage (no diffusion, so the
convergence signal stays sharp), and record the change in particle density per cell.

  accumulation[cell] = final_count - initial_count      (positive => convergence/trapping)

This is inherently a NEIGHBOURHOOD quantity: whether a cell gains particles depends on how the
surrounding flow funnels toward it. A pointwise model that sees only a cell's own velocity
cannot compute it; a graph model that aggregates neighbours can. That is the whole point.

Run:
    python -m titan.analysis.accumulation
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from titan.config import DATA_INTERIM, DATA_PROCESSED, load_config

DEG_M = 111_320.0


def _interp(lat, lon, field):
    """RegularGridInterpolator over (lat, lon), ascending coords, linear, filled at edges."""
    order = np.argsort(lat)
    return RegularGridInterpolator((lat[order], lon), field[order], bounds_error=False, fill_value=np.nan)


def generate(days: int = 5, subgrid: int = 4, cw: float = 0.075, dt_h: float = 1.0) -> str:
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    lat = cube["latitude"].values
    lon = cube["longitude"].values
    um = np.nan_to_num(cube["uo"].mean("time").values)
    vm = np.nan_to_num(cube["vo"].mean("time").values)
    u10 = np.nan_to_num(cube["u10"].mean("time").values)
    v10 = np.nan_to_num(cube["v10"].mean("time").values)
    land = cube["land_mask"].values  # (lat, lon) bool

    fU = _interp(lat, lon, um + cw * u10)   # steady drift velocity incl. windage (m/s)
    fV = _interp(lat, lon, vm + cw * v10)

    # seed subgrid x subgrid points inside every ocean cell
    dlat = np.abs(np.diff(lat)).mean()
    dlon = np.abs(np.diff(lon)).mean()
    offs = (np.arange(subgrid) + 0.5) / subgrid - 0.5
    oy, ox = np.meshgrid(offs * dlat, offs * dlon, indexing="ij")
    plat, plon = [], []
    for i in range(len(lat)):
        for j in range(len(lon)):
            if not land[i, j]:
                plat.append(lat[i] + oy.ravel())
                plon.append(lon[j] + ox.ravel())
    plat = np.concatenate(plat); plon = np.concatenate(plon)
    init_lat, init_lon = plat.copy(), plon.copy()

    dt = dt_h * 3600.0
    steps = int(days * 24 / dt_h)
    alive = np.ones(plat.size, bool)
    for _ in range(steps):
        def vel(la, lo):
            p = np.column_stack([la, lo])
            u = fU(p); v = fV(p)
            return np.nan_to_num(u), np.nan_to_num(v)
        u1, v1 = vel(plat, plon)
        cl = np.cos(np.deg2rad(plat))
        la2 = plat + 0.5 * dt * v1 / DEG_M
        lo2 = plon + 0.5 * dt * u1 / (DEG_M * cl)
        u2, v2 = vel(la2, lo2)
        plat = plat + dt * v2 / DEG_M
        plon = plon + dt * u2 / (DEG_M * np.cos(np.deg2rad(plat)))
        # kill particles that leave the domain
        alive &= (plat > lat.min()) & (plat < lat.max()) & (plon > lon.min()) & (plon < lon.max())
        plat = np.clip(plat, lat.min(), lat.max()); plon = np.clip(plon, lon.min(), lon.max())

    lon_edges = np.append(lon - dlon / 2, lon[-1] + dlon / 2)
    lat_edges = np.append(lat - dlat / 2, lat[-1] + dlat / 2)
    final, _, _ = np.histogram2d(plon[alive], plat[alive], bins=[lon_edges, lat_edges])
    init, _, _ = np.histogram2d(init_lon, init_lat, bins=[lon_edges, lat_edges])
    final = final.T; init = init.T  # (lat, lon)

    accumulation = final - init                       # positive => convergence
    accumulation[land] = 0.0

    out = DATA_PROCESSED / "accumulation_target.npz"
    np.savez(out, accumulation=accumulation.astype(np.float32),
             final=final.astype(np.float32), init=init.astype(np.float32))
    print(f"[accumulation] wrote {out}")
    print(f"[accumulation]   ocean cells={int((~land).sum())}  "
          f"converging={int((accumulation > 0).sum())}  diverging={int((accumulation < 0).sum())}")
    print(f"[accumulation]   accumulation range: [{accumulation.min():.0f}, {accumulation.max():.0f}]")
    return str(out)


if __name__ == "__main__":
    generate()
