"""Milestone 5/6 — convergence zones + cleanup boxes (blueprint Phase 4 Head 5, Phase 6).

The app's "killer feature": tell a cleanup vessel WHERE plastic accumulates. We derive it
directly from the physics we already have, using two complementary signals:

  1. CONVERGENCE (Eulerian): divergence of the surface current field,
        div = du/dx + dv/dy   (per second, computed in metres).
     div < 0 means water is flowing together -> floating debris is trapped there.

  2. ACCUMULATION (Lagrangian): where our simulated particles actually spend time.
     We bin every particle position over the second half of the run into a density grid.
     High density = a real accumulation zone for this source.

"Cleanup boxes" = bounding boxes around the highest-density connected regions. This is the
statistical, honest product: not "one piece here" but "elevated concentration in this zone".

Run:
    python -m titan.analysis.convergence
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from titan.config import DATA_INTERIM, DATA_PROCESSED

DEG_M = 111_320.0


def divergence_field(cube: xr.Dataset) -> xr.DataArray:
    """Time-mean surface-current divergence (1/s); negative = convergence."""
    u, v = cube["uo"], cube["vo"]
    cos_lat = np.cos(np.deg2rad(cube["latitude"]))
    du_dx = u.differentiate("longitude") / (DEG_M * cos_lat)
    dv_dy = v.differentiate("latitude") / DEG_M
    div = (du_dx + dv_dy).mean("time")
    return div.rename("divergence")


def particle_density(klass: str, cube: xr.Dataset, late_frac: float = 0.5) -> np.ndarray:
    """2D histogram (on the cube grid) of particle positions over the run's second half."""
    tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}.zarr")
    lon = tr["lon"].values
    lat = tr["lat"].values
    n_obs = lon.shape[1]
    k0 = int(n_obs * late_frac)
    lon = lon[:, k0:].ravel()
    lat = lat[:, k0:].ravel()
    ok = np.isfinite(lon) & np.isfinite(lat)

    lon_edges = np.append(cube["longitude"].values, cube["longitude"].values[-1] + 0.083)
    lat_edges = np.append(cube["latitude"].values, cube["latitude"].values[-1] + 0.083)
    H, _, _ = np.histogram2d(lon[ok], lat[ok], bins=[lon_edges, lat_edges])
    return H.T  # (lat, lon)


def find_cleanup_boxes(density: np.ndarray, cube: xr.Dataset, top_pct: float = 96,
                       min_cells: int = 3):
    """Return bounding boxes (lon0, lat0, lon1, lat1, strength) of top-density clusters."""
    thresh = np.percentile(density[density > 0], top_pct)
    mask = density >= thresh
    labels, n = ndimage.label(mask)
    lons = cube["longitude"].values
    lats = cube["latitude"].values
    boxes = []
    for i in range(1, n + 1):
        cells = labels == i
        if cells.sum() < min_cells:
            continue
        yy, xx = np.where(cells)
        strength = float(density[cells].sum())
        boxes.append((
            float(lons[xx.min()]), float(lats[yy.min()]),
            float(lons[xx.max()]), float(lats[yy.max()]), strength,
        ))
    boxes.sort(key=lambda b: -b[4])
    return boxes


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    div = divergence_field(cube)
    dens_micro = particle_density("micro", cube)
    boxes = find_cleanup_boxes(dens_micro, cube)

    lons, lats = cube["longitude"].values, cube["latitude"].values
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # (1) convergence field
    div.plot(ax=ax[0], cmap="RdBu", robust=True, cbar_kwargs={"label": "divergence (1/s)"})
    ax[0].set_title("Convergence (blue < 0 = water flows together = traps plastic)")

    # (2) accumulation density + cleanup boxes
    dm = np.ma.masked_where(dens_micro == 0, dens_micro)
    im = ax[1].pcolormesh(lons, lats, dm, cmap="hot_r", shading="auto")
    fig.colorbar(im, ax=ax[1], label="micro-plastic residence density")
    for j, (lo0, la0, lo1, la1, s) in enumerate(boxes[:5]):
        ax[1].add_patch(Rectangle((lo0, la0), lo1 - lo0 or 0.1, la1 - la0 or 0.1,
                                  fill=False, edgecolor="yellow", lw=2.5))
        ax[1].text(lo0, la1 + 0.05, f"Box {chr(65+j)}", color="yellow", fontweight="bold")
    ax[1].set_title(f"Accumulation + CLEANUP BOXES ({len(boxes)} zones found)")
    ax[1].set_xlabel("longitude"); ax[1].set_ylabel("latitude")

    fig.suptitle("Milestone 5: Convergence zones -> where to send the cleanup vessel")
    fig.tight_layout()
    out = DATA_INTERIM / "figs" / "cleanup_boxes.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")

    print("=== CLEANUP BOXES (micro-plastic accumulation zones) ===")
    for j, (lo0, la0, lo1, la1, s) in enumerate(boxes[:5]):
        clon, clat = (lo0 + lo1) / 2, (la0 + la1) / 2
        print(f"  Box {chr(65+j)}: center ({clon:.2f}E, {clat:.2f}N)  "
              f"span [{lo0:.2f}-{lo1:.2f}E, {la0:.2f}-{la1:.2f}N]  strength={s:.0f}")
    print(f"\n[convergence] wrote {out}")


if __name__ == "__main__":
    main()
