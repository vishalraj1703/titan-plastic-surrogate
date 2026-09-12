"""Quick visual sanity checks for the interim cube (Milestone 1 verification).

Run:
    python -m titan.viz.sanity
Produces figures under data/interim/figs/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM


def plot_snapshot(t_index: int = 0) -> Path:
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    snap = cube.isel(time=t_index)

    figdir = DATA_INTERIM / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    speed = np.sqrt(snap["uo"] ** 2 + snap["vo"] ** 2)
    speed.plot(ax=axes[0], cmap="viridis")
    axes[0].set_title("Current speed (m/s)")

    snap["vorticity"].plot(ax=axes[1], cmap="RdBu_r", robust=True)
    axes[1].set_title("Vorticity (1/s)")

    snap["okubo_weiss"].plot(ax=axes[2], cmap="PuOr", robust=True)
    axes[2].set_title("Okubo-Weiss (eddy < 0 < strain)")

    out = figdir / f"cube_snapshot_t{t_index}.png"
    fig.savefig(out, dpi=120)
    print(f"[sanity] wrote {out}")
    return out


if __name__ == "__main__":
    plot_snapshot()
