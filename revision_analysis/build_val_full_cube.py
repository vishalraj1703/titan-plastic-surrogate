"""Reviewer major comment #1/#2: unseen-forcing generalization test.

val_cube.zarr (built by titan.analysis.validate_drifters for the GPS-drifter validation)
covers July 2022 -- a different SEASON and YEAR from the January 2024 training/test window,
same spatial domain. It only has raw uo/vo/u10/v10/land_mask; this script adds the derived
vorticity / Okubo-Weiss / wind_speed fields using the EXACT same formulas as
titan.data.preprocess.build_cube(), so the surrogate sees feature-identical inputs to what
it was trained on -- the only thing that changes is the actual physical forcing.
"""
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_INTERIM

EARTH_R = 6_371_000.0
DEG2M = np.pi * EARTH_R / 180.0


def main():
    cube = xr.open_zarr(DATA_INTERIM / "val_cube.zarr")
    u, v = cube["uo"], cube["vo"]

    dlat = float(np.abs(np.diff(cube["latitude"].values)).mean())
    dlon = float(np.abs(np.diff(cube["longitude"].values)).mean())
    dy = dlat * DEG2M

    du_dy = u.differentiate("latitude") / DEG2M
    dv_dx = v.differentiate("longitude") / (DEG2M * np.cos(np.deg2rad(cube["latitude"])))
    du_dx = u.differentiate("longitude") / (DEG2M * np.cos(np.deg2rad(cube["latitude"])))
    dv_dy = v.differentiate("latitude") / DEG2M

    vorticity = (dv_dx - du_dy).rename("vorticity")
    s_n = du_dx - dv_dy
    s_s = dv_dx + du_dy
    okubo_weiss = (s_n ** 2 - vorticity ** 2 + s_s ** 2).rename("okubo_weiss")
    wind_speed = np.sqrt(cube["u10"] ** 2 + cube["v10"] ** 2).rename("wind_speed")

    full = xr.Dataset({
        "uo": u, "vo": v, "land_mask": cube["land_mask"],
        "u10": cube["u10"], "v10": cube["v10"],
        "vorticity": vorticity, "okubo_weiss": okubo_weiss, "wind_speed": wind_speed,
    })
    out = DATA_INTERIM / "val_cube_full.zarr"
    full.chunk({"time": 31}).to_zarr(out, mode="w")
    print(f"[val-full] wrote {out}  dims={dict(full.sizes)}")
    print(f"[val-full] time range: {full.time.values.min()} -> {full.time.values.max()}")


if __name__ == "__main__":
    main()
