"""Reviewer major comment #5 (round 2): the original benchmark compared GPU-batched surrogate
inference against single-threaded CPU OceanParcels -- not an intrinsic speed comparison. This
adds CPU-only surrogate timings at the same particle scales, so the GPU-vs-CPU-implementation
framing is explicit and a same-hardware (CPU vs CPU) comparison is also available.
"""
import time
import numpy as np
import torch
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.models.dataset import CUBE_FIELDS, DT_SEC
from titan.models.surrogate import ResidualMLP

SCALES = [5000, 15000, 30000, 50000]
REPEATS = 3
HORIZON_H = 72
STEPS = HORIZON_H // 3
DEG_M = 111_320.0


def time_surrogate(n, cube, model, mu, sd, device, seed=1):
    rng = np.random.default_rng(seed)
    lon = rng.normal(4.85, 0.05, n)
    lat = rng.normal(43.33, 0.05, n)
    windage = rng.uniform(0.05, 0.10, n).astype(np.float32)
    is_macro = np.ones(n, np.float32)
    size_class = rng.integers(3, 6, n).astype(np.float32)
    age = np.zeros(n, np.float64)
    t0_time = cube["time"].values[0]

    t0 = time.perf_counter()
    for k in range(STEPS):
        t = t0_time + np.timedelta64(3 * k, "h")
        pts = {"time": xr.DataArray(np.full(n, t)), "latitude": xr.DataArray(lat), "longitude": xr.DataArray(lon)}
        samp = {f: np.nan_to_num(cube[f].interp(**pts, method="linear").values, nan=0.0) for f in CUBE_FIELDS}
        X = np.column_stack([samp["uo"], samp["vo"], samp["vorticity"], samp["okubo_weiss"],
                              samp["u10"], samp["v10"], samp["wind_speed"], age, windage, is_macro, size_class]).astype(np.float32)
        with torch.no_grad():
            res = model(torch.tensor((X - mu) / sd).to(device)).cpu().numpy()
        vel_u, vel_v = samp["uo"] + res[:, 0], samp["vo"] + res[:, 1]
        coslat = np.cos(np.deg2rad(lat))
        lon = lon + vel_u * DT_SEC / (DEG_M * coslat)
        lat = lat + vel_v * DT_SEC / DEG_M
        age += DT_SEC
    return time.perf_counter() - t0


def main():
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    ck = torch.load(DATA_PROCESSED / "surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])

    print("=== Surrogate timing: GPU (RTX 3050) vs CPU-only, same model, same interpolation cost ===\n")
    print(f"{'N particles':>12} | {'GPU (s)':>20} | {'CPU (s)':>20} | GPU speedup over CPU-surrogate")
    print("-" * 90)
    for n in SCALES:
        row = {}
        for device_name in ("cuda", "cpu"):
            model = ResidualMLP(len(feats)).to(device_name)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)
            times = [time_surrogate(n, cube, model, mu, sd, device_name, seed=r) for r in range(REPEATS)]
            row[device_name] = (np.median(times), np.std(times))
        gm, gs = row["cuda"]; cm, cs = row["cpu"]
        print(f"{n:>12} | {gm:6.2f} (sd {gs:4.2f})     | {cm:6.2f} (sd {cs:4.2f})     | {cm/gm:.2f}x")


if __name__ == "__main__":
    main()
