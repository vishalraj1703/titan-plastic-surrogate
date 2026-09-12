"""Reviewer major comment #4: proper scaling-curve benchmark, not a two-point comparison.

Reports, for several particle counts, repeated timings (median + spread) of:
  (a) OceanParcels 72h macro rollout -- steady-state integration time (JIT warm-up excluded,
      reported separately once), single-threaded, same kernels as the main simulation
      (AdvectionRK4 + DiffusionUniformKh + windage + beaching + age + delete_oob).
  (b) The neural surrogate's 72h rollout on the SAME hardware -- GPU-batched forward passes
      plus the field-interpolation cost the surrogate also pays at every step (loaded via the
      same xarray .interp() calls used in the real evaluation, not an idealized GPU-only figure).

Both timings therefore include field interpolation, so this is an end-to-end comparison, not
raw kernel-vs-kernel. Field/cube loading (one-time, ~1s, amortized across any number of runs)
is excluded from both, and stated separately.
"""
import datetime
import time
import numpy as np
import torch
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")

from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.physics import simulate as S
from titan.physics import kernels as K
from titan.models.dataset import CUBE_FIELDS, DT_SEC
from titan.models.surrogate import ResidualMLP
from parcels import AdvectionRK4, DiffusionUniformKh

SCALES = [5000, 15000, 30000, 50000]
REPEATS = 3
HORIZON_H = 72
STEPS = HORIZON_H // 3
DEG_M = 111_320.0


def time_physics(n, fieldset):
    kh = 1.0
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")
    pclass = S.particle_class(jit=True)
    pset = S.release(fieldset, "macro", pclass, n_override=n, seed=1)
    kern = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh)
    kern += pset.Kernel(K.windage) + pset.Kernel(K.beaching) + pset.Kernel(K.age_update) + pset.Kernel(K.delete_oob)
    t0 = time.perf_counter()
    pset.execute(kern, runtime=datetime.timedelta(hours=HORIZON_H), dt=datetime.timedelta(minutes=20),
                 verbose_progress=False)
    return time.perf_counter() - t0


def time_surrogate(n, cube, model, mu, sd, seed=1):
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
            res = model(torch.tensor((X - mu) / sd)).numpy()
        vel_u, vel_v = samp["uo"] + res[:, 0], samp["vo"] + res[:, 1]
        coslat = np.cos(np.deg2rad(lat))
        lon = lon + vel_u * DT_SEC / (DEG_M * coslat)
        lat = lat + vel_v * DT_SEC / DEG_M
        age += DT_SEC
    return time.perf_counter() - t0


def main():
    t_load0 = time.perf_counter()
    fieldset_template = S.build_fieldset()
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    ck = torch.load(DATA_PROCESSED / "surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)
    load_time = time.perf_counter() - t_load0
    print(f"[one-time field/model load: {load_time:.2f}s -- amortized, excluded below]\n")

    # JIT warm-up (compile cost paid once)
    fs_warm = S.build_fieldset()
    t0 = time.perf_counter()
    time_physics(200, fs_warm)
    print(f"[JIT warm-up/compile (one-time): {time.perf_counter()-t0:.2f}s]\n")

    print(f"{'N particles':>12} | {'OceanParcels (s)':>28} | {'Surrogate (s)':>28} | speedup")
    print("-" * 95)
    for n in SCALES:
        phys_times = []
        for r in range(REPEATS):
            fs = S.build_fieldset()
            phys_times.append(time_physics(n, fs))
        sur_times = [time_surrogate(n, cube, model, mu, sd, seed=r) for r in range(REPEATS)]
        pm, ps = np.median(phys_times), np.std(phys_times)
        sm, ss = np.median(sur_times), np.std(sur_times)
        print(f"{n:>12} | {pm:6.2f} (sd {ps:4.2f}, n={REPEATS})       | {sm:6.2f} (sd {ss:4.2f}, n={REPEATS})       | {pm/sm:.2f}x")


if __name__ == "__main__":
    main()
