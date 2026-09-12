"""Milestone 6 — probabilistic multi-source micro-plastic forecast.

Micro-plastic drifts diffusively, so any single simulation is just one random realisation.
An honest forecast is therefore an ENSEMBLE: we run the multi-source micro release many times
with independent diffusion, and summarise the spread. We report three maps:

  1. Expected concentration  = ensemble-mean particle density per cell (the forecast).
  2. Probability of presence = fraction of realisations with plastic in that cell (0..1).
  3. Uncertainty            = coefficient of variation (std / mean) of the density.

This replaces "one dot per particle" with "here is the probability plastic is present, and how
confident we are" — the correct, honest output for an inherently stochastic quantity.

Run:
    python -m titan.analysis.probabilistic
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from parcels import AdvectionRK4, DiffusionUniformKh

from titan.config import DATA_INTERIM, DATA_PROCESSED, load_config
from titan.physics import kernels as K
from titan.physics import simulate as S

N_ENSEMBLE = 8
N_PARTICLES = 2500
RUNTIME_DAYS = 10


def _one_realisation(seed: int, lon_edges, lat_edges) -> np.ndarray:
    """Run one micro ensemble member; return the final-position density grid (lat, lon)."""
    import datetime

    fieldset = S.build_fieldset()
    kh = float(np.mean(load_config()["particles"]["micro"]["diffusivity_kh"]))
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")

    pclass = S.particle_class(jit=True)
    pset = S.release(fieldset, "micro", pclass, n_override=N_PARTICLES, seed=seed)

    kern = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh)
    if getattr(fieldset, "has_wind", False):
        kern += pset.Kernel(K.windage)
    kern += pset.Kernel(K.beaching) + pset.Kernel(K.age_update) + pset.Kernel(K.delete_oob)

    pset.execute(kern, runtime=datetime.timedelta(days=RUNTIME_DAYS),
                 dt=datetime.timedelta(minutes=20))
    lon = np.array([p.lon for p in pset]); lat = np.array([p.lat for p in pset])
    H, _, _ = np.histogram2d(lon, lat, bins=[lon_edges, lat_edges])
    return H.T  # (lat, lon)


def forecast() -> str:
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    lon = cube["longitude"].values; lat = cube["latitude"].values
    dlon = np.abs(np.diff(lon)).mean(); dlat = np.abs(np.diff(lat)).mean()
    lon_edges = np.append(lon - dlon / 2, lon[-1] + dlon / 2)
    lat_edges = np.append(lat - dlat / 2, lat[-1] + dlat / 2)

    dens = np.stack([_one_realisation(s, lon_edges, lat_edges) for s in range(N_ENSEMBLE)])
    print(f"[prob] ran {N_ENSEMBLE} ensemble members")

    mean_c = dens.mean(0)
    presence = (dens > 0).mean(0)                       # probability of presence
    cv = dens.std(0) / (mean_c + 1e-6)                  # uncertainty
    cv[mean_c < mean_c.max() * 0.01] = np.nan           # undefined where ~no plastic

    out = DATA_PROCESSED / "micro_forecast.npz"
    np.savez(out, mean_c=mean_c, presence=presence, cv=cv,
             lon=lon, lat=lat)
    print(f"[prob] wrote {out}")
    print(f"[prob]   cells with >50% presence probability: {int((presence > 0.5).sum())}")
    return str(out)


if __name__ == "__main__":
    forecast()
