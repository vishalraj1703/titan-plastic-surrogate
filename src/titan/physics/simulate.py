"""Milestone 2 — OceanParcels forward simulations that generate GNN training data.

Reads the interim cube (Milestone 1), releases macro OR micro particles (run separately),
applies advection + class-uniform diffusion + beaching, and saves trajectories to zarr.

Behavioural classes (blueprint Phase 2 macro/micro update) — per config/region.yaml:
  micro: high diffusivity (10-100 m^2/s), low windage    -> swirls with eddies
  macro: low diffusivity (~1 m^2/s),      high windage    -> moves as a coherent unit
(Windage + inertial damping arrive in Milestone 2b with the wind product.)

Run (after the cube exists):
    python -m titan.physics.simulate            # full macro + micro runs
    python -m titan.physics.simulate --smoke    # tiny fast run to sanity-check
"""
from __future__ import annotations

import argparse
import datetime

import numpy as np
import xarray as xr
from parcels import (
    AdvectionRK4,
    DiffusionUniformKh,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    Variable,
)

from titan.config import DATA_INTERIM, DATA_PROCESSED, load_config
from titan.physics import kernels as K

# Extra per-particle Variables carried through the sim. kh is NOT here: it is uniform per
# run and supplied to Parcels as a constant field so the built-in diffusion kernel works.
_EXTRA_VARS = [
    Variable("windage", dtype=np.float32),      # C_w (feature; kernel use in M2b)
    Variable("is_macro", dtype=np.int32),
    Variable("size_class", dtype=np.int32),
    Variable("age", dtype=np.float32, initial=0.0),
    Variable("status", dtype=np.int32, initial=0),
]


def particle_class(jit: bool = True):
    """Build the plastic particle class on a JIT (fast, needs C compiler) or Scipy base."""
    base = JITParticle if jit else ScipyParticle
    return base.add_variable(_EXTRA_VARS)


def build_fieldset() -> FieldSet:
    """FieldSet from the interim zarr cube: currents (U,V) + land mask, spherical mesh."""
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    variables = {"U": "uo", "V": "vo", "land_mask": "land_mask"}
    dimensions = {
        "U": {"lon": "longitude", "lat": "latitude", "time": "time"},
        "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
        "land_mask": {"lon": "longitude", "lat": "latitude"},
    }
    # Wind fields (u10/v10) are added WITHOUT a unit converter so kernels sample raw m/s.
    has_wind = "u10" in cube
    if has_wind:
        variables["u10"] = "u10"
        variables["v10"] = "v10"
        dimensions["u10"] = {"lon": "longitude", "lat": "latitude", "time": "time"}
        dimensions["v10"] = {"lon": "longitude", "lat": "latitude", "time": "time"}
    fs = FieldSet.from_xarray_dataset(
        cube,
        variables=variables,
        dimensions=dimensions,
        mesh="spherical",
        allow_time_extrapolation=True,
    )
    fs.has_wind = has_wind
    return fs


def release(fieldset, klass: str, pclass, n_override: int | None = None, seed: int = 0):
    """Release a cloud of one behavioural class from all configured sources."""
    cfg = load_config()
    pc = cfg["particles"][klass]
    sources = cfg["sources"]
    rng = np.random.default_rng(seed)

    n = n_override if n_override is not None else pc["n_particles"]
    # split particles across sources by emission weight (equal if no weights given)
    weights = np.array([src.get("weight", 1.0) for src in sources], float)
    weights = weights / weights.sum()
    counts = np.maximum(1, np.round(n * weights).astype(int))

    lons, lats = [], []
    for src, cnt in zip(sources, counts):
        lons.append(rng.normal(src["lon"], 0.05, cnt))  # ~5 km release blob
        lats.append(rng.normal(src["lat"], 0.05, cnt))
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    m = lon.size

    wind = rng.uniform(*pc["windage_coeff"], m).astype(np.float32)
    sc = rng.integers(pc["size_class"][0], pc["size_class"][1] + 1, m).astype(np.int32)

    return ParticleSet(
        fieldset=fieldset,
        pclass=pclass,
        lon=lon,
        lat=lat,
        windage=wind,
        is_macro=np.full(m, pc["is_macro"], np.int32),
        size_class=sc,
    )


def run(klass: str = "micro", runtime_days: int = 10, jit: bool = True,
        n_override: int | None = None, seed: int = 0, tag: str = "") -> str:
    """Run one forward simulation for a behavioural class; save trajectories to zarr.

    `tag` appends a suffix to the output filename (e.g. tag='test' -> traj_micro_test.zarr)
    so held-out test sets don't overwrite the training trajectories.
    """
    cfg = load_config()
    pc = cfg["particles"][klass]

    fieldset = build_fieldset()
    # Class-uniform diffusivity as a constant field (m^2/s -> deg via spherical mesh).
    kh = float(np.mean(pc["diffusivity_kh"]))
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")

    pclass = particle_class(jit=jit)
    pset = release(fieldset, klass, pclass, n_override=n_override, seed=seed)

    kern = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh)
    if getattr(fieldset, "has_wind", False):
        kern += pset.Kernel(K.windage)  # wind-driven drift (macro >> micro)
    kern += pset.Kernel(K.beaching) + pset.Kernel(K.age_update) + pset.Kernel(K.delete_oob)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out = DATA_PROCESSED / f"traj_{klass}{suffix}.zarr"
    pfile = pset.ParticleFile(name=str(out), outputdt=datetime.timedelta(hours=3))

    pset.execute(
        kern,
        runtime=datetime.timedelta(days=runtime_days),
        dt=datetime.timedelta(minutes=20),
        output_file=pfile,
    )
    print(f"[simulate] {klass}: released {pset.size} survived, wrote {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to sanity-check")
    ap.add_argument("--scipy", action="store_true", help="use ScipyParticle (no C compiler)")
    args = ap.parse_args()

    jit = not args.scipy
    if args.smoke:
        run("micro", runtime_days=2, jit=jit, n_override=200)
    else:
        run("micro", runtime_days=10, jit=jit)
        run("macro", runtime_days=10, jit=jit)
