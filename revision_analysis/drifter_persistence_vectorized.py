"""Fast, vectorized version of drifter_persistence_fix.py: advects all 21 drifters as a
SINGLE ParticleSet (like the rest of this pipeline does for thousands of particles), instead
of one particle at a time, which was extremely slow due to per-drifter kernel/compile
overhead repeated across many small pset.execute() calls. Each particle is released at its
own drifter's start time (Parcels supports per-particle release times natively).
"""
import datetime
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_INTERIM, DATA_PROCESSED
from titan.analysis.validate_drifters import fetch_drifters, _build_fieldset, MAX_DAYS
from titan.physics import kernels as K

DEG_M = 111_320.0


def main():
    from parcels import AdvectionRK4, JITParticle, ParticleSet

    cube = xr.open_zarr(DATA_INTERIM / "val_cube.zarr")
    t0_cube = cube["time"].values[0]
    fieldset = _build_fieldset()
    drifters = fetch_drifters()

    step_h = 6
    nsteps = MAX_DAYS * 24 // step_h

    drifter_ids, starts, lons0, lats0 = [], [], [], []
    real_lookup = {}  # did -> (ts, lo, la)
    for did, (ts, lo, la) in drifters.items():
        start = max(ts[0], t0_cube)
        if start >= ts[-1]:
            continue
        i0 = int(np.argmin(np.abs(ts - start)))
        drifter_ids.append(did)
        starts.append(start.astype("datetime64[s]").astype(object))
        lons0.append(lo[i0]); lats0.append(la[i0])
        real_lookup[did] = (ts, lo, la)

    n = len(drifter_ids)
    print(f"[vec] releasing {n} drifters in one ParticleSet")

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                        lon=lons0, lat=lats0, time=starts)
    kern = pset.Kernel(AdvectionRK4) + pset.Kernel(K.delete_oob)

    # record virtual position at each of nsteps checkpoints (absolute times from t0_cube)
    virt_lon = np.full((n, nsteps), np.nan)
    virt_lat = np.full((n, nsteps), np.nan)
    # map original particle id -> row (Parcels may reorder/remove on deletion, so track by pid)
    pid_to_row = {int(p.id): i for i, p in enumerate(pset)}

    t_end = max(starts) + datetime.timedelta(hours=int(nsteps * step_h))
    checkpoints = [starts[0] + datetime.timedelta(hours=step_h * (k + 1)) for k in range(nsteps)]
    # Simpler: step forward in fixed 6h increments from the EARLIEST start, recording whichever
    # particles are alive and past their own release time at each checkpoint.
    cur_time = min(starts)
    for k in range(nsteps):
        target_time = cur_time + datetime.timedelta(hours=step_h)
        pset.execute(kern, endtime=target_time, dt=datetime.timedelta(minutes=20), verbose_progress=False)
        for p in pset:
            row = pid_to_row.get(int(p.id))
            if row is not None:
                virt_lon[row, k] = p.lon
                virt_lat[row, k] = p.lat
        cur_time = target_time
        print(f"[vec] step {k+1}/{nsteps} done, {pset.size}/{n} particles active", flush=True)

    out = DATA_PROCESSED / "drifter_validation_vectorized.npz"
    np.savez(out, virt_lon=virt_lon, virt_lat=virt_lat, drifter_ids=np.array(drifter_ids),
             starts=np.array([str(s) for s in starts]), nsteps=nsteps, step_h=step_h)
    print(f"[vec] wrote {out}")


if __name__ == "__main__":
    main()
