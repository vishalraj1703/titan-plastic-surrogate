"""Generate OceanParcels ground-truth trajectories under JULY 2022 forcing (val_cube_full),
using the exact same release/kernel logic as titan.physics.simulate, just pointed at a
different cube. This is the "unseen forcing" ground truth against which we test the
surrogate (trained only on January 2024) for genuine environmental generalization.
"""
import datetime
import sys
sys.path.insert(0, r"E:\oceanography\src")
import xarray as xr
from parcels import AdvectionRK4, DiffusionUniformKh, FieldSet
from titan.config import DATA_INTERIM, DATA_PROCESSED, load_config
from titan.physics import simulate as S
from titan.physics import kernels as K

N_PER_CLASS = 2000  # smaller than the main 5000/class run -- sufficient for a generalization check


def build_val_fieldset():
    cube = xr.open_zarr(DATA_INTERIM / "val_cube_full.zarr")
    variables = {"U": "uo", "V": "vo", "land_mask": "land_mask", "u10": "u10", "v10": "v10"}
    dimensions = {
        "U": {"lon": "longitude", "lat": "latitude", "time": "time"},
        "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
        "land_mask": {"lon": "longitude", "lat": "latitude"},
        "u10": {"lon": "longitude", "lat": "latitude", "time": "time"},
        "v10": {"lon": "longitude", "lat": "latitude", "time": "time"},
    }
    fs = FieldSet.from_xarray_dataset(cube, variables=variables, dimensions=dimensions,
                                       mesh="spherical", allow_time_extrapolation=True)
    fs.has_wind = True
    return fs


def run(klass, seed=99):
    cfg = load_config()
    pc = cfg["particles"][klass]
    fieldset = build_val_fieldset()
    kh = float(sum(pc["diffusivity_kh"]) / 2)
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")
    pclass = S.particle_class(jit=True)
    pset = S.release(fieldset, klass, pclass, n_override=N_PER_CLASS, seed=seed)
    kern = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh)
    kern += pset.Kernel(K.windage) + pset.Kernel(K.beaching) + pset.Kernel(K.age_update) + pset.Kernel(K.delete_oob)
    out = DATA_PROCESSED / f"traj_{klass}_val2022.zarr"
    pfile = pset.ParticleFile(name=str(out), outputdt=datetime.timedelta(hours=3))
    pset.execute(kern, runtime=datetime.timedelta(days=10), dt=datetime.timedelta(minutes=20),
                 output_file=pfile)
    print(f"[simulate_val] {klass}: released {N_PER_CLASS}, survived {pset.size}, wrote {out}")


if __name__ == "__main__":
    run("micro")
    run("macro")
