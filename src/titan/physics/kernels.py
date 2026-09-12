"""Custom OceanParcels kernels (Parcels 3.1 API).

Parcels v3 convention: kernels write to the injected increment names `particle_dlon`,
`particle_dlat` (NOT particle.lon directly). Advection + diffusion use Parcels' own
built-ins (AdvectionRK4, DiffusionUniformKh) because they handle the m/s -> deg/s unit
conversion via the field UnitConverters correctly. We only hand-write the two behaviours
Parcels doesn't ship: beaching and age tracking.

Deferred to Milestone 2b (needs the separate CMEMS wind product, not yet downloaded):
  * Windage   V_wind = C_w * (rho_air/rho_water) * |U10| * U10   (macro >> micro)
  * Inertial damping (macro plastic lags the flow with relaxation time tau)
Until winds are ingested, macro vs micro differ through diffusivity (macro ~1 m^2/s vs
micro 10-100 m^2/s), which is a real and learnable behavioural contrast on its own.
"""
import math

from parcels import StatusCode

# NB: Parcels rebuilds kernels from their AST and does NOT capture module-level globals,
# so every constant must be defined *inside* the kernel body (hence the local `deg_m`).
# Local variable names must also not collide with FieldSet field names (u10/v10) or the
# JIT C-generator redeclares them — so wind samples go into w_u/w_v.


def windage(particle, fieldset, time):
    """Wind-driven surface drift (blueprint Phase 2, macro/micro update).

    Standard linear windage parameterisation used throughout the drift literature
    (e.g. Kubota 1994): V_drift = C_w * U10, where C_w is the per-particle windage
    coefficient (micro 1-3%, macro 5-10%). u10/v10 are sampled in m/s (added to the
    FieldSet without a unit converter) and converted to degree increments here.

    NB: this replaces the blueprint's quadratic C_w*(rho_air/rho_water)*|U10|*U10 form,
    which is not dimensionally a velocity. The linear form is the accepted standard and
    matches the C_w ranges in config/region.yaml. Density constants are retained in config
    for a future full quadratic drag closure if desired.
    """
    deg_m = 111320.0
    w_u = fieldset.u10[time, particle.depth, particle.lat, particle.lon]
    w_v = fieldset.v10[time, particle.depth, particle.lat, particle.lon]
    vx = particle.windage * w_u  # m/s
    vy = particle.windage * w_v
    coslat = math.cos(particle.lat * 3.141592653589793 / 180.0)
    particle_dlon += vx * particle.dt / (deg_m * coslat)  # noqa: F821
    particle_dlat += vy * particle.dt / deg_m  # noqa: F821


def beaching(particle, fieldset, time):
    """Flag + remove particles that drift onto a land cell (blueprint Phase 2.1)."""
    on_land = fieldset.land_mask[time, particle.depth, particle.lat, particle.lon]
    if on_land > 0.5:
        particle.status = 1  # 0 == active, 1 == beached
        particle.delete()


def age_update(particle, fieldset, time):
    """Advance particle age in seconds (a GNN training feature)."""
    particle.age += math.fabs(particle.dt)


def delete_oob(particle, fieldset, time):
    """Remove particles that leave the domain (windage can push them past the edge).

    Parcels flags an out-of-bounds sample by setting particle.state; we consume that
    error state by deleting the particle so the run continues instead of aborting.
    """
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()
    elif particle.state == StatusCode.ErrorInterpolation:
        particle.delete()
