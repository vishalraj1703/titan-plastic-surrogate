"""Compute current-advection separation and persistence separation for all 21 drifters from
the vectorized run, plus paired per-drifter bootstrap CIs (fixes the earlier bug where the
persistence baseline was silently computed from only 3 of 21 drifters).
"""
import numpy as np
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.config import DATA_PROCESSED
from titan.analysis.validate_drifters import fetch_drifters

DEG_M = 111_320.0


def sep_km(lo1, la1, lo2, la2):
    dx = (lo1 - lo2) * DEG_M * np.cos(np.deg2rad(la2))
    dy = (la1 - la2) * DEG_M
    return np.sqrt(dx**2 + dy**2) / 1000.0


def main():
    d = np.load(DATA_PROCESSED / "drifter_validation_vectorized.npz", allow_pickle=True)
    virt_lon, virt_lat = d["virt_lon"], d["virt_lat"]
    drifter_ids = d["drifter_ids"]
    starts = [np.datetime64(s) for s in d["starts"]]
    nsteps, step_h = int(d["nsteps"]), int(d["step_h"])
    n = len(drifter_ids)

    drifters = fetch_drifters()

    curr_sep = np.full((n, nsteps), np.nan)
    pers_sep = np.full((n, nsteps), np.nan)

    for i, did in enumerate(drifter_ids):
        ts, lo, la = drifters[str(did)]
        start = starts[i]
        i0 = int(np.argmin(np.abs(ts - start)))
        lon0, lat0 = lo[i0], la[i0]
        for k in range(nsteps):
            tnow = start + np.timedelta64(step_h * (k + 1), "h")
            if tnow > ts[-1]:
                break
            j = int(np.argmin(np.abs(ts - tnow)))
            real_lon, real_lat = lo[j], la[j]
            if np.isfinite(virt_lon[i, k]):
                curr_sep[i, k] = sep_km(virt_lon[i, k], virt_lat[i, k], real_lon, real_lat)
            pers_sep[i, k] = sep_km(lon0, lat0, real_lon, real_lat)

    np.savez(DATA_PROCESSED / "drifter_validation_final.npz",
              curr_sep=curr_sep, pers_sep=pers_sep, drifter_ids=drifter_ids)

    print("=== Corrected: current-advection vs persistence, ALL drifters, paired per-drifter ===\n")
    rng = np.random.default_rng(0)
    for day in (1, 3, 7, 14):
        idx = day * 24 // step_h - 1
        c, p = curr_sep[:, idx], pers_sep[:, idx]
        valid = np.isfinite(c) & np.isfinite(p)
        nv = int(valid.sum())
        cc, pp = c[valid], p[valid]
        diff = cc - pp
        point = np.median(diff)
        boots = np.array([np.median(diff[rng.integers(0, nv, nv)]) for _ in range(2000)])
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        print(f"day {day:2d}: n={nv}  advection median={np.median(cc):.1f}km  persistence median={np.median(pp):.1f}km  "
              f"paired diff (adv-pers)={point:+.1f}km  95% CI [{lo_ci:+.1f}, {hi_ci:+.1f}]  n_worse_for_advection={int((diff>0).sum())}/{nv}")


if __name__ == "__main__":
    main()
