"""Reviewer major comment #4 (round 3): a per-drifter audit trail for the corrected
persistence analysis -- drifter IDs, inclusion rule, and per-drifter advection/persistence/
paired-difference values at each horizon, so the corrected result is independently checkable.
"""
import csv
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
    drifters = fetch_drifters()

    rows = []
    for i, did in enumerate(drifter_ids):
        ts, lo, la = drifters[str(did)]
        start = starts[i]
        i0 = int(np.argmin(np.abs(ts - start)))
        lon0, lat0 = lo[i0], la[i0]
        n_fixes = len(ts)
        track_span_days = float((ts[-1] - ts[0]) / np.timedelta64(1, "D"))
        for day in (1, 3, 7, 14):
            k = day * 24 // step_h - 1
            tnow = start + np.timedelta64(step_h * (k + 1), "h")
            if tnow > ts[-1] or not np.isfinite(virt_lon[i, k]):
                continue
            j = int(np.argmin(np.abs(ts - tnow)))
            real_lon, real_lat = lo[j], la[j]
            adv = sep_km(virt_lon[i, k], virt_lat[i, k], real_lon, real_lat)
            pers = sep_km(lon0, lat0, real_lon, real_lat)
            rows.append({
                "drifter_id": str(did), "horizon_days": day,
                "release_time": str(start), "n_hourly_fixes": n_fixes,
                "track_span_days": round(track_span_days, 2),
                "advection_km": round(float(adv), 2),
                "persistence_km": round(float(pers), 2),
                "paired_diff_adv_minus_pers_km": round(float(adv - pers), 2),
            })

    out = DATA_PROCESSED / "drifter_audit_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[audit] wrote {out} ({len(rows)} rows, {len(drifter_ids)} drifters)")

    print("\nInclusion rule: a drifter contributes to horizon H if (a) it has >=3 days of hourly")
    print("GPS fixes within the July 2022 NW-Mediterranean window (ID/time/lon/lat from NOAA")
    print("Global Drifter Program via ERDDAP, drifter_hourly_qc.csv), and (b) its track extends")
    print("to at least H days after its release time (max(first fix, forcing-cube start)).")


if __name__ == "__main__":
    main()
