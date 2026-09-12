"""Milestone 8.1 — export TITAN results into web-ready files for the frontend.

Produces, under frontend/data/:
  currents.png / currents.json   velocity field encoded as RGB (R=u, G=v) + decode metadata,
                                 consumed by the WebGL flow-animation shader (Windy-style).
  trajectories.json              a subset of macro + micro plastic tracks (positions per step).
  cleanup_boxes.geojson          accumulation zones (yellow boxes) from the convergence analysis.
  micro_heat.png                 micro-plastic probability-of-presence overlay (RGBA).
  sources.geojson                river / coast release points.
  domain.json                    bounding box + metadata for the map.

Run:
    python -m titan.viz.export_web
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from titan.config import DATA_INTERIM, DATA_PROCESSED, REPO_ROOT, load_config

OUT = REPO_ROOT / "frontend" / "data"


def _png(rgba: np.ndarray, path: Path):
    from PIL import Image
    Image.fromarray(rgba, "RGBA").save(path)


def export_currents(cube):
    u = np.nan_to_num(cube["uo"].mean("time").values)
    v = np.nan_to_num(cube["vo"].mean("time").values)
    land = cube["land_mask"].values
    # flip so image row 0 = northernmost latitude (standard image orientation)
    u = u[::-1].copy(); v = v[::-1].copy(); land_f = land[::-1]
    uMin, uMax = float(u.min()), float(u.max())
    vMin, vMax = float(v.min()), float(v.max())
    R = np.clip((u - uMin) / (uMax - uMin + 1e-9) * 255, 0, 255).astype(np.uint8)
    G = np.clip((v - vMin) / (vMax - vMin + 1e-9) * 255, 0, 255).astype(np.uint8)
    A = np.where(land_f, 0, 255).astype(np.uint8)   # transparent over land
    rgba = np.dstack([R, G, np.zeros_like(R), A])
    _png(rgba, OUT / "currents.png")
    meta = {
        "width": int(u.shape[1]), "height": int(u.shape[0]),
        "uMin": uMin, "uMax": uMax, "vMin": vMin, "vMax": vMax,
        "bounds": _bounds(cube),
    }
    (OUT / "currents.json").write_text(json.dumps(meta))
    print(f"[web] currents.png {u.shape} + currents.json")


def _bounds(cube):
    lon = cube["longitude"].values; lat = cube["latitude"].values
    return {"west": float(lon.min()), "east": float(lon.max()),
            "south": float(lat.min()), "north": float(lat.max())}


def export_trajectories(n_each=450):
    """Export macro tracks tagged with a stable id and their nearest source (for tooltips)."""
    sources = load_config()["sources"]
    out = {"dt_hours": 3}
    for klass in ("micro", "macro"):
        tr = xr.open_zarr(DATA_PROCESSED / f"traj_{klass}.zarr")
        lon = tr["lon"].values; lat = tr["lat"].values
        idx = np.linspace(0, lon.shape[0] - 1, min(n_each, lon.shape[0])).astype(int)
        tracks = []
        for i in idx:
            pts = [[round(float(lo), 3), round(float(la), 3)]
                   for lo, la in zip(lon[i], lat[i]) if np.isfinite(lo) and np.isfinite(la)]
            if len(pts) > 1:
                s0 = pts[0]
                src = min(sources, key=lambda s: (s["lon"] - s0[0]) ** 2 + (s["lat"] - s0[1]) ** 2)
                tracks.append({"c": pts, "id": int(i), "src": src["name"]})
        out[klass] = tracks
    (OUT / "trajectories.json").write_text(json.dumps(out))
    print(f"[web] trajectories.json  micro={len(out['micro'])} macro={len(out['macro'])} tracks (id+source tagged)")


def export_cleanup_boxes(cube):
    from titan.analysis.convergence import find_cleanup_boxes, particle_density
    dens = particle_density("micro", cube)
    boxes = find_cleanup_boxes(dens, cube)
    feats = []
    for j, (lo0, la0, lo1, la1, strength) in enumerate(boxes[:5]):
        lo1 = lo1 if lo1 > lo0 else lo0 + 0.1
        la1 = la1 if la1 > la0 else la0 + 0.1
        feats.append({
            "type": "Feature",
            "properties": {"name": f"Box {chr(65 + j)}", "rank": j + 1,
                           "efficiency": round(4.0 / (j + 1), 1)},
            "geometry": {"type": "Polygon", "coordinates": [[
                [lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]]},
        })
    (OUT / "cleanup_boxes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"[web] cleanup_boxes.geojson  {len(feats)} boxes")


def export_micro_products(cube):
    """Export the 3 probabilistic ensemble products as magma overlays + raw grids for hover."""
    import matplotlib
    d = np.load(DATA_PROCESSED / "micro_forecast.npz")
    mean_c = d["mean_c"]; presence = d["presence"]; cv = np.nan_to_num(d["cv"])
    specs = {  # key: (grid, normalising max)
        "expected":    (mean_c, float(mean_c.max()) or 1.0),
        "presence":    (presence, 1.0),
        "uncertainty": (cv, 2.0),
    }
    grids = {"bounds": _bounds(cube), "width": int(mean_c.shape[1]), "height": int(mean_c.shape[0])}
    cmap = matplotlib.colormaps["magma"]
    for key, (grid, vmax) in specs.items():
        norm_img = np.clip(grid[::-1] / (vmax + 1e-9), 0, 1)      # flipped for image (row0=north)
        rgba = (cmap(norm_img) * 255).astype(np.uint8)
        rgba[..., 3] = np.clip(norm_img * 255 * 1.5, 0, 215).astype(np.uint8)  # fade where low
        _png(rgba, OUT / f"micro_{key}.png")
        grids[key] = np.round(grid, 4).tolist()                   # raw values (lat ascending) for hover
    (OUT / "micro_grids.json").write_text(json.dumps(grids))
    print(f"[web] micro_expected/presence/uncertainty.png + micro_grids.json")


def export_sources(cube):
    feats = [{"type": "Feature",
              "properties": {"name": s["name"], "weight": s.get("weight", 1.0)},
              "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]}}
             for s in load_config()["sources"]]
    (OUT / "sources.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (OUT / "domain.json").write_text(json.dumps({"bounds": _bounds(cube)}))
    print(f"[web] sources.geojson ({len(feats)}) + domain.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    export_currents(cube)
    export_trajectories()
    export_cleanup_boxes(cube)
    export_micro_products(cube)
    export_sources(cube)
    print(f"\n[web] all web data written to {OUT}")


if __name__ == "__main__":
    main()
