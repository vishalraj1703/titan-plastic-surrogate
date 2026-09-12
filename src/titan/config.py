"""Load the single-source-of-truth run configuration (config/region.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/titan/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "region.yaml"

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Return the parsed region.yaml as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Region: {cfg['region']['name']}")
    print(
        f"  box: lon [{cfg['region']['lon_min']}, {cfg['region']['lon_max']}]  "
        f"lat [{cfg['region']['lat_min']}, {cfg['region']['lat_max']}]"
    )
    print(f"  UTM EPSG: {cfg['region']['utm_epsg']}")
    print(f"  currents dataset: {cfg['copernicus']['currents_dataset']}")
