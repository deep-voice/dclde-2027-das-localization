"""Fetch individual sample objects from the public DCLDE bucket (no gcloud/auth needed).
Used by the EDA notebooks to pull a few real files into a local, gitignored cache.

    from scripts.fetch_sample import fetch
    p = fetch("dclde_2027_das-finwhale-localization/geometry/north_DAS_multicoord.csv")
"""
from __future__ import annotations
import os
import urllib.parse
import urllib.request

BUCKET = "noaa-passive-bioacoustic"
ROOT = "dclde/2027"
CACHE = os.environ.get("DCLDE_SAMPLE_DIR", "data/_samples")


def fetch(rel_path: str, cache_dir: str = CACHE) -> str:
    """Download `<ROOT>/<rel_path>` to <cache_dir>/<rel_path> (skip if present). Returns local path."""
    obj = f"{ROOT}/{rel_path}"
    local = os.path.join(cache_dir, rel_path)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local) or os.path.getsize(local) == 0:
        url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(obj)}"
        urllib.request.urlretrieve(url, local)
    return local
