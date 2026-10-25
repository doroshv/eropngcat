"""Paths and global constants shared by all pipeline steps."""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

IMAGE = ROOT / "erass1_allsky.png"
ERASS1_FITS = ROOT / "eRASS1_Main.v1.2.fits"
XMMSL_FITS = ROOT / "xmmsl3_v1.0.fits"
# 8 GB release; after step 01 only data/xmm_dr15_bright.parquet is needed and this can be deleted.
XMM_DR15_FITS = ROOT / "5XMM_DR15cat_v1.0.fits"

SEED = 20260916

# Worker processes: each holds the full image (~0.5 GB with working arrays); 6 fits in 16 GB RAM.
WORKERS = int(os.environ.get("EPNGCAT_WORKERS", min(6, os.cpu_count() or 1)))

# eRASS1-DE footprint in Galactic longitude (taken from the DR1 catalogue extent).
L_SPLIT = 179.944

# Catalogue bands. RGB of the image is 0.3-0.6 / 0.6-1.0 / 1.0-2.3 keV (PR caption);
# closest catalogue bands are P1 0.2-0.5, P2 0.5-1.0, P3 1.0-2.0 keV; band 1 is 0.2-2.3 keV.
BANDS = ["1", "P1", "P2", "P3"]


def is_german(lii):
    """Return True where Galactic longitude lies in the eRASS1-DE half."""
    return ((np.asarray(lii) - L_SPLIT) % 360.0) < 180.0


def step_dir(name):
    """Return (and create) the output directory of a pipeline step."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    """Write a JSON file with stable formatting."""
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def load_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text())
