"""High-resolution eRASS1-DE rate maps (MPE DR1 HiPS) resampled onto the tile TAN grids.

Source: https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/HalfSkyMaps_dr1/ (HiPS, Merloni+ 2024),
bands 025 (0.2-0.5 keV), 026 (0.5-1.0 keV), 027 (1.0-2.0 keV) ~ image R, G, B and catalogue P1-P3.
HiPS order 6, 512x512 FITS tiles, ICRS, 6.4" pixels, count rate per HiPS pixel. Lower HiPS orders
are built with a median hierarchy (which erases point sources), so only order-6 tiles are used and
are averaged here: each output pixel is the mean of SUB x SUB nearest-neighbour samples, which
conserves flux to within the sampling granularity.

Output grid: same tangent point and extent as the tile patch, HIRES_FACTOR times finer (pixel
centres of the patch coincide with every HIRES_FACTOR-th output pixel). Units: counts/s/arcmin^2.
Pixels outside the DE HiPS coverage are NaN.
"""

import io
import time
from functools import lru_cache

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy_healpix import HEALPix

from epngcat.patches import tan_inverse

BASE = "https://erosita.mpe.mpg.de/erodat/static/hips/eRASS1_{band}_Rate_c010"
BANDS = ["025", "026", "027"]
BAND_LABEL = {"025": "0.2-0.5 keV", "026": "0.5-1.0 keV", "027": "1.0-2.0 keV"}
ORDER = 6
TILE = 512
HIPS_PIX_ARCMIN = 0.0017893158394882323 * 60
HIRES_FACTOR = 4  # patch 2'/px -> 0.5'/px
SUB = 5  # samples per axis per output pixel (0.1' spacing ~ HiPS pixel)
RETRIES = 4

_HP6 = HEALPix(nside=2**ORDER, order="nested", frame="icrs")
_HP15 = HEALPix(nside=2 ** (ORDER + 9), order="nested", frame="icrs")
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


@lru_cache(maxsize=24)  # ~72 MB per worker; neighbouring tiles share HiPS tiles
def hips_tile(band, npix):
    """Order-6 HiPS tile as float32 (512, 512), or None if the tile does not exist (outside DE)."""
    url = f"{BASE.format(band=band)}/Norder{ORDER}/Dir{npix // 10000 * 10000}/Npix{npix}.fits"
    for attempt in range(RETRIES):
        try:
            r = _session().get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with fits.open(io.BytesIO(r.content)) as h:
                return np.asarray(h[0].data, dtype=np.float32)
        except (requests.RequestException, OSError):
            time.sleep(2**attempt)
    raise RuntimeError(f"cannot fetch {url}")


def deinterleave(v):
    """Nested sub-pixel index -> (x, y) within a HiPS tile."""
    x = np.zeros_like(v)
    y = np.zeros_like(v)
    for k in range(9):
        x |= ((v >> (2 * k)) & 1) << k
        y |= ((v >> (2 * k + 1)) & 1) << k
    return x, y


def sample(band, coord):
    """HiPS values at ICRS positions (nearest HiPS pixel); NaN outside coverage.

    Array layout of a tile (verified on eRASS1 sources): value = data[511 - x, y] where (x, y) are
    the de-interleaved bits of the order-15 nested index relative to the tile.
    """
    idx = _HP15.skycoord_to_healpix(coord)
    tile = idx >> 18
    x, y = deinterleave(idx & ((1 << 18) - 1))
    out = np.full(len(idx), np.nan, np.float32)
    for t in np.unique(tile):
        data = hips_tile(band, int(t))
        if data is None:
            continue
        sel = tile == t
        out[sel] = data[TILE - 1 - x[sel], y[sel]]
    return out


def hires_grid(patch):
    """Size, centre index and pixel scale [deg] of the high-resolution grid of a patch."""
    n = (patch.npix - 1) * HIRES_FACTOR + 1
    return n, (n - 1) / 2, patch.scale / HIRES_FACTOR


def hires_cube(patch, chunk_rows=32):
    """(3, n, n) rate cube [counts/s/arcmin^2] of bands 025/026/027 on the patch TAN grid.

    Rows are processed in chunks to bound memory (SUB^2 sky positions per output pixel).
    """
    n, c, scale = hires_grid(patch)
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    cube = np.empty((len(BANDS), n, n), np.float32)
    for r0 in range(0, n, chunk_rows):
        rows = np.arange(r0, min(r0 + chunk_rows, n))
        iy, ix = np.meshgrid(rows.astype(float), np.arange(n, dtype=float), indexing="ij")
        shape = (len(rows), n, SUB, SUB)
        sx = np.broadcast_to(ix[..., None, None] + off[None, None, None, :], shape).ravel()
        sy = np.broadcast_to(iy[..., None, None] + off[None, None, :, None], shape).ravel()
        l, b = tan_inverse((sx - c) * scale, (sy - c) * scale, patch.lc, patch.bc)
        coord = SkyCoord(l, b, unit="deg", frame="galactic").icrs
        for k, band in enumerate(BANDS):
            v = sample(band, coord).reshape(len(rows), n, SUB * SUB)
            cube[k, rows] = np.mean(v, axis=-1) / HIPS_PIX_ARCMIN**2
    return cube
