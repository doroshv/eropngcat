"""Spherical helpers."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def radec_unit(lon, lat):
    """Return unit vectors (n, 3) for longitude/latitude in degrees."""
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def sep_arcsec(l1, b1, l2, b2):
    """Great-circle separation in arcsec (haversine)."""
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    h = np.sin((b2 - b1) / 2) ** 2 + np.cos(b1) * np.cos(b2) * np.sin((l2 - l1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600


def crossmatch(lon1, lat1, lon2, lat2, radius_arcsec):
    """Nearest neighbour of each object of set 1 in set 2 within radius.

    Returns (index into set 2 or -1, separation in arcsec or inf).
    """
    tree = cKDTree(radec_unit(lon2, lat2))
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600) / 2)
    d, i = tree.query(radec_unit(lon1, lat1), distance_upper_bound=chord)
    ok = np.isfinite(d)
    sep = np.full(len(d), np.inf)
    sep[ok] = np.degrees(2 * np.arcsin(d[ok] / 2)) * 3600
    return np.where(ok, i, -1), sep


SNR_EXCLUDE_MAX_ARCMIN = 300  # larger remnants (only Antlia, 24 deg, diffuse) are flagged, not excluded


def snr_mask(l, b, max_size=None):
    """Name of the SNRcat remnant whose mask contains each position ('' if none).

    max_size [arcmin] restricts the remnants considered (use SNR_EXCLUDE_MAX_ARCMIN for exclusion).
    """
    from epngcat.config import OUT

    snr = pd.read_parquet(OUT / "01_catalogs" / "snrcat.parquet").sort_values("size_arcmin")
    if max_size is not None:
        snr = snr[snr.size_arcmin <= max_size]
    out = np.full(len(l), "", dtype=object)
    tree = cKDTree(radec_unit(l, b))
    # Largest first so that smaller (more specific) remnants overwrite.
    for s in snr[::-1].itertuples():
        chord = 2 * np.sin(np.radians(s.mask_radius_arcmin / 60) / 2)
        out[tree.query_ball_point(radec_unit([s.l], [s.b])[0], chord)] = s.name
    return out
