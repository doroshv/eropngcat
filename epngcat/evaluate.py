"""Matching detections to the eRASS1-DE catalogue: completeness, reliability, offsets.

The catalogue is far deeper than the image, so every real detection has a catalogue counterpart
but spurious detections also find one by chance. The chance rate p_c is measured by matching
detections displaced by SHIFT; the reliability then follows from
f_match = R + (1 - R) p_c.
"""

from functools import cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from epngcat.config import OUT, SEED
from epngcat.sky import SNR_EXCLUDE_MAX_ARCMIN, radec_unit, sep_arcsec, snr_mask

MATCH_R = 4.0  # arcmin; baseline offsets reach ~4 arcmin, chance matches are corrected for
F_FLOOR = 1e-13  # counterparts must be at least this bright (0.2-2.3 keV, erg/s/cm2)
SHIFT = 15.0  # arcmin, displacement for the chance-match estimate


def in_snr(l, b):
    """Flag positions inside the exclusion mask of a known supernova remnant."""
    return snr_mask(np.asarray(l), np.asarray(b), SNR_EXCLUDE_MAX_ARCMIN) != ""


@cache
def catalog(split=None):
    """eRASS1-DE catalogue with tile/split columns, optionally restricted to one split.

    Sources inside known supernova remnants are removed (see in_snr): SNR emission would be
    confused with point sources and must not enter calibration or evaluation.
    """
    cat = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    tt = pd.read_parquet(OUT / "04_tiles" / "erass1_tiles.parquet")
    cat["tile"], cat["split"] = tt.tile.values, tt.split.values
    cat = cat[np.isfinite(cat.ML_FLUX_1)]
    if split is not None:
        cat = cat[cat.split == split]
    cat = cat[~in_snr(cat.LII.values, cat.BII.values)]
    return cat.reset_index(drop=True)


def clean(det):
    """Drop detections inside known supernova remnants."""
    return det[~in_snr(det.l.values, det.b.values)] if len(det) else det


def brightest_within(l, b, cat, radius_arcmin=MATCH_R, f_floor=F_FLOOR):
    """Index of the brightest catalogue source (F >= f_floor) within radius of each position (-1 if none)."""
    c = cat[f_floor <= cat.ML_FLUX_1]
    tree = cKDTree(radec_unit(c.LII.values, c.BII.values))
    chord = 2 * np.sin(np.radians(radius_arcmin / 60) / 2)
    flux = c.ML_FLUX_1.values
    idx = c.index.values
    out = np.full(len(l), -1)
    for k, nb in enumerate(tree.query_ball_point(radec_unit(l, b), chord)):
        if nb:
            out[k] = idx[nb[int(np.argmax(flux[nb]))]]
    return out


def shifted(l, b, shift_arcmin=SHIFT, seed=SEED):
    """Positions displaced by a fixed angular distance in random directions."""
    rng = np.random.default_rng(seed)
    pa = rng.uniform(0, 2 * np.pi, len(l))
    d = np.radians(shift_arcmin / 60)
    l0, b0 = np.radians(l), np.radians(b)
    b1 = np.arcsin(np.sin(b0) * np.cos(d) + np.cos(b0) * np.sin(d) * np.cos(pa))
    l1 = l0 + np.arctan2(np.sin(pa) * np.sin(d) * np.cos(b0), np.cos(d) - np.sin(b0) * np.sin(b1))
    return np.degrees(l1) % 360, np.degrees(b1)


def reliability(det, cat, radius_arcmin=MATCH_R):
    """(reliability, match fraction, chance fraction) of a detection list."""
    if len(det) == 0:
        return np.nan, np.nan, np.nan
    f = (brightest_within(det.l.values, det.b.values, cat, radius_arcmin) >= 0).mean()
    pc = (brightest_within(*shifted(det.l.values, det.b.values), cat, radius_arcmin) >= 0).mean()
    return (f - pc) / (1 - pc), f, pc


def match_detections(det, cat, radius_arcmin=MATCH_R):
    """Attach the brightest counterpart (catalogue columns prefixed with 'cat_') and offset [arcsec]."""
    idx = brightest_within(det.l.values, det.b.values, cat, radius_arcmin)
    out = det.copy()
    ok = idx >= 0
    for c in ["IAUNAME", "LII", "BII", "ML_FLUX_1", "ML_FLUX_P1", "ML_FLUX_P2", "ML_FLUX_P3", "EXT", "POS_ERR"]:
        col = cat[c].values
        v = np.full(
            len(det), None if col.dtype.kind in "OSU" else np.nan, dtype=object if col.dtype.kind in "OSU" else float
        )
        v[ok] = col[idx[ok]]
        out["cat_" + c] = v
    out["sep"] = np.where(ok, sep_arcsec(det.l.values, det.b.values, out.cat_LII.values, out.cat_BII.values), np.nan)
    return out


def completeness(det, cat, bins, radius_arcmin=MATCH_R, point_only=True):
    """Chance-corrected fraction of catalogue sources (per flux bin) with a detection within radius.

    p_c (a random position has a detection within radius) is measured with displaced catalogue
    positions; completeness = (f - p_c) / (1 - p_c).
    """
    c = cat[cat.EXT == 0] if point_only else cat
    tree = cKDTree(radec_unit(det.l.values, det.b.values))
    chord = 2 * np.sin(np.radians(radius_arcmin / 60) / 2)
    d, _ = tree.query(radec_unit(c.LII.values, c.BII.values), distance_upper_bound=chord)
    ds, _ = tree.query(radec_unit(*shifted(c.LII.values, c.BII.values)), distance_upper_bound=chord)
    pc = np.isfinite(ds).mean()
    k = np.digitize(c.ML_FLUX_1.values, bins)
    n = np.bincount(k, minlength=len(bins) + 1)[1:-1]
    h = np.bincount(k, weights=np.isfinite(d), minlength=len(bins) + 1)[1:-1]
    return (h / np.maximum(n, 1) - pc) / (1 - pc), n


def flux_at_completeness(frac, comp, bins, n=None):
    """Flux (log-interpolated) where the monotone (isotonic) completeness curve reaches frac.

    Completeness is fitted as a non-decreasing function of log flux, weighted by the number of
    catalogue sources per bin, so a few noisy bright bins (small numbers, extended or confused
    sources) do not break the estimate. Bins with no sources (n == 0) are ignored. NaN if the
    fitted curve never reaches frac.
    """
    from sklearn.isotonic import IsotonicRegression

    centres = np.log10(np.sqrt(bins[1:] * bins[:-1]))
    w = np.ones_like(centres) if n is None else np.asarray(n, float)
    ok = (w > 0) & np.isfinite(comp)
    x, c = (
        centres[ok],
        IsotonicRegression().fit(centres[ok], np.asarray(comp)[ok], sample_weight=w[ok]).predict(centres[ok]),
    )
    above = np.nonzero(c >= frac)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(10 ** x[0])
    t = (frac - c[i - 1]) / (c[i] - c[i - 1])
    return float(10 ** (x[i - 1] + t * (x[i] - x[i - 1])))
