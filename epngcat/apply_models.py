"""Apply the models selected in step 09 to pipeline detections -> catalogue columns."""

import pickle
from functools import cache

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from epngcat.config import OUT, is_german
from epngcat.detect import SCALE
from epngcat.fluxmodels import features
from epngcat.projection import load_alignment
from epngcat.s09_models import BANDS, final_position, pos_error
from epngcat.sky import snr_mask

EXT_WIDTH = 1.5  # free-width PSF scale above which a source is flagged as possibly extended
# Saturated-blob area [arcmin2] that also flags a source: the PSF fit does not converge on the
# largest blobs, leaving PSF_WIDTH NaN, and NaN > EXT_WIDTH is False - so exactly the most
# extended regions in the sky (Vela, Sco X-1, Virgo) escaped the width test. A saturated point
# source spans a PSF (the median flagged blob is ~150 arcmin2); thousands means extended emission.
EXT_SAT_AREA = 500.0


@cache
def models():
    """Models pickled by step 09."""
    with open(OUT / "09_models" / "models.pkl", "rb") as fh:
        return pickle.load(fh)


def apply(det):
    """Catalogue table (positions, uncertainties, fluxes, flags) for pipeline detections."""
    mod = models()
    pos = mod["position"]
    split = load_alignment()
    shifts = pos["sky_shift_models"]
    l, b, fallback = final_position(det, split, (pos["estimator"], pos["corrected"]), shifts)
    sat = det.sat_area.values > 0
    feats = features(det)[mod["features"]]
    radec = SkyCoord(l, b, unit="deg", frame="galactic").icrs
    out = pd.DataFrame(
        {
            "RA": radec.ra.deg,
            "DEC": radec.dec.deg,
            "LII": l,
            "BII": b,
            "POS_ERR": pos_error(pos["error_model"], det.snr.values, sat),
        }
    )
    for band in BANDS:
        # 16-84% interval: quantile models around the best prediction, widened as calibrated in step 09.
        p = mod["flux"][band].predict(feats)
        lo, hi = (mod["quantiles"][band][q].predict(feats) for q in (0.16, 0.84))
        k = mod["widen"][band]
        out[f"FLUX_{band}"] = 10**p
        out[f"FLUX_{band}_LO"] = 10 ** (p - k * np.clip(p - lo, 1e-3, None))
        out[f"FLUX_{band}_HI"] = 10 ** (p + k * np.clip(hi - p, 1e-3, None))
    out["SNR"] = np.where(sat, np.inf, det.snr.values)
    out["SATURATED"] = sat
    out["SAT_AREA_ARCMIN2"] = det.sat_area.values * (SCALE * 60) ** 2
    out["PSF_WIDTH"] = det.get("width_n", pd.Series(np.nan, index=det.index)).values
    out["FLAG_EXTENDED"] = (out.PSF_WIDTH > EXT_WIDTH) | (out.SAT_AREA_ARCMIN2 > EXT_SAT_AREA)
    out["FLAG_POS_FALLBACK"] = fallback
    out["SNR_NAME"] = snr_mask(l, b)
    out["HALF"] = np.where(is_german(l), "DE", "RU")
    out["TILE"] = det.tile.values
    return out
