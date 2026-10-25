"""Synthetic-source tests of the per-detection measurements."""

import numpy as np
import pandas as pd
import pytest

from epngcat.config import OUT
from epngcat.detect import gaussian_kernel, psf_cov
from epngcat.measure import measure_patch
from epngcat.patches import Patch


def _synthetic_patch(x0, y0, amp, jac, psf, stretch, npix=61, bkg=0.3, seed=0):
    cov = psf_cov(jac, psf)
    yy, xx = np.mgrid[:npix, :npix]
    q = np.stack([xx - x0, yy - y0], -1)
    g = np.exp(-0.5 * np.einsum("...i,ij,...j->...", q, np.linalg.inv(cov), q))
    rng = np.random.default_rng(seed)
    lin = bkg + amp[None, None, :] * g[..., None] + rng.normal(0, 0.01, (npix, npix, 3))
    a = np.asarray(stretch["A"])
    raw = np.clip(np.arcsinh(lin) * a + stretch["v0"], 0, 255)
    return Patch(0.0, 0.0, 2 / 60, npix, raw, np.ones((npix, npix), bool), jac)


def test_psf_fit_recovers_position_and_amplitude():
    stretch = {"A": [90.0, 140.0, 190.0], "v0": 0.0}
    psf = {"t2": 5.6, "s2": 0.25}
    jac = np.array([[-0.30, -0.80], [0.10, -0.20]])  # strongly sheared, as near the image edge
    amp = np.array([1.0, 0.6, 0.4])
    patch = _synthetic_patch(30.7, 29.4, amp, jac, psf, stretch)
    det = pd.DataFrame({"x": [30.0], "y": [30.0], "sat_area": [0.0], "snr": [20.0]})
    (m,) = measure_patch(patch, det, stretch, psf)
    assert abs(m["x_psf"] - 30.7) < 0.1 and abs(m["y_psf"] - 29.4) < 0.1
    assert abs(m["A_R"] - 1.0) < 0.05 and abs(m["A_B"] - 0.4) < 0.05
    assert abs(m["width"] - 1.0) < 0.05
    assert gaussian_kernel(psf_cov(jac, psf)).shape[0] > 1


@pytest.mark.skipif(not (OUT / "03_align/alignment.json").exists(), reason="needs pipeline output (make all)")
def test_native_pixels_cover_patch_across_seam():
    from epngcat.patches import make_patch, native_pixels
    from epngcat.projection import load_alignment

    proj = load_alignment()
    patch = make_patch(proj, 180.2, 10.0, 1.2, 2 / 60)  # straddles the l = 180 image edge
    iy, ix, x, y, rgb = native_pixels(proj, patch)
    assert (ix < 100).any() and (ix > 3600).any()  # pixels from both image edges
    assert x.shape == (3, len(ix))
    # Native pixels are ~5.3' across, so a 2.4 deg patch needs O(700) of them.
    assert len(ix) > 200
