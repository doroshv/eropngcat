"""Tests of the detection kernel geometry."""

import numpy as np
from numpy.testing import assert_allclose

from epngcat.detect import DEFAULTS, detect_patch, gaussian_kernel, psf_cov
from epngcat.patches import Patch


def test_psf_cov_isotropic_for_scaled_rotation():
    rot = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    assert_allclose(psf_cov(0.5 * rot, {"t2": 4.0, "s2": 1.0}), 8.0 * np.eye(2), atol=1e-12)


def test_kernel_moments_match_covariance():
    cov = np.array([[9.0, 2.0], [2.0, 4.0]])
    k = gaussian_kernel(cov, nsig=6)
    h = k.shape[0] // 2
    yy, xx = np.mgrid[-h : h + 1, -h : h + 1]
    assert_allclose(k.sum(), 1.0)
    assert_allclose([(k * xx * xx).sum(), (k * xx * yy).sum(), (k * yy * yy).sum()], [9, 2, 4], rtol=0.02)


def test_detection_override_keeps_original_raw_values_and_omits_saturated_seed():
    npix = 31
    rgb = np.random.default_rng(2).integers(0, 4, (npix, npix, 3), dtype=np.uint8)
    rgb[15, 15] = 255
    rgb[8, 22] = 40
    patch = Patch(0.0, 0.0, 2 / 60, npix, rgb, np.ones((npix, npix), bool), np.eye(2), rgb[..., 0] == 255)
    lin = rgb.astype(float)
    lin[15, 15] = 0.0
    params = {**DEFAULTS, "thresh": 2.0, "kernel_sigma": 1.0, "bkg_size": 7}
    out = detect_patch(patch, params, lin=lin, include_saturated=False)
    assert not np.asarray(out["sat_area"]).any()
    index = np.argmin(np.hypot(np.asarray(out["x"]) - 22, np.asarray(out["y"]) - 8))
    assert out["raw_R"][index] == 40
