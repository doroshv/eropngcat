"""Saturated-source sigmoid halo fitting."""

import numpy as np

from epngcat.brightscene import (
    MIN_WIDTH,
    WIDTH_BOUNDARY_FACTOR,
    BrightSourceFit,
    _profile_fit_is_credible,
    fit_bright_sources,
    sigmoid,
)
from epngcat.patches import Patch
from epngcat.pipeline import reject_truncated_bright


def _patch(seed=4):
    rng = np.random.default_rng(seed)
    npix, scale = 121, 2 / 60
    yy, xx = np.mgrid[:npix, :npix]
    center = np.array([60.6, 59.4])
    radius = np.hypot(xx - center[0], yy - center[1]) * scale * 60
    amplitudes = np.array([180.0, 240.0, 320.0])
    cube = np.stack([12.0 + sigmoid(radius, a, 20.0, 5.0) for a in amplitudes], -1)
    cube += rng.normal(0, 0.8, cube.shape)
    rgb = np.clip(cube, 0, 255).astype(np.uint8)
    return (
        Patch(0.0, 0.0, scale, npix, rgb, np.ones((npix, npix), bool), np.eye(2), (rgb >= 255).any(-1)),
        center,
    )


def test_sigmoid_fit_recovers_saturated_halo_and_renders_it():
    patch, center = _patch()
    area = [patch.sat.sum()]
    fit = fit_bright_sources(patch, None, [center], area)
    assert fit.success.all()
    assert np.isfinite(fit.half_radius).all() and (fit.half_radius > 0).all()
    assert np.all(fit.width > WIDTH_BOUNDARY_FACTOR * MIN_WIDTH)
    assert np.all(fit.amplitude[0] > 100)
    rendered = fit.render(patch.npix, patch.scale)
    assert rendered.shape == patch.rgb.shape
    assert np.all(rendered[60, 61] > rendered[5, 5])


def test_refit_can_move_common_centroid_without_losing_band_geometry():
    patch, center = _patch()
    seed = center + [1.2, -0.8]
    first = fit_bright_sources(patch, None, [seed], [patch.sat.sum()])
    final = fit_bright_sources(
        patch,
        None,
        first.centers,
        [patch.sat.sum()],
        initial=first,
        refine_centroid=True,
    )
    assert np.hypot(*(final.centers[0] - center)) < np.hypot(*(seed - center))
    assert final.success.all()


def test_refit_rejects_implausibly_large_centroid_displacement():
    patch, center = _patch()
    seed = center + [4.0, 0.0]  # 8 arcmin at the test-patch scale
    first = fit_bright_sources(patch, None, [seed], [patch.sat.sum()])
    final = fit_bright_sources(
        patch,
        None,
        first.centers,
        [patch.sat.sum()],
        initial=first,
        refine_centroid=True,
    )
    assert not final.success.any()
    np.testing.assert_allclose(final.centers[0], seed)
    assert not final.amplitude.any()


def test_profile_quality_rejects_no_improvement_and_excess_model_flux():
    radius = np.arange(2.0, 62.0, 4.0)
    error = np.full_like(radius, 0.2)
    observed = sigmoid(radius, 10.0, 18.0, 4.0)
    assert _profile_fit_is_credible(observed, error, radius, observed)
    assert not _profile_fit_is_credible(observed, error, radius, np.zeros_like(observed))
    assert not _profile_fit_is_credible(observed, error, radius, 5.0 * observed)


def test_truncated_bright_component_is_disabled_without_touching_interior_source():
    fit = BrightSourceFit(
        centers=np.array([[0.5, 50.0], [50.0, 50.0]]),
        amplitude=np.ones((2, 3)),
        half_radius=np.ones((2, 3)),
        width=np.ones((2, 3)),
        baseline=np.zeros((2, 3)),
        clip_level=np.ones((2, 3)),
        clip_width=np.ones((2, 3)),
        success=np.ones((2, 3), bool),
        initial_centers=np.array([[0.5, 50.0], [50.0, 50.0]]),
    )
    rejected = reject_truncated_bright(fit, 101)
    np.testing.assert_array_equal(rejected, [True, False])
    assert not fit.success[0].any() and not fit.amplitude[0].any()
    assert fit.success[1].all() and fit.amplitude[1].all()
