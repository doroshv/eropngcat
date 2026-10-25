"""Scene background fit: the B-spline must not run away where the patch has no data."""

import numpy as np

from epngcat.pipeline import cleaning_diagnostics
from epngcat.scene import Scene, gauss


def _scene(hole=True, npix=101, scale=2 / 60, seed=0):
    """Native pixels on a 5'-ish grid over a patch, optionally with the left half missing."""
    rng = np.random.default_rng(seed)
    step = 2.5
    g = np.arange(-0.5, npix, step)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    if hole:  # as at the l = 180 cut: half the patch lies off the sky ellipse
        keep = x > npix / 2
        x, y = x[keep], y[keep]
    bkg = 0.5 + 0.004 * y  # smooth, gentle gradient
    lin = (bkg[:, None] + rng.normal(0, 0.02, (len(x), 3))).astype(float)
    xy = np.broadcast_to(x, (3, len(x))).copy(), np.broadcast_to(y, (3, len(y))).copy()
    return Scene(npix, scale, *xy, lin, np.zeros(len(x), bool), np.full(3, 0.02), np.eye(2) * 5.6)


def test_background_does_not_ring_into_the_empty_half():
    sc = _scene(hole=True)
    _, coefs = sc.solve([], [])
    bkg, _ = sc.render([], [], np.zeros((0, 3)), coefs)
    # Sky level is ~0.5-0.9; unpenalised coefficients reach tens.
    assert np.abs(coefs).max() < 3.0
    assert bkg.min() > -0.1 and bkg.max() < 2.0
    # The covered half is still fitted to the noise level.
    ix = np.arange(sc.npix) > sc.npix / 2
    truth = 0.5 + 0.004 * np.arange(sc.npix)[:, None]
    assert np.abs(bkg[:, ix, 1] - truth).max() < 0.1


def test_background_follows_a_gradient_when_fully_covered():
    sc = _scene(hole=False)
    _, coefs = sc.solve([], [])
    bkg, _ = sc.render([], [], np.zeros((0, 3)), coefs)
    truth = 0.5 + 0.004 * np.arange(sc.npix)[:, None]
    assert np.abs(bkg[..., 1] - truth).max() < 0.05


def test_cleaning_diagnostics_rejects_broad_black_holes_but_not_compact_cores():
    stretch = {"A": [100.0, 140.0, 160.0], "v0": -6.0}
    raw = np.full((20, 20, 3), 5.0)
    core = np.ones((20, 20), bool)

    compact = raw.copy()
    compact[:2, :, 0] = 0.0  # 10%: comparable to the demanding Cyg X-1 core
    assert not cleaning_diagnostics(raw, compact, core, stretch)["corrupt"]

    floor_hole = raw.copy()
    floor_hole[:4, :, 0] = 0.0  # 20% pinned to the display floor
    assert cleaning_diagnostics(raw, floor_hole, core, stretch)["corrupt"]

    broad_positive = np.full_like(raw, 0.5)  # no clipping, but 90% of the mean removed
    report = cleaning_diagnostics(raw, broad_positive, core, stretch)
    assert report["corrupt"] and max(report["mean_change_fraction"]) > 0.8


def test_failed_sigmoid_band_gets_a_psf_without_touching_successful_bands():
    sc = _scene(hole=False)
    x0 = y0 = 50.0
    sc.lin[:, 2] += 2.0 * gauss(sc.nx[2], sc.ny[2], x0, y0, sc.icov)
    coefs = sc.fit_background()
    amps = sc.fill_missing_source_amplitudes(
        [x0],
        [y0],
        np.zeros((1, 3)),
        coefs,
        [[False, False, True]],
    )
    assert amps[0, 2] > 1.4
    assert amps[0, 0] == 0.0 and amps[0, 1] == 0.0
