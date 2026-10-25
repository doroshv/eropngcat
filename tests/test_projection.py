"""Round-trip tests of the all-sky projections, including poles and the l=180 seam."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from epngcat.projection import AllSkyProjection, forward_norm


def _sep(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    h = np.sin((b2 - b1) / 2) ** 2 + np.cos(b1) * np.cos(b2) * np.sin((l2 - l1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(h)))


@pytest.mark.parametrize("kind", ["hammer", "mollweide"])
def test_roundtrip_including_poles_and_seam(kind):
    p = AllSkyProjection(kind=kind, x0=1885.0, y0=957.0, ax=650.0, ay=650.0, rot=1e-3, l0=0.3)
    rng = np.random.default_rng(0)
    l = np.concatenate([rng.uniform(0, 360, 20000), np.full(50, 180.25), np.full(50, 180.35)])
    b = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 20000))), np.linspace(-89.999, 89.999, 100)])
    b = np.concatenate([b, [89.9999, -89.9999]])
    l = np.concatenate([l, [10.0, 250.0]])
    l2, b2 = p.pix2sky(*p.sky2pix(l, b))
    assert not np.isnan(l2).any()
    assert _sep(l, b, l2, b2).max() < 1e-7


@pytest.mark.parametrize("kind", ["hammer", "mollweide"])
def test_boundary_is_ellipse(kind):
    b = np.linspace(-90, 90, 181)
    x, y = forward_norm(np.full_like(b, np.pi), np.radians(b), kind)
    assert_allclose((x / (2 * np.sqrt(2))) ** 2 + (y / np.sqrt(2)) ** 2, 1.0, atol=1e-9)


def test_outside_is_nan():
    p = AllSkyProjection(x0=100, y0=50, ax=10, ay=10)
    l, b = p.pix2sky(np.array([0.0, 100.0]), np.array([0.0, 50.0]))
    assert np.isnan(l[0]) and np.isnan(b[0])
    assert_allclose([l[1], b[1]], [0.0, 0.0], atol=1e-12)


def test_split_projection_roundtrip():
    from epngcat.projection import SplitProjection

    de = AllSkyProjection(x0=1881.5, y0=958.5, ax=650.0, ay=650.0, l0=0.02)
    ru = AllSkyProjection(x0=1881.3, y0=958.7, ax=651.0, ay=649.0, rot=2e-4, l0=0.05)
    sp = SplitProjection(de, ru)
    rng = np.random.default_rng(1)
    l = rng.uniform(0, 360, 5000)
    b = np.degrees(np.arcsin(rng.uniform(-1, 1, 5000)))
    px, py = sp.sky2pix(l, b)
    l2, b2 = sp.pix2sky(px, py)
    ok = _sep(l, b, l2, b2) < 1e-7  # NaN compares False
    # Round-trip may only fail where the halves overlap/gap: within ~1 px of the joins.
    assert np.abs(px[~ok] - de.x0).max() < 1.0
    assert ok.mean() > 0.99


def test_channel_shift_roundtrip():
    from epngcat.projection import SplitProjection

    de = AllSkyProjection(x0=1881.5, y0=958.5, ax=650.0, ay=650.0)
    sp = SplitProjection(de, de, shift_de=((0, 0), (0, 0), (0.8, 0.02)))
    l, b = np.array([250.0, 30.0]), np.array([10.0, -20.0])
    px, py = sp.sky2pix_channel(l, b, 2)
    x0, y0 = sp.sky2pix(l, b)
    assert_allclose(px - x0, [0.8, 0.0])
    l2, b2 = sp.pix2sky_channel(px, py, 2)
    assert _sep(l, b, l2, b2).max() < 1e-7
