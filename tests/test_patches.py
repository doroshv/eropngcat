"""Tests of the sky tiling and gnomonic helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from epngcat.config import OUT, is_german
from epngcat.patches import NTILE, core_halfwidth, in_tile, tan_forward, tan_inverse, tile_of, tile_table


def test_tiles_partition_sky_and_halves():
    rng = np.random.default_rng(0)
    l = rng.uniform(0, 360, 200000)
    b = np.degrees(np.arcsin(rng.uniform(-1, 1, 200000)))
    t = tile_table()
    ids = tile_of(l, b)
    row = {k: v[ids] for k, v in t.items()}
    assert in_tile(l, b, row).all()
    # A tile never straddles the DE/RU boundary.
    de_frac = np.bincount(ids, weights=is_german(l), minlength=len(t["tile"])) / np.maximum(
        np.bincount(ids, minlength=len(t["tile"])), 1
    )
    assert np.all((de_frac == 0) | (de_frac == 1))
    assert (NTILE % 2 == 0).all() and NTILE.max() == 180


def test_tile_extent_small_everywhere():
    t = tile_table()
    hw = np.array([core_halfwidth({k: v[i] for k, v in t.items()}) for i in range(0, len(t["tile"]), 7)])
    assert hw.max() < 1.6


def test_tan_roundtrip_near_pole():
    l = np.array([0.0, 90.0, 180.0, 270.0, 33.0])
    b = np.array([89.5, 89.9, 89.99, 88.1, 90.0])
    xi, eta = tan_forward(l, b, 45.0, 89.0)
    l2, b2 = tan_inverse(xi, eta, 45.0, 89.0)
    assert_allclose(b2, b, atol=1e-9)
    assert_allclose(((l2 - l + 180) % 360 - 180)[:-1], 0, atol=1e-7)


@pytest.mark.skipif(not (OUT / "03_align/alignment.json").exists(), reason="needs pipeline output (make all)")
def test_tile_wcs_matches_patch_geometry():
    from epngcat.patches import make_patch
    from epngcat.projection import load_alignment
    from epngcat.s14_tile_products import tile_wcs

    for lc, bc in [(250.0, 30.0), (45.0, 89.0)]:
        patch = make_patch(load_alignment(), lc, bc, 1.0, 2 / 60)
        ix, iy = np.array([0.0, 10.5, 60.0]), np.array([3.0, 40.0, 59.0])
        l, b = patch.pix2sky(ix, iy)
        w = tile_wcs(patch).wcs_pix2world(ix, iy, 0)
        assert_allclose(w[1], b, atol=1e-8)
        assert_allclose((w[0] - l + 180) % 360 - 180, 0, atol=1e-8)
