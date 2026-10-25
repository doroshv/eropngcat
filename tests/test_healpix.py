"""Tile -> HEALPix assignment of step 16."""

import healpy as hp
import numpy as np
from astropy.io import fits

from epngcat.patches import NROW, in_tile, tile_table
from epngcat.s16_healpix import rendered_coverage, row_pixels, sample_tile

NSIDE = 32  # 12k pixels, enough to hit every row and both poles


def test_every_pixel_is_claimed_by_exactly_one_tile():
    seen = np.zeros(hp.nside2npix(NSIDE), int)
    for row in range(NROW):
        pix, _, _, _ = row_pixels(NSIDE, row)
        seen[pix] += 1
    assert (seen == 1).all(), f"{(seen == 0).sum()} unclaimed, {(seen > 1).sum()} double-counted"


def test_pixels_land_in_the_core_of_the_tile_they_are_assigned_to():
    t = tile_table()
    for row in (0, NROW // 2, NROW - 1):
        pix, l, b, tid = row_pixels(NSIDE, row)
        assert len(pix)
        # The pole pixel of the top row sits at b = 90, outside every core's half-open b range.
        keep = b < 90
        rows = {c: t[c][tid[keep]] for c in ("l_lo", "l_width", "b_lo", "b_hi")}
        assert in_tile(l[keep], b[keep], rows).all()


def test_counterfactual_saturated_core_is_kept_in_clean_map(tmp_path):
    shape = (3, 9, 9)
    data = np.zeros(shape, np.float32)
    data[:, 4, 4] = 255
    clean = np.full(shape, 10.0, np.float32)
    path = tmp_path / "tile.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=fits.Header({"L_CEN": 0.0, "B_CEN": 0.0})),
            fits.ImageHDU(data, name="DATA"),
            fits.ImageHDU(clean, name="SUB"),
            # Stage 16 must ignore processing masks: the fitted core is usable in SUB.
            fits.ImageHDU(np.zeros((9, 9), np.uint8), name="VALID"),
        ]
    ).writeto(path)
    sampled = sample_tile(path, np.array([0.0]), np.array([0.0]), None)
    assert np.isfinite(sampled["clean"]).all()
    np.testing.assert_allclose(sampled["clean"], 10.0)


def test_rendered_coverage_uses_only_the_sky_footprint():
    class IdentityProjection:
        @staticmethod
        def sky2pix(l, b):
            return np.asarray(l), np.asarray(b)

    footprint = np.zeros((7, 7), bool)
    footprint[1:6, 1:6] = True
    covered = rendered_coverage(
        np.array([3.0, 0.0, 6.0]),
        np.array([3.0, 3.0, 6.0]),
        projection=IdentityProjection(),
        footprint=footprint,
    )
    np.testing.assert_array_equal(covered, [True, False, False])
