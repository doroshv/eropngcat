"""Check shared sky selection and area accounting for number counts."""

import healpy as hp
import numpy as np
import pandas as pd

from epngcat import s18_figures as figures


def test_same_footprint_selects_sources_and_area(monkeypatch):
    class Projection:
        de = type("Half", (), {"x0": -1000})()

        def sky2pix(self, lon, lat):
            return np.full(len(lon), 50.0), np.full(len(lat), 50.0)

    from epngcat import image

    monkeypatch.setattr(figures, "load_alignment", lambda: Projection())
    monkeypatch.setattr(image, "sky_mask", lambda: np.ones((100, 100), bool))
    monkeypatch.setattr(figures, "snr_mask", lambda lon, lat: np.where((lon > 210) & (lon < 240), "mask", ""))
    nside = 32
    mask, area = figures.count_footprint(nside)
    lon, lat = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
    expected = (
        (np.abs(lat) > figures.B_CUT)
        & (np.abs(((lon + 90) % 180) - 90) > figures.COUNT_JOIN_DEG)
        & ~((lon > 210) & (lon < 240))
    )
    np.testing.assert_array_equal(mask, expected)
    for catalogue in (pd.DataFrame({"l": lon, "b": lat}), pd.DataFrame({"l": lon, "b": lat}).iloc[::-1]):
        selected = figures.in_count_footprint(catalogue.l, catalogue.b, mask)
        assert selected.sum() * hp.nside2pixarea(nside, degrees=True) == sum(area.values())
