"""ROSAT count-rate conversion and chance-corrected completeness for step 12."""

import numpy as np

from epngcat.s01_prepare_catalogs import rass_flux
from epngcat.s12_rosat import completeness


def test_rass_flux_matches_the_schmitt_ecf():
    """ECF runs from 3.0 (HR1 = -1) to 13.6 (HR1 = +1) 1e-12 erg/s/cm2 per ct/s, 8.31 at HR1 = 0."""
    np.testing.assert_allclose(rass_flux([1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]), [3.01e-12, 8.31e-12, 13.61e-12])
    np.testing.assert_allclose(rass_flux(2.0, 0.0), 1.662e-11)


def test_completeness_divides_out_the_chance_rate():
    """Half matched of which half are chance -> 0 true completeness; all matched -> 1."""
    sel = np.ones(4, bool)
    assert completeness(np.array([1, 1, 0, 0], bool), np.array([1, 1, 0, 0], bool), sel)["completeness"] == 0.0
    assert completeness(np.ones(4, bool), np.array([1, 0, 0, 0], bool), sel)["completeness"] == 1.0
    assert completeness(np.ones(4, bool), np.zeros(4, bool), np.zeros(4, bool)) == {"n": 0}
