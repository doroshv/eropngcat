"""Step-9 positional error model."""

import numpy as np

from epngcat.s09_models import fit_pos_error, pos_error


def test_pos_error_binned_monotone_and_clamped():
    rng = np.random.default_rng(0)
    snr = rng.uniform(3.5, 40, 5000)
    sigma = np.where(snr < 10, 80.0, 25.0)  # step like the centroid -> profile-fit estimator switch
    sep = sigma * np.sqrt(-2 * np.log(rng.uniform(size=snr.size)))  # Rayleigh radial offsets
    sat = np.zeros(snr.size, bool)
    sat[:100] = True
    model = fit_pos_error(snr, sat, sep, np.zeros(snr.size))
    assert np.all(np.diff(model["bins_sigma"]) <= 0)
    lo, hi = pos_error(model, np.array([1.0, 1e3]), False)
    np.testing.assert_allclose([lo, hi], [model["bins_sigma"][0], model["bins_sigma"][-1]])
    np.testing.assert_allclose(hi, 25.0, rtol=0.15)
    assert pos_error(model, np.inf, True) == model["sat"]
