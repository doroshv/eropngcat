"""Global uncertainty-systematic calibration and application."""

import numpy as np
import pandas as pd

from epngcat.s09_models import BANDS
from epngcat.uncertainty import apply_uncertainty_systematics, unique_pairs


def test_unique_pairs_rejects_many_to_one_associations():
    image = pd.DataFrame(
        {
            "LII": [0.0, 1.0, 1.01, 2.0],
            "BII": [0.0, 0.0, 0.0, 0.0],
            "FLUX_1": [2.0, 2.0, 2.0, 0.5],
            "FLAG_EXTENDED": [False, False, False, False],
            "FLAG_POS_FALLBACK": [False, False, False, False],
        }
    )
    reference = pd.DataFrame(
        {
            "LII": [0.0, 1.005, 2.0],
            "BII": [0.0, 0.0, 0.0],
            "EXT": [0, 0, 0],
            "ML_FLUX_1": [2.0, 2.0, 2.0],
        }
    )
    matched, truth = unique_pairs(image, reference, flux_limit=1.0)
    assert len(matched) == len(truth) == 1
    assert matched.LII.iloc[0] == 0.0
    assert truth.LII.iloc[0] == 0.0


def test_apply_uncertainty_systematics_adds_terms_in_quadrature():
    cat = pd.DataFrame({"POS_ERR": [4.0]})
    calibration = {
        "position": {"add_arcsec": 3.0, "fraction": 0.5},
        "flux": {},
    }
    for band in BANDS:
        cat[f"FLUX_{band}"] = 10.0
        cat[f"FLUX_{band}_LO"] = 8.0
        cat[f"FLUX_{band}_HI"] = 13.0
        calibration["flux"][band] = {"add_flux": 0.0, "fraction": 0.1}

    out = apply_uncertainty_systematics(cat, calibration)
    np.testing.assert_allclose(out.POS_ERR, np.sqrt(4.0**2 + (0.5 * 4.0) ** 2 + 3.0**2))
    for band in BANDS:
        np.testing.assert_allclose(out[f"FLUX_{band}_LO"], 10.0 - np.sqrt(2.0**2 + 1.0**2))
        np.testing.assert_allclose(out[f"FLUX_{band}_HI"], 10.0 + np.sqrt(3.0**2 + 1.0**2))


def test_released_terms_do_not_depend_on_test_residuals():
    from epngcat.uncertainty import flux_systematics, position_systematics

    rng = np.random.default_rng(17)
    n = 600
    image = pd.DataFrame(
        {
            "SPLIT": ["train"] * 300 + ["val"] * 150 + ["test"] * 150,
            "POS_ERR": np.full(n, 20.0),
            "MATCH_SEP": rng.rayleigh(30.0, n),
        }
    )
    reference = pd.DataFrame({"POS_ERR": np.ones(n)})
    for band in BANDS:
        image[f"FLUX_{band}"] = 1e-12
        image[f"FLUX_{band}_LO"] = 0.9e-12
        image[f"FLUX_{band}_HI"] = 1.1e-12
        reference[f"ML_FLUX_{band}"] = 1e-12 * np.exp(rng.normal(0, 0.2, n))
        reference[f"ML_FLUX_ERR_{band}"] = 1e-14
    original_pos = position_systematics(image, reference)
    original_flux, _ = flux_systematics(image, reference, original_pos)
    altered = image.copy()
    changed_ref = reference.copy()
    test = image.SPLIT == "test"
    altered.loc[test, "MATCH_SEP"] *= 20
    altered.loc[test, "POS_ERR"] *= 20
    for band in BANDS:
        altered.loc[test, f"FLUX_{band}"] *= 5
        changed_ref.loc[test, f"ML_FLUX_{band}"] *= 10
    changed_pos = position_systematics(altered, changed_ref)
    changed_flux, _ = flux_systematics(altered, changed_ref, changed_pos)
    for key in ("selected", "add_arcsec", "fraction"):
        assert original_pos[key] == changed_pos[key]
    assert original_pos["test"] != changed_pos["test"]
    for band in BANDS:
        for key in ("selected", "add_flux", "fraction"):
            assert original_flux[band][key] == changed_flux[band][key]
        selected = original_flux[band]["candidates"][original_flux[band]["selected"]]["parameters_train"]
        assert selected["add_flux"] == original_flux[band]["add_flux"]
        assert selected["fraction"] == original_flux[band]["fraction"]
