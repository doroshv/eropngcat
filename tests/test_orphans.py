"""Chance correction and binning of the orphan statistics (step 17)."""

import numpy as np
import pandas as pd

from epngcat.s17_orphans import K_MATCH, _curve, by_bin, calibrate, corrected, match_all, predict, select, shift_to_ru


def test_chance_correction_inflates_the_orphan_fraction():
    """Half the sources orphaned while a tenth of positions match by chance -> 0.5 / 0.9."""
    orphan = np.array([True, True, False, False])
    v = corrected(orphan, np.ones(4, bool))
    assert v["fraction_raw"] == 0.5
    assert np.isnan(v["fraction"])  # every shifted position matched: nothing is measurable
    v = corrected(orphan, np.array([True, False, False, False]))
    np.testing.assert_allclose(v["fraction"], 0.5 / 0.75)


def test_by_bin_keeps_the_edges_so_open_bins_can_be_plotted():
    """The last bin runs to infinity; _curve must still place it at a finite x."""
    cat = pd.DataFrame({"HALF": ["DE"] * 40 + ["RU"] * 40, "SNR": np.r_[np.full(40, 5.0), np.full(40, 30.0)]})
    orphan = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    out = by_bin(cat, orphan, np.zeros(80, bool), "SNR", np.array([0, 10, np.inf]))
    assert out["DE"]["0-10"]["fraction"] == 1.0
    assert out["RU"]["10-inf"]["fraction"] == 0.0
    x, y = _curve(out["RU"])
    assert len(x) == 1 and np.isfinite(x[0]) and y == [0.0]


def _training_frame(n=400):
    """Half the sources are blobs, and blobs are spurious twice as often at every SNR."""
    rng = np.random.default_rng(0)
    snr = np.repeat([4.0, 20.0], n)
    blob = np.tile(np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)], 2)
    p = np.where(blob, 0.6, 0.3) * np.where(snr > 10, 0.5, 1.0)
    cat = pd.DataFrame({"SNR": snr, "FLAG_EXTENDED": blob, "SATURATED": np.zeros(2 * n, bool), "FLUX_1": snr * 1e-13})
    return cat, rng.random(2 * n) < p


def test_calibration_separates_the_two_classes_and_predicts_per_source():
    """A blob must get its own, higher probability, not the value of the merged bin."""
    cat, spurious = _training_frame()
    edges = np.array([0.0, 10.0, np.inf])
    cal = calibrate(cat, np.ones(len(cat), bool), spurious, np.zeros(len(cat), bool), edges)
    assert cal["cells"]["blob"][0]["fraction"] > cal["cells"]["point"][0]["fraction"] + 0.15
    assert cal["cells"]["point"][1]["fraction"] < cal["cells"]["point"][0]["fraction"]
    p = predict(cat, cal)
    blob = cat.FLAG_EXTENDED.values
    assert p[blob & (cat.SNR == 4)][0] == cal["cells"]["blob"][0]["fraction"]
    assert np.isfinite(p).all()


def test_prior_shift_is_identity_when_the_prevalences_agree():
    """Shifting to the same prevalence must leave the probabilities untouched, and raise them if higher."""
    cat = pd.DataFrame({"FLUX_1": [1e-13, 1e-12]})
    edges = np.array([0.0, 5e-13, np.inf])
    p = np.array([0.2, 0.2])
    same = [{"pi_de": 0.3, "pi_ru": 0.3}, {"pi_de": 0.3, "pi_ru": 0.3}]
    np.testing.assert_allclose(shift_to_ru(p, cat, same, edges), p)
    higher = [{"pi_de": 0.3, "pi_ru": 0.6}, {"pi_de": 0.3, "pi_ru": 0.3}]
    out = shift_to_ru(p, cat, higher, edges)
    assert out[0] > p[0] and out[1] == p[1]


def test_select_counts_expected_real_and_spurious():
    """Yield at a threshold: N selected, and the split implied by the probabilities themselves."""
    v = select(np.array([0.05, 0.1, 0.9]), thresholds=(0.5,))["0.5"]
    assert v["n"] == 2
    np.testing.assert_allclose(v["expected_spurious"], 0.15)
    np.testing.assert_allclose(v["expected_real"], 1.85)


def test_match_all_scales_the_radius_with_the_combined_error():
    """A 200" offset matches a source with a 70" error but not one with a 30" error (K_MATCH = 3)."""
    assert K_MATCH == 3.0
    ext = {"x": (np.array([10.0]), np.array([20.0 + 200 / 3600]), None)}
    flags = match_all(
        np.array([10.0, 10.0]), np.array([20.0, 20.0]), np.array([70.0, 30.0]), ext, {"x": np.array([5.0])}
    )
    assert flags["x"].tolist() == [True, False]
