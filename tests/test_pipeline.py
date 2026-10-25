"""Tests of pipeline helpers."""

import numpy as np
import pandas as pd

from epngcat.brightscene import BrightSourceFit
from epngcat.patches import Patch
from epngcat.pipeline import (
    dedup,
    halo_background_level,
    patch_snr_overlaps,
    reject_implausible_bright,
)


def test_dedup_keeps_brighter_across_tiles_only():
    det = pd.DataFrame(
        {
            "l": [10.0, 10.02, 10.5, 10.51],
            "b": [5.0, 5.0, 5.0, 5.0],
            "tile": [1, 2, 3, 3],  # last pair is in the same tile: kept (deblended pair)
            "snr": [6.0, 9.0, 7.0, 8.0],
            "sat_area": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = dedup(det)
    assert sorted(out.snr) == [7.0, 8.0, 9.0]
    assert np.isclose(out.l.iloc[0], 10.02)


def test_flux_at_completeness_monotone_and_robust():
    from epngcat.evaluate import flux_at_completeness

    bins = np.geomspace(1e-14, 1e-10, 9)
    comp = np.array([0.0, 0.05, 0.3, 0.7, 0.95, 0.97, 0.6, 0.98])  # noisy bright bin
    n = np.array([1000, 800, 500, 200, 80, 30, 3, 2])
    f50 = flux_at_completeness(0.5, comp, bins, n)
    f90 = flux_at_completeness(0.9, comp, bins, n)
    assert bins[2] < f50 < bins[4]
    assert np.isfinite(f90) and f90 > f50


def test_halo_background_uses_core_and_trims_bright_tail():
    rgb = np.full((7, 7, 3), 1.0)
    rgb[3, 3, 0] = 100.0
    patch = Patch(10.0, 0.0, 0.5, 7, rgb, np.ones((7, 7), bool), np.eye(2))
    tile = {"l_lo": 9.0, "l_width": 2.0, "b_lo": -1.0, "b_hi": 1.0}
    assert np.isclose(halo_background_level(patch, tile, None), 1.0)


def test_snr_overlap_uses_patch_footprint_and_rejects_far_hemisphere():
    patch = Patch(10.0, 0.0, 0.1, 21, np.zeros((21, 21, 3)), np.ones((21, 21), bool), np.eye(2))
    cat = pd.DataFrame(
        {
            "name": ["inside", "touching", "outside", "opposite"],
            "l": [10.0, 11.1, 12.0, 190.0],
            "b": [0.0, 0.0, 0.0, 0.0],
            "mask_radius_arcmin": [1.0, 4.0, 4.0, 1000.0],
        }
    )
    assert patch_snr_overlaps(patch, cat) == ["inside", "touching"]
    tile = {"l_c": 10.0, "b_c": 0.0, "l_lo": 9.5, "l_width": 1.0, "b_lo": -1.0, "b_hi": 1.0}
    assert patch_snr_overlaps(patch, cat, tile) == ["inside"]


def test_implausible_halo_qa_is_per_band():
    fit = BrightSourceFit(
        centers=np.array([[10.0, 10.0]]),
        amplitude=np.array([[10.0, 100.0, 2.0]]),
        half_radius=np.array([[5.0, 5.0, 40.0]]),
        width=np.array([[3.0, 3.0, 10.0]]),
        baseline=np.zeros((1, 3)),
        clip_level=np.ones((1, 3)),
        clip_width=np.ones((1, 3)),
        success=np.ones((1, 3), bool),
        initial_centers=np.array([[10.0, 10.0]]),
    )
    result = reject_implausible_bright(fit, [25.0])
    assert fit.success.tolist() == [[True, False, False]]
    assert fit.amplitude.tolist() == [[10.0, 0.0, 0.0]]
    assert result["corrupt"]


def test_psf_calibration_ignores_its_previous_output(tmp_path, monkeypatch):
    from epngcat import pipeline
    from epngcat.config import save_json

    folder = tmp_path / "07b_psf_scene"
    folder.mkdir()
    monkeypatch.setattr(pipeline, "OUT", tmp_path)
    calibration = {"psf": {"t2": 4.0, "s2": 1.0}, "scene_psf": {"t2": 4.0, "s2": 1.0}}
    clean_start = pipeline.scene_psf(calibration)
    save_json({"t2": 99.0}, folder / "psf_scene.json")
    assert pipeline.scene_psf(calibration) == clean_start
