"""Detection of stuck-value image defects."""

import numpy as np

from epngcat.image import DEFECT_RUN, defects


def _frame(seed=0):
    """Smooth faint background with a bright blob, quantised to 8 bit like the press image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:40, :40]
    a = 20 + 8 * np.sin(xx / 7.0) + rng.normal(0, 2, (40, 40))
    a += 220 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 8.0)
    return np.repeat(np.clip(a, 0, 255).round()[..., None], 3, -1).astype(np.float32)


def test_stuck_column_through_a_bright_blob_is_flagged():
    img = _frame()
    img[14:30, 20, 2] = 28.0  # as in the real image: B frozen while its neighbours are bright
    bad = defects(img)
    # Flagged where the stuck value disagrees with the flanking columns, i.e. across the blob.
    assert bad[18:23, 20].all()
    assert not bad[:, 19].any() and not bad[:, 21].any()
    assert bad.sum() < 16


def test_flat_faint_background_is_not_flagged():
    img = _frame()
    img[5:25, 3, 1] = img[5:25, 3, 1].mean().round()  # a stuck run that matches its surroundings
    assert not defects(img)[:, 3].any()


def test_short_runs_are_ignored():
    img = _frame()
    img[19 : 19 + DEFECT_RUN - 2, 20, 2] = 28.0  # one pixel short of a run
    assert not defects(img)[:, 20].any()
