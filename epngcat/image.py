"""Loading of the all-sky RGB image."""

from functools import cache

import numpy as np
from PIL import Image

from epngcat.config import IMAGE

BACKGROUND_VALUE = 3  # RGB value of the empty frame outside the sky ellipse
DEFECT_RUN = 6  # identical values in a row/column before a segment counts as stuck
DEFECT_JUMP = 30  # display units by which a stuck segment must differ from the lines flanking it


@cache
def load_rgb():
    """Return the image as float32 array (ny, nx, 3) in the original 0-255 units."""
    return np.asarray(Image.open(IMAGE))[..., :3].astype(np.float32)


def sky_mask():
    """Boolean mask of the sky ellipse: largest non-background component with holes filled."""
    from scipy import ndimage

    img = load_rgb()
    fg = ~(img == BACKGROUND_VALUE).all(-1)
    lab, _ = ndimage.label(ndimage.binary_closing(fg, iterations=2))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(lab == sizes.argmax())


def defects(img):
    """Pixels frozen along a row or column and inconsistent with the two lines flanking them.

    The image carries short one-pixel-wide segments of a single channel stuck at a constant value:
    column 1880 of B, for instance, holds 28 over rows 1851-1877 while its neighbours reach 255,
    cutting straight through a bright source. A median window is dominated by unaffected background,
    so a stuck segment is compared with the lines immediately beside it instead.
    """
    from scipy import ndimage

    bad = np.zeros(img.shape[:2], bool)
    for k in range(3):
        a = img[..., k]
        for axis in (0, 1):
            line = np.ones(DEFECT_RUN - 1, bool).reshape((-1, 1) if axis == 0 else (1, -1))
            same = np.diff(a, axis=axis) == 0  # gap i joins pixel i and i + 1
            long = ndimage.binary_dilation(ndimage.binary_erosion(same, line), line)
            long = np.insert(long, long.shape[axis], False, axis=axis)  # gap i joins pixels i, i + 1
            prev = np.roll(long, 1, axis=axis)
            prev[(0, slice(None)) if axis == 0 else (slice(None), 0)] = False
            run = long | prev  # a pixel is stuck if either of its gaps lies in a long run
            lo, hi = np.roll(a, 1, axis=1 - axis), np.roll(a, -1, axis=1 - axis)
            flank = np.abs(a - (lo.astype(np.float32) + hi) / 2)
            flank[(slice(None), [0, -1]) if axis == 0 else ([0, -1], slice(None))] = 0.0
            bad |= run & (flank > DEFECT_JUMP)
    return bad


@cache
def defect_mask():
    """Defect mask of the all-sky image (see defects)."""
    return defects(load_rgb())
