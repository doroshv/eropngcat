"""Source detection on tangent-plane patches.

A detection run is fully described by a parameter dict (see DEFAULTS); the same code serves the
baseline (raw luminance) and the tuned pipeline (linearised, channel-weighted image).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

from epngcat.config import WORKERS
from epngcat.patches import SAT, core_halfwidth, in_tile, make_patch
from epngcat.projection import load_alignment

SCALE = 2 / 60  # deg per patch pixel (native image pixel is ~5.3 arcmin at the centre)
PAD = 0.6  # deg around the tile core

DEFAULTS = {
    "stretch": None,  # None = raw values; else {"A": [aR, aG, aB], "v0": v0}
    "weights": [1.0, 1.0, 1.0],  # channel weights of the detection image
    "bkg_size": 15,  # median-filter box [patch px] for the background
    "kernel_sigma": 1.5,  # isotropic Gaussian matched-filter sigma [patch px] (used if psf is None)
    "psf": None,  # {"t2": sky-space variance [patch px^2], "s2": native-pixel variance [native px^2]}
    "thresh": 5.0,  # detection threshold in units of the filtered-image noise
    "min_sep": 2,  # half-size of the local-maximum footprint [patch px]
}


def linearize(rgb, stretch):
    """Invert an asinh display stretch: I = sinh((v - v0) / A) per channel (identity if None)."""
    if stretch is None:
        return rgb.astype(np.float64)
    a = np.asarray(stretch["A"], float)
    return np.sinh((rgb - stretch["v0"]) / a)


def psf_cov(jac, psf):
    """PSF covariance in patch pixels: t2 * I + s2 * J^-1 J^-T (J = d native px / d patch px)."""
    ji = np.linalg.inv(jac)
    return psf["t2"] * np.eye(2) + psf["s2"] * ji @ ji.T


def gaussian_kernel(cov, nsig=3.5):
    """Unit-sum elliptical Gaussian kernel for a 2x2 covariance."""
    h = int(np.ceil(nsig * np.sqrt(np.linalg.eigvalsh(cov).max())))
    yy, xx = np.mgrid[-h : h + 1, -h : h + 1]
    q = np.stack([xx, yy], -1)
    k = np.exp(-0.5 * np.einsum("...i,ij,...j->...", q, np.linalg.inv(cov), q))
    return k / k.sum()


def sat_mask(patch):
    """Saturated-pixel mask of a patch (native nearest-neighbour mask if available)."""
    return patch.sat if patch.sat is not None else (patch.rgb >= SAT).any(-1)


def robust_sigma(x):
    """MAD-based standard deviation."""
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def detect_patch(patch, params, lin=None, include_saturated=True):
    """Detect peaks in a patch.

    Parameters
    ----------
    lin : ndarray, optional
        Linear three-channel image used only for detection.  Supplying a halo-subtracted image
        leaves the returned raw pixel values and saturation metadata tied to the original patch.
    include_saturated : bool
        Append one detection per saturated component.  A halo-cleaned second pass sets this false
        and merges the saturated seeds from the original image afterwards.
    """
    lin = linearize(patch.rgb, params["stretch"]) if lin is None else np.asarray(lin, float)
    valid = patch.valid
    bkg = np.stack([ndimage.median_filter(lin[..., k], size=params["bkg_size"], mode="reflect") for k in range(3)], -1)
    res = lin - bkg
    noise = np.array([robust_sigma(res[..., k][valid]) for k in range(3)])
    w = np.asarray(params["weights"], float) / np.where(noise > 0, noise, np.inf)
    det = (res * w).sum(-1)
    det = np.where(valid, det, 0.0)
    if params["psf"] is None:
        filt = ndimage.gaussian_filter(det, params["kernel_sigma"])
    else:
        filt = ndimage.convolve(det, gaussian_kernel(psf_cov(patch.jac, params["psf"])), mode="reflect")
    fnoise = robust_sigma(filt[valid])
    snr = filt / fnoise
    peaks = (snr == ndimage.maximum_filter(snr, size=2 * params["min_sep"] + 1)) & (snr > params["thresh"]) & valid
    # Saturated blobs fragment into several maxima: replace them by one detection per blob.
    sat_map = sat_mask(patch)
    sat_lab, nsat = ndimage.label(ndimage.binary_dilation(sat_map, iterations=1))
    sat_area = np.zeros(nsat + 1)
    if nsat:
        peaks &= ndimage.binary_dilation(sat_lab > 0, iterations=params["min_sep"]) == 0
        sat_area[1:] = ndimage.sum(np.ones_like(snr), sat_lab, index=np.arange(1, nsat + 1))
    iy, ix = np.nonzero(peaks)
    if nsat and include_saturated:
        cy, cx = np.array(ndimage.center_of_mass(sat_map | (sat_lab > 0), sat_lab, np.arange(1, nsat + 1))).T
        iy = np.concatenate([iy, np.round(cy).astype(int)])
        ix = np.concatenate([ix, np.round(cx).astype(int)])
    elif nsat:
        cy = cx = np.empty(0)
    # Sub-pixel position: 3x3 centroid of the filtered image.
    xs, ys = ix.astype(float), iy.astype(float)
    nsat_det = nsat if include_saturated else 0
    npk = len(ix) - nsat_det
    if nsat_det:
        xs[npk:], ys[npk:] = cx, cy
    inner = (ix > 0) & (iy > 0) & (ix < patch.npix - 1) & (iy < patch.npix - 1)
    inner[npk:] = False
    for k in np.nonzero(inner)[0]:
        cut = np.clip(filt[iy[k] - 1 : iy[k] + 2, ix[k] - 1 : ix[k] + 2], 0, None)
        if cut.sum() > 0:
            xs[k] += (cut.sum(0) * [-1, 0, 1]).sum() / cut.sum()
            ys[k] += (cut.sum(1) * [-1, 0, 1]).sum() / cut.sum()
    raw = patch.rgb[iy, ix]
    return {
        "x": xs,
        "y": ys,
        "snr": snr[iy, ix],
        "raw_R": raw[:, 0],
        "raw_G": raw[:, 1],
        "raw_B": raw[:, 2],
        "bkg_R": bkg[iy, ix, 0],
        "bkg_G": bkg[iy, ix, 1],
        "bkg_B": bkg[iy, ix, 2],
        "noise_R": np.full(len(ix), noise[0]),
        "noise_G": np.full(len(ix), noise[1]),
        "noise_B": np.full(len(ix), noise[2]),
        "saturated": sat_map[iy, ix] | (raw >= SAT).any(-1),
        "sat_area": np.concatenate([np.zeros(npk), sat_area[1:]]) if include_saturated else np.zeros(npk),
    }


def detection_frame(tile, patch, detections):
    """Add tile and sky coordinates to a detection dictionary."""
    l, b = patch.pix2sky(detections["x"], detections["y"])
    out = pd.DataFrame(detections)
    out.insert(0, "b", b)
    out.insert(0, "l", l)
    out.insert(0, "tile", tile["tile"])
    return out, in_tile(l, b, tile)


_PROJ = None


def tile_patch(tile, scale=SCALE, pad=PAD):
    """Patch covering a tile (a mapping with tile_table fields) plus padding."""
    global _PROJ
    if _PROJ is None:
        _PROJ = load_alignment()
    return make_patch(_PROJ, tile["l_c"], tile["b_c"], core_halfwidth(tile) + pad, scale)


def detect_tile_full(tile, params):
    """Detect sources in a tile patch. Returns (patch, all detections incl. padding, core mask)."""
    params = {**DEFAULTS, **params}
    patch = tile_patch(tile)
    d = detect_patch(patch, params)
    out, core = detection_frame(tile, patch, d)
    return patch, out, core


def detect_tile(args):
    """Detect sources in one tile; keep those whose position lies in the tile core."""
    tile, params = args
    _, det, core = detect_tile_full(tile, params)
    return det[core].reset_index(drop=True)


def detect_tiles(tiles, params, workers=None):
    """Run detection over a tile DataFrame in parallel."""
    rows = [(t._asdict(), params) for t in tiles.itertuples(index=False)]
    workers = workers or WORKERS
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(detect_tile, rows, chunksize=16))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
