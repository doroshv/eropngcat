"""Step 16: all-sky HEALPix maps assembled from the tile products.

Two maps, each with the three channels kept separate and in linear image units (the asinh display
stretch of the press image inverted, I = sinh((v - v0) / A)), so that surface brightnesses add:

  healpix_data_nsideN.fits   the image itself, reprojected without masking
  healpix_clean_nsideN.fits  the same with fitted sources and bright-source halos subtracted, for
                             large-scale emission; clipped cores contain the counterfactual fitted
                             background written by step 14 rather than holes; only pixels genuinely
                             absent from the rendered sky near the l=180 outline are hp.UNSEEN

Both are Galactic, RING ordered, with columns R, G, B. The tile VALID and MODELLED planes are not
used as HEALPix masks: image defects, saturated cores and other modelled pixels therefore remain in
the maps. Missing tile products are an error rather than another class of pixels to interpolate.

Values are surface brightness in counts/s/arcmin2, converted from linearised image units with the
step-15 calibration (rate = slope * I + offset per channel, in the HiPS band, fitted on the DE half
and carried to RU). The constants and their uncertainty go into the header, so the maps can be put
back into linear units; --linear writes those instead and skips the calibration entirely.

Every HEALPix pixel is taken from the one tile whose core contains it (patches.tile_of), so the
0.6 deg tile overlaps neither double-count nor leave seams; pixels are grouped by tile row, so each
tile file is opened once. Sampling is bilinear in the tile's own TAN frame. nside 1024 (3.4')
oversamples the ~5.3' resolution of the image; healpy.ud_grade goes coarser and averages properly.

A third file, healpix_clean_filled_nsideN.fits, is the clean map with the rendered-footprint gap
interpolated from the finest coarser scale that still sees sky and then smoothed to the image PSF.
It is continuous, so it can be transformed or rendered as a colour composite, while the unfilled
clean map remains the primary product.

Outputs (out/16_healpix/): healpix_data_nsideN.fits, healpix_clean_nsideN.fits,
healpix_clean_filled_nsideN.fits
"""

import healpy as hp
import numpy as np
from astropy.io import fits
from scipy import ndimage

from epngcat.config import OUT, load_json, step_dir
from epngcat.detect import SCALE, linearize
from epngcat.image import sky_mask
from epngcat.patches import NROW, ROW_H, tan_forward, tile_of
from epngcat.projection import load_alignment

OUTDIR = step_dir("16_healpix")
NSIDE = 1024
PLANES = {"data": "DATA", "clean": "SUB"}  # output name -> tile FITS extension
FILL_NSIDE_MIN = 8  # coarsest scale the gap filling of the companion map falls back to (7.3 deg)


def row_pixels(nside, row):
    """HEALPix pixels whose centres fall in tile row `row`, with their (l, b) and tile id."""
    b_lo, b_hi = -90 + row * ROW_H, -90 + (row + 1) * ROW_H
    pix = hp.query_strip(nside, np.radians(90 - b_hi), np.radians(90 - b_lo), inclusive=True)
    theta, phi = hp.pix2ang(nside, pix)
    l, b = np.degrees(phi) % 360, 90 - np.degrees(theta)
    keep = (b >= b_lo) & (b < b_hi)  # the inclusive strip overshoots; the centre decides the row
    if row == NROW - 1:
        keep |= b >= b_hi  # the pole itself belongs to the top row
    pix, l, b = pix[keep], l[keep], b[keep]
    return pix, l, b, tile_of(l, b)


def sample_tile(path, l, b, stretch):
    """Linearised tile planes at sky positions, without applying tile masks."""
    with fits.open(path) as hdul:
        lc, bc = hdul[0].header["L_CEN"], hdul[0].header["B_CEN"]
        planes = {name: hdul[ext].data for name, ext in PLANES.items()}
    shape = next(iter(planes.values())).shape
    c = (shape[-1] - 1) / 2
    xi, eta = tan_forward(l, b, lc, bc)
    yx = [eta / SCALE + c, xi / SCALE + c]
    out = {}
    for name, cube in planes.items():
        v = np.stack([ndimage.map_coordinates(p, yx, order=1, mode="nearest") for p in cube], -1)
        out[name] = linearize(v, stretch)
    return out


def rendered_coverage(l, b, projection=None, footprint=None):
    """Pixels actually present inside the original rendered-sky outline.

    This deliberately excludes the processing masks in ``Patch.valid``. In particular, saturated
    cores and pixels flagged by the frozen-line detector remain covered. The mask describes only
    the true gap where the split projection falls outside the drawn sky near l=180.
    """
    projection = load_alignment() if projection is None else projection
    footprint = sky_mask() if footprint is None else np.asarray(footprint)
    px, py = projection.sky2pix(l, b)
    return ndimage.map_coordinates(footprint.astype(np.uint8), [py, px], order=0, mode="constant", cval=0).astype(bool)


def _coarse_mean(v, valid, nlow):
    """Mean of the valid pixels of `v` inside each nside-`nlow` pixel (NaN where none are valid)."""
    num = hp.ud_grade(np.where(valid, v, 0.0), nlow)
    den = hp.ud_grade(valid.astype(np.float64), nlow)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def fill_holes(v, nside, nmin=FILL_NSIDE_MIN):
    """Fill the NaN pixels of one channel from progressively coarser averages of the valid ones.

    A hole is filled at the finest scale that still sees valid sky around it. In normal pipeline
    output these holes are only the rendered-footprint gap near l=180; source cores are modelled
    upstream and are not holes. Nothing measured or modelled is altered.
    """
    out = v.copy()
    nlow = nside // 2
    while nlow >= nmin and not np.isfinite(out).all():
        coarse = hp.ud_grade(_coarse_mean(out, np.isfinite(out), nlow), nside)
        out = np.where(np.isfinite(out), out, coarse)
        nlow //= 2
    return np.where(np.isfinite(out), out, np.nanmedian(v))


def main(nside=NSIDE, linear=False):
    """Run step 16."""
    stretch = load_json(OUT / "07_detect_optimize" / "detection_params.json")["params"]["stretch"]
    units = None if linear else load_json(OUT / "15_units" / "units.json")
    tiles = OUT / "14_tiles" / "fits"
    maps = {name: np.full((hp.nside2npix(nside), 3), np.nan, np.float32) for name in PLANES}
    coverage = np.zeros(hp.nside2npix(nside), bool)
    projection, footprint = load_alignment(), sky_mask()
    missing = []
    for row in range(NROW):
        pix, l, b, tid = row_pixels(nside, row)
        if not len(pix):  # a row narrower than a HEALPix pixel holds no pixel centre
            continue
        covered = rendered_coverage(l, b, projection, footprint)
        order = np.argsort(tid, kind="stable")
        pix, l, b, tid, covered = pix[order], l[order], b[order], tid[order], covered[order]
        coverage[pix] = covered
        for start, stop in zip(*_runs(tid)):
            path = tiles / f"tile_{tid[start]:05d}.fits"
            if not path.exists():
                missing.append(path)
                continue
            s = slice(start, stop)
            sampled = sample_tile(path, l[s], b[s], stretch)
            for name, v in sampled.items():
                if name == "clean":
                    v = np.where(covered[s, None], v, np.nan)
                maps[name][pix[s]] = v
        print(f"row {row + 1}/{NROW}: {len(pix)} pixels", flush=True)

    if missing:
        example = ", ".join(str(p) for p in missing[:3])
        raise FileNotFoundError(f"{len(missing)} tile products are missing; refusing to fill them: {example}")
    if not np.isfinite(maps["data"]).all():
        raise ValueError("non-finite pixels in the unmasked DATA map")
    bad_clean = coverage[:, None] & ~np.isfinite(maps["clean"])
    if bad_clean.any():
        raise ValueError(f"{bad_clean.sum()} non-finite CLEAN values inside the rendered footprint")

    head = [
        ("STRETCH", "I = sinh((v - v0) / A)", "display stretch inverted per channel"),
        ("V0", stretch["v0"]),
        *[(f"A_{c}", stretch["A"][k]) for k, c in enumerate("RGB")],
    ]
    if units is not None:
        head += [("CALIB", "rate = SLOPE_c * I + OFFS_c", "step 15, HiPS band, fitted on DE")]
        for c in "RGB":
            u = units["channels"][c]
            head += [
                (f"SLOPE_{c}", u["slope"], f"ct/s/arcmin2 per linear unit, {u['band']}"),
                (f"OFFS_{c}", u["offset"], "ct/s/arcmin2"),
                (f"SERR_{c}", u["slope_frac_err"], "fractional 1-sigma on SLOPE"),
            ]
    for name, m in maps.items():
        path = OUTDIR / f"healpix_{name}_nside{nside}.fits"
        v = m.copy()
        if units is not None:
            for k, c in enumerate("RGB"):
                u = units["channels"][c]
                v[:, k] = u["slope"] * v[:, k] + u["offset"]
        masking = "NONE" if name == "data" else "FOOTPRINT"
        hp.write_map(
            path,
            [np.where(np.isfinite(v[:, k]), v[:, k], hp.UNSEEN) for k in range(3)],
            column_names=["R", "G", "B"],
            column_units=["linear" if units is None else "ct/s/arcmin2"] * 3,
            coord="G",
            overwrite=True,
            dtype=[np.float32] * 3,
            extra_header=[
                ("PLANE", PLANES[name], "tile FITS extension the map was built from"),
                ("MASKING", masking, "no source/core masks; CLEAN excludes absent footprint only"),
                *head,
            ],
        )
        seen = np.isfinite(m[:, 1]).mean()
        print(f"{path.name}: {seen:.3f} of the sphere covered")
        if name != "clean":
            continue
        # A companion map with the genuine rendered-footprint gap filled. Missing tile products
        # and non-finite covered pixels failed above, so no processing failure can enter this fill.
        gaps = ~coverage
        filled = np.stack([fill_holes(v[:, k], nside) for k in range(3)], -1)
        fwhm = load_json(OUT / "05_calibrate" / "calibration.json")["psf"]["sky_sigma_arcmin"] * 2.355
        filled = np.stack([hp.smoothing(filled[:, k], fwhm=np.radians(fwhm / 60)) for k in range(3)], -1)
        fpath = OUTDIR / f"healpix_clean_filled_nside{nside}.fits"
        hp.write_map(
            fpath,
            [filled[:, k] for k in range(3)],
            column_names=["R", "G", "B"],
            column_units=["linear" if units is None else "ct/s/arcmin2"] * 3,
            coord="G",
            overwrite=True,
            dtype=[np.float32] * 3,
            extra_header=[
                ("PLANE", "SUB", "tile FITS extension the map was built from"),
                ("MASKING", "NONE", "rendered-footprint gap interpolated; no source/core masks"),
                ("FILLED", float(gaps.mean()), "fraction of pixels interpolated, not measured"),
                ("SMOOTH", float(fwhm), "Gaussian FWHM [arcmin] applied after filling"),
                *head,
            ],
        )
        print(f"{fpath.name}: {gaps.mean():.3f} of the sphere interpolated, smoothed by {fwhm:.1f} arcmin FWHM")


def _runs(a):
    """Start and stop indices of the runs of equal values in a sorted array."""
    edge = np.flatnonzero(np.diff(a)) + 1
    return np.concatenate([[0], edge]), np.concatenate([edge, [len(a)]])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nside", type=int, default=NSIDE, help=f"HEALPix nside (default {NSIDE})")
    ap.add_argument("--linear", action="store_true", help="write linearised image units, not calibrated rates")
    a = ap.parse_args()
    main(a.nside, a.linear)
