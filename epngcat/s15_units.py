"""Step 15: surface-brightness calibration of the press image against the eRASS1-DE HiPS rate maps.

The step-14 training pairs already put both on one grid: HIRES is the DR1 HiPS rate cube in
counts/s/arcmin2 (bands 0.2-0.5 / 0.5-1 / 1-2 keV) at 0.5'/px, LOWRES the press image in display
units at 2'/px, with every HIRES_FACTOR-th HIRES pixel centred on a LOWRES pixel. Smoothing HIRES
to the press PSF and sampling it onto the LOWRES grid gives a per-pixel pair (I, rate), where
I = sinh((v - v0) / A) is the linearised image of detect.linearize.

The relation is affine, not a pure scale. If the image was rendered as v = A asinh(rate / s) + v0'
and linearised with a v0 that differs by d = (v0' - v0) / A, then

    I = sinh(asinh(rate / s) + d) = (rate / s) cosh d + sqrt(1 + (rate / s)^2) sinh d,

which for rate << s is I = a rate + b. So the fit is rate = slope * I + offset over the faint,
unclipped pixels where that holds, and the offset measures how far the assumed zero point is from
the one the image was rendered with.

Each tile is fitted on its own and the calibration is the median over tiles, with the MAD as the
uncertainty: robust against the few tiles dominated by a bright source or a poor HiPS overlap.

Bands do not match exactly (image R/G/B are nominally 0.3-0.6 / 0.6-1.0 / 1.0-2.3 keV against HiPS
0.2-0.5 / 0.5-1.0 / 1.0-2.0), so the result is an equivalent rate in the HiPS band, and R - the
largest mismatch, and the most sensitive to absorption - is the least well determined.

HiPS covers the DE half only. The constants are carried to RU, which assumes the halves share a
rendering; `de_ru_ratio` measures that at the l = L_SPLIT + 180 boundary, the one of the two that
is not on the rim of the projection.

Outputs (out/15_units/): units.json, units.pdf
"""

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage

from epngcat.config import L_SPLIT, OUT, WORKERS, load_json, save_json, step_dir
from epngcat.detect import linearize
from epngcat.hires import BAND_LABEL, BANDS, HIRES_FACTOR
from epngcat.plotting import figure_size, save_figure

OUTDIR = step_dir("15_units")
MAX_DISPLAY = 200.0  # display units; above this the asinh is too compressed for the affine form
MIN_PIX = 200  # usable pixels a tile needs to be fitted
MIN_R = 0.5  # and the correlation it needs, so flat tiles do not vote on the slope
CLIP = 3.0  # sigma, one round of outlier rejection per tile


def tile_fit(path):
    """Per-channel (slope, offset, r, n) of rate against linearised intensity for one hires file."""
    stretch = load_json(OUT / "07_detect_optimize" / "detection_params.json")["params"]["stretch"]
    sigma = np.sqrt(load_json(OUT / "05_calibrate" / "calibration.json")["psf"]["t2"]) * HIRES_FACTOR
    with fits.open(path) as h:
        cube = h["HIRES"].data.astype(np.float64)
        low = np.moveaxis(h["LOWRES"].data, 0, -1).astype(np.float64)
        valid = h["VALID"].data.astype(bool)
    if not np.isfinite(cube).any():
        return None
    covered = np.isfinite(cube).all(0)[::HIRES_FACTOR, ::HIRES_FACTOR]
    rate = np.stack([ndimage.gaussian_filter(np.nan_to_num(c), sigma) for c in cube], -1)
    rate = rate[::HIRES_FACTOR, ::HIRES_FACTOR]
    lin = linearize(low, stretch)
    keep = covered & valid & (low < MAX_DISPLAY).all(-1)
    if keep.sum() < MIN_PIX:
        return None
    out = []
    for k in range(3):
        x, y = lin[..., k][keep], rate[..., k][keep]
        for _ in range(2):  # fit, reject outliers, refit
            a = np.stack([x, np.ones_like(x)], -1)
            (slope, offset), *_ = np.linalg.lstsq(a, y, rcond=None)
            r = y - (slope * x + offset)
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            if not s > 0:
                break
            good = np.abs(r) < CLIP * s
            if good.sum() < MIN_PIX:
                break
            x, y = x[good], y[good]
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 else np.nan
        out.append((float(slope), float(offset), corr, int(len(x))))
    return out


def de_ru_ratio(stretch, b_step=0.25):
    """Median linear-intensity ratio across the DE/RU meridian at l = L_SPLIT + 180, per channel.

    The other boundary, l = L_SPLIT, lies on the rim of the projection where both sides are
    attenuated, so it cannot test continuity.
    """
    from epngcat.image import load_rgb
    from epngcat.projection import load_alignment

    proj, img = load_alignment(), load_rgb()
    b = np.concatenate([np.arange(-78, -8, b_step), np.arange(8, 78, b_step)])

    def strip(l):
        out = np.empty((len(b), 3))
        for k in range(3):
            px, py = proj.sky2pix_channel(np.full_like(b, l) % 360, b, k)
            out[:, k] = ndimage.map_coordinates(img[..., k], [py, px], order=1, mode="nearest")
        return linearize(out, stretch)

    lo, hi = strip(L_SPLIT + 180 - 0.5), strip(L_SPLIT + 180 + 0.5)
    return np.median(hi / np.clip(lo, 1e-6, None), 0)


def main(limit=None):
    """Run step 15."""
    stretch = load_json(OUT / "07_detect_optimize" / "detection_params.json")["params"]["stretch"]
    paths = sorted((OUT / "14_tiles" / "hires").glob("tile_*_hires.fits"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit("no hires tiles: run step 14 with --hires first")
    print(f"fitting {len(paths)} hires tiles", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        fits_per_tile = [r for r in ex.map(tile_fit, paths, chunksize=8) if r is not None]
    arr = np.array(fits_per_tile)  # (n_tiles, 3, 4)

    cal, summary = {}, {}
    for k, c in enumerate("RGB"):
        slope, offset, corr, n = (arr[:, k, j] for j in range(4))
        ok = (corr > MIN_R) & (n >= MIN_PIX) & np.isfinite(slope)
        s, o = slope[ok], offset[ok]
        med = float(np.median(s))
        cal[c] = {
            "band": BAND_LABEL[BANDS[k]],
            "slope": med,
            "offset": float(np.median(o)),
            "slope_mad": float(1.4826 * np.median(np.abs(s - med))),
            "slope_frac_err": float(1.4826 * np.median(np.abs(s - med)) / abs(med)) if med else np.nan,
            "r_median": float(np.median(corr[ok])),
            "n_tiles": int(ok.sum()),
        }
        summary[c] = s
    ratio = de_ru_ratio(stretch)
    out = {
        "units": "counts/s/arcmin2 = slope * I_linear + offset",
        "stretch": stretch,
        "channels": cal,
        "de_ru_ratio": {c: float(ratio[k]) for k, c in enumerate("RGB")},
        "n_tiles_read": len(fits_per_tile),
        "max_display": MAX_DISPLAY,
    }
    save_json(out, OUTDIR / "units.json")
    for c, v in cal.items():
        print(
            f"{c} ({v['band']}): rate = {v['slope']:.4g} * I + {v['offset']:.4g}  "
            f"+-{100 * v['slope_frac_err']:.0f}%  r={v['r_median']:.2f}  n_tiles={v['n_tiles']}"
        )
    print("DE/RU continuity ratio:", {c: round(float(ratio[k]), 3) for k, c in enumerate("RGB")})

    fig, axs = plt.subplots(1, 3, figsize=figure_size(1, 3))
    for k, (c, s) in enumerate(summary.items()):
        axs[k].hist(s, bins=60, color="C0")
        axs[k].axvline(cal[c]["slope"], color="k", lw=1.5)
        axs[k].set_title(f"{c} ({cal[c]['band']}): {cal[c]['slope']:.3g} +- {100 * cal[c]['slope_frac_err']:.0f}%")
        axs[k].set_xlabel(r"Slope [ct s$^{-1}$ arcmin$^{-2}$ / image unit]")
    fig.suptitle("Step 15: press-image intensity -> eRASS1-DE HiPS rate")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "units.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="fit only the first N hires tiles")
    main(ap.parse_args().limit)
