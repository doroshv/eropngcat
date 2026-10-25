"""Step 07b: the PSF width the scene model needs, calibrated by nulling the subtraction residual.

Step 05 fits the PSF on the cores of individual sources. That width (sigma_sky = 4.55') is too
narrow for *subtraction*: with it the scene model over-subtracts every point source, leaving a
donut - a negative centre of -0.13 of the source's own peak and a positive ring beyond 10'. The
width that matters for subtracting is the one that makes that residual vanish, and it can only be
measured through the full scene fit, with the background and the neighbours in place.

So this step scans sigma, runs the real scene solve on a sample of tiles, and stacks the central
residual of isolated unsaturated point sources. The adopted sigma is where that crosses zero,
interpolated between the two bracketing grid points. Measured per channel: they agree to well
within the grid spacing, so one shared value is adopted (`scene.Scene` carries a single covariance
and cannot represent three anyway).

Outputs (out/07b_psf_scene/): psf_scene.json
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from epngcat.config import OUT, SEED, WORKERS, load_json, save_json, step_dir
from epngcat.detect import SCALE, linearize
from epngcat.patches import native_pixels, tile_table
from epngcat.pipeline import build_scene, tile_state
from epngcat.projection import load_alignment

OUTDIR = step_dir("07b_psf_scene")
ARCMIN = SCALE * 60
SIGMA_GRID = (4.6, 5.1, 5.6, 6.1)  # arcmin, sky-space PSF sigma
N_TILES = 120
SNR_MIN = 8.0  # sources fainter than this add noise, not signal, to the stack
R_CORE = 4.0  # arcmin, the radius the residual is measured over


def _tile_rows(tid):
    """Central residual of every clean point source of one tile, for each sigma on the grid."""
    warnings.filterwarnings("ignore")
    det_params = load_json(OUT / "07_detect_optimize" / "detection_params.json")["params"]
    calib = load_json(OUT / "05_calibrate" / "calibration.json")
    calib["scene_psf"] = calib["psf"]
    tiles = pd.DataFrame(tile_table())
    row = tiles[tiles.tile == tid]
    if not len(row):
        return []
    try:
        st = tile_state(row.iloc[0].to_dict(), det_params, calib)
    except Exception:
        return []
    patch, det, core = st["patch"], st["det"], st["core"]
    if not core.any():
        return []
    _, _, nx, ny, nrgb = native_pixels(load_alignment(), patch)
    lin = linearize(patch.rgb.astype(np.float32), det_params["stretch"])
    sat_map = (patch.rgb.astype(np.float32) >= 250).any(-1)
    npix = lin.shape[0]
    yy, xx = np.mgrid[:npix, :npix]
    xs, ys = st["fit"][0], st["fit"][1]

    resid = {}
    for sigma in SIGMA_GRID:
        sc = build_scene(
            patch, det, nx, ny, nrgb, det_params["stretch"], {"t2": (sigma / ARCMIN) ** 2, "s2": calib["psf"]["s2"]}
        )
        amps, coefs = sc.solve(xs, ys)
        bkg, src = sc.render(xs, ys, amps, coefs)
        resid[sigma] = lin - bkg - src

    out = []
    for i in np.flatnonzero(core & (det.sat_area.values == 0) & (det.snr.values >= SNR_MIN)):
        r = np.hypot(xx - xs[i], yy - ys[i]) * ARCMIN
        far = (r > 35) & (r < 50) & ~sat_map
        centre = (r < R_CORE) & ~sat_map
        if far.sum() < 200 or centre.sum() < 3:
            continue
        rec = {"tile": tid}
        ok = True
        for k, ch in enumerate("RGB"):
            peak = float(np.nanmax(lin[..., k][centre])) - float(np.nanmedian(lin[..., k][far]))
            if not np.isfinite(peak) or peak <= 0:
                ok = False
                break
            for sigma in SIGMA_GRID:
                v = resid[sigma][..., k]
                rec[f"{sigma}_{ch}"] = float((np.nanmedian(v[centre]) - np.nanmedian(v[far])) / peak)
        if ok:
            out.append(rec)
    return out


def zero_crossing(sigmas, values):
    """Linear interpolation to the sigma where the stacked residual changes sign."""
    for a, b, va, vb in zip(sigmas[:-1], sigmas[1:], values[:-1], values[1:]):
        if va <= 0 <= vb or vb <= 0 <= va:
            return float(a + (b - a) * (0 - va) / (vb - va))
    return float("nan")


def main():
    """Run step 07b."""
    cat = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    de = cat[cat.de & (cat.split == "train") & (cat.coverage > 0.9) & (np.abs(cat.b_c) > 10)]
    tids = sorted(de.sample(min(N_TILES, len(de)), random_state=SEED).tile)
    print(f"{len(tids)} tiles, sigma grid {SIGMA_GRID}, workers {WORKERS}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for j, r in enumerate(ex.map(_tile_rows, tids), 1):
            rows += r
            if j % 20 == 0:
                print(f"  {j}/{len(tids)} tiles, {len(rows)} sources", flush=True)
    d = pd.DataFrame(rows)
    res = {
        "n_sources": int(len(d)),
        "split": "train",
        "tile_ids": tids,
        "n_tiles": len(tids),
        "sigma_grid": list(SIGMA_GRID),
        "r_core_arcmin": R_CORE,
        "snr_min": SNR_MIN,
        "stack": {},
        "sigma_null": {},
    }
    print(f"\n{'channel':>8s} " + " ".join(f"{s:>9.1f}" for s in SIGMA_GRID) + f" {'sigma_null':>11s}")
    for ch in "RGB":
        vals = [float(np.nanmedian(d[f"{s}_{ch}"])) for s in SIGMA_GRID]
        res["stack"][ch] = vals
        res["sigma_null"][ch] = zero_crossing(SIGMA_GRID, vals)
        print(f"{ch:>8s} " + " ".join(f"{v:9.3f}" for v in vals) + f" {res['sigma_null'][ch]:11.2f}")
    adopted = float(np.nanmean([res["sigma_null"][c] for c in "RGB"]))
    res["sky_sigma_arcmin"] = adopted
    res["t2"] = (adopted / ARCMIN) ** 2
    res["spread_arcmin"] = float(
        np.nanmax([res["sigma_null"][c] for c in "RGB"]) - np.nanmin([res["sigma_null"][c] for c in "RGB"])
    )
    calib = load_json(OUT / "05_calibrate" / "calibration.json")
    print(
        f"\nadopted sigma {adopted:.2f}' (step 05 gives {calib['psf']['sky_sigma_arcmin']:.2f}'), "
        f"channel spread {res['spread_arcmin']:.2f}'"
    )
    save_json(res, OUTDIR / "psf_scene.json")


if __name__ == "__main__":
    main()
