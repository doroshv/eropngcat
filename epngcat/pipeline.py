"""Final image-only pipeline per tile: detection (step 07 parameters) + measurements."""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from epngcat.brightscene import BrightSourceFit, fit_bright_sources
from epngcat.config import OUT, WORKERS, load_json
from epngcat.detect import SAT, SCALE, detect_patch, detection_frame, linearize, psf_cov, tile_patch
from epngcat.measure import measure_patch
from epngcat.patches import core_halfwidth, in_tile, make_patch, native_pixels, tan_forward
from epngcat.projection import load_alignment
from epngcat.scene import Scene
from epngcat.sky import radec_unit

# The two-pass sigmoid branch is entered only for patches containing a saturated component.
# EPNGCAT_WINGS=0 preserves the previous one-pass path for controlled comparisons.
WINGS = bool(int(os.environ.get("EPNGCAT_WINGS", "1")))

# A real bright-source halo may alter the mean of a tile substantially, but it must not drive a
# macroscopic fraction of the cleaned core below the darkest value representable by the source
# image.  The limits retain Cyg X-1 (the most demanding tested genuine case: 11.1% at the floor,
# 0.64 mean fractional change) while rejecting the red-only diffuse saturation fields that were
# being split into many fictitious sigmoid sources. A later test case with a correctly removed
# blue halo reached 13.3% in an unrelated green edge region, hence the modest 15% allowance.
CLEAN_FLOOR_FRAC_MAX = 0.15
CLEAN_MEAN_CHANGE_MAX = 0.80
BRIGHT_EDGE_MARGIN = 2.0  # patch px; a saturated component cut by the boundary has a biased centroid/area

# Halo fitting is unsafe when diffuse red emission supplies much of the signal in a tile.  This
# threshold was placed at the first visually problematic R-band field (15:59:06 -36:29:15); its
# robust linearised core level is 1.61, while representative clean bright-source fields are <1.
HALO_BACKGROUND_R_MAX = 1.5
HALO_BACKGROUND_QUANTILE = 0.90

# Population relations from the completed DE+RU halo fits.  S is saturated area in arcmin^2.
# They are used only as generous rejection envelopes, not to constrain otherwise credible fits.
HALO_AREA_CAL_MAX = 1350.0
HALO_AMPLITUDE_INTERCEPT = np.array([3.1964, 1.4937, -0.4324])
HALO_AMPLITUDE_SLOPE = np.array([0.022144, 0.011109, 0.014517])
HALO_AMPLITUDE_FLOOR = np.array([3.2, 1.5, 1.5])
HALO_AMPLITUDE_FACTOR = 5.0
HALO_RADIUS_INTERCEPT = 11.25
HALO_RADIUS_SLOPE = 0.0144
HALO_RADIUS_FACTOR = 3.0
HALO_RADIUS_MIN = 30.0
HALO_RADIUS_MAX = 60.0

DEDUP_R = 3.0  # arcmin: same source detected in two neighbouring tile patches


@cache
def snr_catalog():
    """Known SNR positions and conservative mask radii used by the halo gate."""
    path = OUT / "01_catalogs" / "snrcat.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["name", "l", "b", "mask_radius_arcmin"])
    return pd.read_parquet(path, columns=["name", "l", "b", "mask_radius_arcmin"])


def patch_snr_overlaps(patch, catalogue=None, tile=None):
    """Names of SNR masks intersecting the tile (or full patch if no tile is supplied).

    Distances are measured to the nearest edge of the patch in its gnomonic plane.  The explicit
    front-hemisphere test prevents a source on the opposite side of the sky from projecting into
    the finite patch rectangle.
    """
    catalogue = snr_catalog() if catalogue is None else catalogue
    if len(catalogue) == 0:
        return []
    l, b = catalogue.l.values, catalogue.b.values
    xi, eta = tan_forward(l, b, patch.lc, patch.bc)
    lr, br, lc, bc = map(np.radians, (l, b, patch.lc, patch.bc))
    cosc = np.sin(bc) * np.sin(br) + np.cos(bc) * np.cos(br) * np.cos(lr - lc)
    halfwidth = core_halfwidth(tile) if tile is not None else 0.5 * patch.npix * patch.scale
    dx = np.maximum(np.abs(xi) - halfwidth, 0.0)
    dy = np.maximum(np.abs(eta) - halfwidth, 0.0)
    # A gnomonic radius is tan(angular radius); convert the gap back to a sky angle.
    gap_arcmin = np.degrees(np.arctan(np.radians(np.hypot(dx, dy)))) * 60.0
    overlap = (cosc > 0) & (gap_arcmin <= catalogue.mask_radius_arcmin.values)
    return catalogue.name.values[overlap].tolist()


def halo_background_level(patch, tile, stretch, upper_quantile=HALO_BACKGROUND_QUANTILE):
    """Robust R-band level in the tile core, excluding the brightest ten per cent."""
    iy, ix = np.mgrid[: patch.npix, : patch.npix]
    l, b = patch.pix2sky(ix.ravel(), iy.ravel())
    core = in_tile(l, b, tile).reshape(patch.npix, patch.npix) & patch.valid
    red = linearize(patch.rgb.astype(float), stretch)[..., 0]
    values = red[core & np.isfinite(red)]
    if not len(values):
        return np.inf
    ceiling = np.quantile(values, upper_quantile)
    return float(np.median(values[values <= ceiling]))


def halo_gate_diagnostics(patch, tile, stretch, catalogue=None):
    """Pre-fit vetoes for diffuse fields in which a radial halo is not identifiable."""
    background = halo_background_level(patch, tile, stretch)
    snrs = patch_snr_overlaps(patch, catalogue, tile)
    reasons = []
    if background > HALO_BACKGROUND_R_MAX:
        reasons.append("high_r_background")
    if snrs:
        reasons.append("snr_overlap")
    return {
        "stage": "pre_fit_gate",
        "corrupt": bool(reasons),
        "reasons": reasons,
        "r_background": background,
        "r_background_max": HALO_BACKGROUND_R_MAX,
        "snr": snrs,
    }


def reject_implausible_bright(fit, sat_area):
    """Disable source/band halo fits outside conservative population envelopes.

    The radius is the latent sigmoid's ten-per-cent radius.  Amplitude and radius are checked
    separately because clipping can make an enormous latent amplitude look superficially benign.
    """
    area = np.minimum(np.asarray(sat_area, float) * (SCALE * 60.0) ** 2, HALO_AREA_CAL_MAX)
    expected_amplitude = np.maximum(
        HALO_AMPLITUDE_INTERCEPT[None, :] + area[:, None] * HALO_AMPLITUDE_SLOPE[None, :],
        HALO_AMPLITUDE_FLOOR[None, :],
    )
    amplitude_limit = HALO_AMPLITUDE_FACTOR * expected_amplitude
    expected_radius = HALO_RADIUS_INTERCEPT + HALO_RADIUS_SLOPE * area
    radius_limit = np.clip(HALO_RADIUS_FACTOR * expected_radius, HALO_RADIUS_MIN, HALO_RADIUS_MAX)
    radius10 = fit.half_radius + np.log(9.0) * fit.width
    fitted_amplitude = fit.amplitude.copy()
    rejected = fit.success & (
        ~np.isfinite(fitted_amplitude)
        | ~np.isfinite(radius10)
        | (fitted_amplitude > amplitude_limit)
        | (radius10 > radius_limit[:, None])
    )
    fit.amplitude[rejected] = 0.0
    fit.success[rejected] = False
    return {
        "stage": "fit_qa",
        "corrupt": bool(rejected.any()),
        "rejected": rejected.tolist(),
        "amplitude": fitted_amplitude.tolist(),
        "amplitude_limit": amplitude_limit.tolist(),
        "radius10": radius10.tolist(),
        "radius10_limit": radius_limit.tolist(),
    }


def fit_tile_bright(patch, stretch, centers, sat_area, neighbours=(), initial=None, contexts=None, refine=False):
    """Fit each saturated source on an adaptive source-centred patch.

    The ordinary tile padding is intentionally small and often cuts a degree-scale halo off near
    one edge.  Only saturated sources pay for this larger patch; their fitted models are transformed
    back into the tile frame before detection and scene fitting.
    """
    centers = np.asarray(centers, float).reshape(-1, 2)
    sat_area = np.asarray(sat_area, float)
    neighbours = np.asarray(neighbours, float).reshape(-1, 2)
    projection = load_alignment()
    if contexts is None:
        contexts = []
        l, b = patch.pix2sky(centers[:, 0], centers[:, 1])
        for i, (longitude, latitude) in enumerate(zip(l, b)):
            core_radius = np.sqrt(max(sat_area[i], 1.0) / np.pi) * patch.scale * 60.0
            halfwidth = np.clip(4.0 * core_radius / 60.0, 4.0, 8.0)
            contexts.append(make_patch(projection, longitude, latitude, halfwidth, patch.scale))

    amplitude, half_radius, width, baseline, clip_level, clip_width, success, fitted_centers = ([] for _ in range(8))
    for i, halo_patch in enumerate(contexts):
        x0 = y0 = halo_patch.c
        local_neighbours = np.empty((0, 2))
        if len(neighbours):
            nl, nb = patch.pix2sky(neighbours[:, 0], neighbours[:, 1])
            local_neighbours = np.stack(halo_patch.sky2pix(nl, nb), -1)
            inside = (
                (local_neighbours[:, 0] >= 0)
                & (local_neighbours[:, 1] >= 0)
                & (local_neighbours[:, 0] < halo_patch.npix)
                & (local_neighbours[:, 1] < halo_patch.npix)
            )
            local_neighbours = local_neighbours[inside]
        local_initial = None
        if initial is not None:
            il, ib = patch.pix2sky(initial.centers[i, 0], initial.centers[i, 1])
            ix, iy = halo_patch.sky2pix(il, ib)
            local_initial = BrightSourceFit(
                np.array([[ix, iy]]),
                initial.amplitude[i : i + 1].copy(),
                initial.half_radius[i : i + 1].copy(),
                initial.width[i : i + 1].copy(),
                initial.baseline[i : i + 1].copy(),
                initial.clip_level[i : i + 1].copy(),
                initial.clip_width[i : i + 1].copy(),
                initial.success[i : i + 1].copy(),
                np.array([[x0, y0]]),
            )
        fitted = fit_bright_sources(
            halo_patch,
            stretch,
            [[x0, y0]],
            [sat_area[i]],
            neighbours=local_neighbours,
            initial=local_initial,
            refine_centroid=refine,
        )
        fl, fb = halo_patch.pix2sky(*fitted.centers[0])
        fitted_centers.append(patch.sky2pix(fl, fb))
        amplitude.append(fitted.amplitude[0])
        half_radius.append(fitted.half_radius[0])
        width.append(fitted.width[0])
        baseline.append(fitted.baseline[0])
        clip_level.append(fitted.clip_level[0])
        clip_width.append(fitted.clip_width[0])
        success.append(fitted.success[0])
    result = BrightSourceFit(
        np.asarray(fitted_centers),
        np.asarray(amplitude),
        np.asarray(half_radius),
        np.asarray(width),
        np.asarray(baseline),
        np.asarray(clip_level),
        np.asarray(clip_width),
        np.asarray(success),
        centers.copy() if initial is None else initial.initial_centers.copy(),
    )
    return result, contexts


def reject_truncated_bright(fit, npix, margin=BRIGHT_EDGE_MARGIN):
    """Disable sigmoid components whose saturated blob is truncated by the patch boundary."""
    distance = np.min(
        np.stack(
            [
                fit.centers[:, 0],
                fit.centers[:, 1],
                npix - 1 - fit.centers[:, 0],
                npix - 1 - fit.centers[:, 1],
            ],
            axis=1,
        ),
        axis=1,
    )
    rejected = distance < margin
    fit.amplitude[rejected] = 0.0
    fit.success[rejected] = False
    return rejected


def load_params():
    """Tuned detection parameters and calibration."""
    det = load_json(OUT / "07_detect_optimize" / "detection_params.json")["params"]
    calib = load_json(OUT / "05_calibrate" / "calibration.json")
    return det, calib


def scene_psf(calib):
    """PSF for the scene model: step 07b's subtraction-nulling width where it exists.

    Step 05 fits the width of an isolated core; the scene needs the width that leaves no residual
    once the background spline and the neighbours are fitted alongside, and the two differ by ~25%
    (4.55' against ~5.9'). Measurement keeps the step-05 PSF - it is fitting a source, not removing
    one - and the scene amplitudes reach the flux models through the `A_*_s` features either way.
    """
    psf = {"t2": calib["psf"]["t2"], "s2": calib["psf"]["s2"]}
    # Step 07b must start from its declared upstream PSF, even on a rerun.
    if "scene_psf" in calib:
        return {key: calib["scene_psf"][key] for key in ("t2", "s2")}
    path = OUT / "07b_psf_scene" / "psf_scene.json"
    if path.exists():
        psf["t2"] = load_json(path)["t2"]
    return psf


def cleaning_diagnostics(raw, clean, core, stretch):
    """Diagnose a catastrophic halo subtraction on the tile core.

    ``raw`` and ``clean`` are linear image units.  The display floor is not zero in those units
    because the fitted asinh stretch has a non-zero ``v0``.  A second guard compares the average
    absolute change with the average raw signal, catching broad over-subtractions that remain just
    above that floor.
    """
    raw, clean, core = np.asarray(raw), np.asarray(clean), np.asarray(core, bool)
    floor = linearize(np.zeros((1, 1, 3)), stretch).reshape(3)
    if not core.any():
        return {"corrupt": True, "floor_fraction": [1.0] * 3, "mean_change_fraction": [np.inf] * 3}
    floor_fraction = np.mean(clean[core] <= floor, axis=0)
    mean_change = np.mean(np.abs(raw[core] - clean[core]), axis=0)
    raw_mean = np.maximum(np.mean(np.abs(raw[core]), axis=0), floor)
    mean_change_fraction = mean_change / raw_mean
    corrupt = bool(
        np.any(floor_fraction > CLEAN_FLOOR_FRAC_MAX) or np.any(mean_change_fraction > CLEAN_MEAN_CHANGE_MAX)
    )
    return {
        "corrupt": corrupt,
        "floor_fraction": floor_fraction.tolist(),
        "mean_change_fraction": mean_change_fraction.tolist(),
    }


def tile_state(tile, det_params, calib, use_wings=None):
    """Full per-tile processing state: patch, all detections (incl. padding) with measurements, core mask, scene.

    A tile with no detection in its core still gets a scene: the background is fitted either way,
    and the tile products (step 14) and the HEALPix maps (step 16) need it.
    """
    use_wings = WINGS if use_wings is None else bool(use_wings)
    patch = tile_patch(tile)
    raw = detect_patch(patch, det_params)
    raw_sat = np.asarray(raw["sat_area"]) > 0
    bright = None
    halo_rejection = None
    if use_wings and raw_sat.any():
        gate = halo_gate_diagnostics(patch, tile, det_params["stretch"])
        if gate["corrupt"]:
            fallback = tile_state(tile, det_params, calib, use_wings=False)
            fallback["halo_rejected"] = True
            fallback["halo_rejection"] = gate
            return fallback
        # Pass 1: use only saturated blobs. Faint detections from the raw halo-contaminated image
        # are deliberately discarded.
        sat_seed = {name: np.asarray(value)[raw_sat] for name, value in raw.items()}
        seed_xy = np.stack([sat_seed["x"], sat_seed["y"]], -1)
        bright0, halo_contexts = fit_tile_bright(patch, det_params["stretch"], seed_xy, sat_seed["sat_area"])
        reject_truncated_bright(bright0, patch.npix)
        initial_qa = reject_implausible_bright(bright0, sat_seed["sat_area"])
        if not bright0.success.any():
            fallback = tile_state(tile, det_params, calib, use_wings=False)
            fallback["halo_rejected"] = True
            fallback["halo_rejection"] = initial_qa
            return fallback
        detection_image = linearize(patch.rgb, det_params["stretch"]) - bright0.render(patch.npix, patch.scale)
        faint = detect_patch(patch, det_params, lin=detection_image, include_saturated=False)
        combined = {name: np.concatenate([np.asarray(faint[name]), np.asarray(sat_seed[name])]) for name in faint}
        det, core = detection_frame(tile, patch, combined)
        preliminary_keep = (det.snr >= det_params["thresh"]) | (det.sat_area > 0)
        neighbours = det.loc[preliminary_keep & (det.sat_area == 0), ["x", "y"]].values
        sat_rows = np.flatnonzero(det.sat_area.values > 0)
        # Pass 2: mask the newly detected neighbours and free the common bright-source centroid.
        bright, _ = fit_tile_bright(
            patch,
            det_params["stretch"],
            det.loc[sat_rows, ["x", "y"]].values,
            det.loc[sat_rows, "sat_area"].values,
            neighbours=neighbours,
            initial=bright0,
            contexts=halo_contexts,
            refine=True,
        )
        reject_truncated_bright(bright, patch.npix)
        final_qa = reject_implausible_bright(bright, det.loc[sat_rows, "sat_area"].values)
        if not bright.success.any():
            fallback = tile_state(tile, det_params, calib, use_wings=False)
            fallback["halo_rejected"] = True
            fallback["halo_rejection"] = final_qa
            return fallback
        if final_qa["corrupt"]:
            halo_rejection = final_qa
        det.loc[sat_rows, ["x", "y"]] = bright.centers
        l, b = patch.pix2sky(det.x.values, det.y.values)
        det.loc[:, "l"], det.loc[:, "b"] = l, b
        core = in_tile(l, b, tile)
    else:
        det, core = detection_frame(tile, patch, raw)
    keep = (det.snr >= det_params["thresh"]) | (det.sat_area > 0)
    det, core = det[keep].reset_index(drop=True), core[keep.values]
    psf = {"t2": calib["psf"]["t2"], "s2": calib["psf"]["s2"]}
    _, _, nx, ny, nrgb = native_pixels(load_alignment(), patch)
    meas = pd.DataFrame(measure_patch(patch, det, det_params["stretch"], psf, native=(nx, ny, nrgb)))
    meas = meas.drop(columns=[c for c in ("sat_area", "snr") if c in meas])
    # Measurement-level background/noise/raw values supersede the detection-level ones.
    det = det.drop(columns=[c for c in det if c in meas])
    det = pd.concat([det, meas], axis=1)
    sc = build_scene(
        patch,
        det,
        nx,
        ny,
        nrgb,
        det_params["stretch"],
        scene_psf(calib),
        censored=bright is None,
    )
    x0, y0 = start_positions(det)
    sat = det.sat_area.values > 0
    raw_native = sc.lin.copy()
    if bright is not None:
        sc.lin -= bright.component(sc.nx, sc.ny, patch.scale)
    # A saturated source is represented by its observed sigmoid, not by a second Gaussian.
    bright_rows = np.flatnonzero(sat)
    modeled_bright = np.zeros(len(det), bool)
    if bright is not None:
        modeled_bright[bright_rows] = bright.success.any(1)
    fit_rows = np.flatnonzero(~modeled_bright)
    fixed_coefs = None
    if bright is not None:
        # Anchor the flexible spline outside the bright transition and all point-source cores.
        # Otherwise a small halo mismatch is cheaply absorbed as a broad central depression and
        # reappears as a dark hole in the counterfactual clean image.
        background_use = np.ones(len(sc.lin), bool)
        arcmin = patch.scale * 60.0
        max_cut = 0.35 * patch.npix * arcmin
        for i in np.flatnonzero(bright.success.any(1)):
            x, y = bright.centers[i]
            cut = min(float(np.max(bright.half_radius[i] + 3.0 * bright.width[i])), max_cut)
            background_use &= np.hypot(sc.nx[1] - x, sc.ny[1] - y) * arcmin > cut
        for x, y in zip(x0, y0):
            background_use &= np.hypot(sc.nx[1] - x, sc.ny[1] - y) * arcmin > 12.0
        if background_use.sum() >= 100:
            fixed_coefs = sc.fit_background(background_use)
            native_bkg = np.stack(
                [sc.model([], [], np.zeros((0, 3)), fixed_coefs, k, sc.nx[k], sc.ny[k], "bkg") for k in range(3)],
                -1,
            )
            sc.lin = raw_native - bright.component(sc.nx, sc.ny, patch.scale, background=native_bkg)
            fixed_coefs = sc.fit_background(background_use)
    fx, fy, famps, coefs = sc.refine(
        np.asarray(x0)[fit_rows],
        np.asarray(y0)[fit_rows],
        fixed=sat[fit_rows],
        fixed_coefs=fixed_coefs,
    )
    xs, ys = np.asarray(x0, float), np.asarray(y0, float)
    amps = np.zeros((len(det), 3))
    xs[fit_rows], ys[fit_rows], amps[fit_rows] = fx, fy, famps
    if bright is not None:
        good = bright.success.any(1)
        xs[bright_rows[good]], ys[bright_rows[good]] = bright.centers[good].T
        missing = np.zeros_like(amps, dtype=bool)
        # A row with no successful sigmoid was already included in the ordinary PSF solve. For a
        # mixed-success row, fill only its failed bands with a Gaussian at the common centroid.
        missing[bright_rows] = ~bright.success & good[:, None]
        amps = sc.fill_missing_source_amplitudes(xs, ys, amps, coefs, missing)
    cols = {"x_scene": xs, "y_scene": ys}
    for k, c in enumerate("RGB"):
        cols[f"A_{c}_s"] = amps[:, k]
        cols[f"sbkg_{c}"] = sc.model(fx, fy, famps, coefs, k, xs, ys, "bkg")
        cols[f"H_A_{c}"] = np.full(len(det), np.nan)
        cols[f"H_R50_{c}"] = np.full(len(det), np.nan)
        cols[f"H_WIDTH_{c}"] = np.full(len(det), np.nan)
        cols[f"H_CLIP_WIDTH_{c}"] = np.full(len(det), np.nan)
        cols[f"H_OK_{c}"] = np.zeros(len(det), bool)
    cols["halo_shift"] = np.full(len(det), np.nan)
    cols["halo_ok"] = np.zeros(len(det), bool)
    if bright is not None:
        for k, c in enumerate("RGB"):
            cols[f"H_A_{c}"][sat] = bright.amplitude[:, k]
            cols[f"H_R50_{c}"][sat] = bright.half_radius[:, k]
            cols[f"H_WIDTH_{c}"][sat] = bright.width[:, k]
            cols[f"H_CLIP_WIDTH_{c}"][sat] = bright.clip_width[:, k]
            cols[f"H_OK_{c}"][sat] = bright.success[:, k]
        cols["halo_shift"][sat] = np.hypot(*(bright.centers - bright.initial_centers).T) * SCALE * 60.0
        cols["halo_ok"][sat] = bright.success.any(1)
    det = pd.concat([det, pd.DataFrame(cols, index=det.index)], axis=1)
    for est in ("cen", "psf", "psfw", "cenn", "psfn", "psfnw", "scene"):
        if f"x_{est}" not in det:
            det[f"x_{est}"] = np.nan
            det[f"y_{est}"] = np.nan
        ok = det[f"x_{est}"].notna().values
        lb = np.full((2, len(det)), np.nan)
        if ok.any():
            lb[:, ok] = patch.pix2sky(det[f"x_{est}"].values[ok], det[f"y_{est}"].values[ok])
        det[f"l_{est}"], det[f"b_{est}"] = lb
    det["l_det"], det["b_det"] = det.l, det.b
    bkg_lin, psf_lin = sc.render(xs, ys, amps, coefs)
    halo_lin = np.zeros_like(bkg_lin) if bright is None else bright.render(patch.npix, patch.scale, background=bkg_lin)
    source_lin = psf_lin + halo_lin
    raw_lin = linearize(patch.rgb.astype(float), det_params["stretch"])
    clean_lin = raw_lin - source_lin
    modelled = patch.rgb >= SAT
    clean_lin[modelled] = bkg_lin[modelled]
    if bright is not None:
        iy, ix = np.mgrid[: patch.npix, : patch.npix]
        l, b = patch.pix2sky(ix.ravel(), iy.ravel())
        core_pixels = in_tile(l, b, tile).reshape(patch.npix, patch.npix) & patch.valid
        clean_qa = cleaning_diagnostics(
            raw_lin,
            clean_lin,
            core_pixels,
            det_params["stretch"],
        )
        clean_qa["stage"] = "clean_image_qa"
        if clean_qa["corrupt"]:
            fallback = tile_state(tile, det_params, calib, use_wings=False)
            fallback["halo_rejected"] = True
            fallback["halo_rejection"] = clean_qa
            return fallback
    return {
        "patch": patch,
        "det": det,
        "core": core,
        "scene": sc,
        "fit": (fx, fy, famps, coefs),
        "bright": bright,
        "background": bkg_lin,
        "source": source_lin,
        "clean": clean_lin,
        "modelled": modelled,
        "halo_rejected": halo_rejection is not None,
        "halo_rejection": halo_rejection,
    }


def process_tile(args):
    """Detect and measure sources of one tile; returns core detections with measurements.

    All detections above threshold in the padded patch act as neighbours in the PSF and scene fits.
    Measured positions are converted to Galactic coordinates (l_<est>, b_<est>).
    """
    st = tile_state(*args)
    return st["det"][st["core"]].reset_index(drop=True)


def start_positions(det):
    """Scene starting positions: native PSF fit, else native centroid, else detection.

    Saturated sources start at (and keep, see Scene.refine) their saturated-core centroid.
    """
    x = det.get("x_psfn", pd.Series(np.nan, index=det.index)).fillna(det.get("x_cenn", det.x)).fillna(det.x)
    y = det.get("y_psfn", pd.Series(np.nan, index=det.index)).fillna(det.get("y_cenn", det.y)).fillna(det.y)
    sat = det.sat_area.values > 0
    return np.where(sat, det.x, x), np.where(sat, det.y, y)


def build_scene(patch, det, nx, ny, nrgb, stretch, psf, censored=True):
    """Scene object for a patch (noise from the per-detection annulus estimates)."""
    noise = np.array([np.nanmedian(det[f"noise_{c}"]) for c in "RGB"])
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, 1.0)
    return Scene(
        patch.npix,
        patch.scale,
        nx,
        ny,
        linearize(nrgb, stretch),
        nrgb >= SAT,
        noise,
        psf_cov(patch.jac, psf),
        censored=censored,
    )


def dedup(det, radius_arcmin=DEDUP_R):
    """Drop the lower-S/N member of detection pairs from different tiles closer than radius."""
    rank = np.where(det.sat_area.values > 0, np.inf, det.snr.values)
    tree = cKDTree(radec_unit(det.l.values, det.b.values))
    pairs = tree.query_pairs(2 * np.sin(np.radians(radius_arcmin / 60) / 2), output_type="ndarray")
    tile = det.tile.values
    pairs = pairs[tile[pairs[:, 0]] != tile[pairs[:, 1]]]
    drop = np.where(rank[pairs[:, 0]] >= rank[pairs[:, 1]], pairs[:, 1], pairs[:, 0])
    return det.drop(index=det.index[np.unique(drop)]).reset_index(drop=True)


def process_tiles(tiles, workers=None):
    """Run the full per-tile pipeline in parallel over a tile table (cross-tile duplicates removed)."""
    det_params, calib = load_params()
    rows = [(t._asdict(), det_params, calib) for t in tiles.itertuples(index=False)]
    with ProcessPoolExecutor(workers or WORKERS) as ex:
        parts = [p for p in ex.map(process_tile, rows, chunksize=8) if len(p)]
    return dedup(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
