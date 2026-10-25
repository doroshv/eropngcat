"""Global DE uncertainty systematics from unique, reliable eRASS1 matches.

The source-level models already provide a positional scale and asymmetric flux prediction
intervals.  This module asks whether the residual variance of clean one-to-one DE associations
requires an additional additive floor, a multiplicative term, both, or neither. Candidate models
are fitted on training spatial blocks, selected by BIC on validation blocks, and evaluated on the
downstream test blocks. The selected training-fitted terms are applied unchanged to the
released catalogue. Global image registration remains a shared upstream calibration.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from epngcat.s09_models import BANDS
from epngcat.sky import radec_unit, sep_arcsec

MATCH_ARCMIN = 4.0
POS_Z_MAX = 2.0
KINDS = ("base", "additive", "multiplicative", "both")


def unique_pairs(image, reference, flux_limit, radius_arcmin=MATCH_ARCMIN):
    """One-to-one image/eRASS1 associations above the image sensitivity limit.

    Candidate edges use a deliberately generous fixed radius.  A pair survives only when its
    image detection and reference source both have graph degree one, removing blends, duplicate
    detections and all other ambiguous associations.  Morphological/reliability cuts are applied
    after graph construction so a flagged neighbouring detection still makes the match ambiguous.
    """
    image = image.reset_index(drop=True)
    reference = reference[(reference.EXT == 0) & (flux_limit <= reference.ML_FLUX_1)].reset_index(drop=True)
    tree = cKDTree(radec_unit(reference.LII.values, reference.BII.values))
    chord = 2 * np.sin(np.radians(radius_arcmin / 60.0) / 2)
    edges = []
    for i, neighbours in enumerate(tree.query_ball_point(radec_unit(image.LII.values, image.BII.values), chord)):
        if not neighbours:
            continue
        j = np.asarray(neighbours, int)
        separation = sep_arcsec(
            np.full(len(j), image.LII.iloc[i]),
            np.full(len(j), image.BII.iloc[i]),
            reference.LII.values[j],
            reference.BII.values[j],
        )
        edges.extend((i, int(jj), float(ss)) for jj, ss in zip(j, separation))
    pairs = pd.DataFrame(edges, columns=["image_index", "reference_index", "separation_arcsec"])
    if not len(pairs):
        return image.iloc[:0].copy(), reference.iloc[:0].copy()
    image_degree = pairs.image_index.value_counts()
    reference_degree = pairs.reference_index.value_counts()
    pairs = pairs[
        pairs.image_index.map(image_degree).eq(1) & pairs.reference_index.map(reference_degree).eq(1)
    ].reset_index(drop=True)
    im = image.iloc[pairs.image_index].reset_index(drop=True)
    ref = reference.iloc[pairs.reference_index].reset_index(drop=True)
    im["MATCH_SEP"] = pairs.separation_arcsec.values
    # Keep the graph construction conservative (all image detections can make an association
    # ambiguous), then restrict the calibration sample itself to sources above the measured
    # image sensitivity limit.
    good = ~im.FLAG_EXTENDED & ~im.FLAG_POS_FALLBACK & (flux_limit <= im.FLUX_1)
    return im[good].reset_index(drop=True), ref[good].reset_index(drop=True)


def _parameters(kind, q, scale=1.0):
    """Map optimiser parameters `q` of a model kind to (additive term, fractional term).

    The additive term is fitted in units of `scale` so both parameters are of order 0.1.
    """
    if kind == "base":
        return 0.0, 0.0
    if kind == "additive":
        return float(q[0] * scale), 0.0
    if kind == "multiplicative":
        return 0.0, float(q[0])
    return float(q[0] * scale), float(q[1])


def _fit(kind, objective, scale=1.0):
    """Minimise `objective((add, fraction))` over the free terms of `kind`; both terms in [0, 5]."""
    if kind == "base":
        return (0.0, 0.0)
    npar = 2 if kind == "both" else 1
    start = np.full(npar, 0.05)
    result = minimize(
        lambda q: objective(_parameters(kind, q, scale)),
        start,
        method="Powell",
        bounds=[(0.0, 5.0)] * npar,
    )
    return _parameters(kind, result.x, scale)


def _position_score(image, reference, rows, parameters):
    """Rayleigh likelihood and coverage of the match offsets for (additive arcsec, fraction).

    The image error becomes sqrt(POS_ERR^2 (1 + fraction^2) + add^2) and the eRASS1 per-axis error
    is added in quadrature.
    """
    add, mult = parameters
    base = image.POS_ERR.values[rows]
    ref = reference.POS_ERR.values[rows] / np.sqrt(2.0)  # eRASS1 POS_ERR is radial
    radius = image.MATCH_SEP.values[rows]
    sigma2 = base**2 + (mult * base) ** 2 + add**2 + ref**2
    nll = np.log(sigma2) + radius**2 / (2 * sigma2)  # Rayleigh; -log(radius) is model independent
    sigma = np.sqrt(sigma2)
    return {
        "n": int(len(rows)),
        "nll": float(nll.sum()),
        "mean_nll": float(nll.mean()),
        "within_1sigma": float(np.mean(radius <= sigma)),
        "within_2sigma": float(np.mean(radius <= 2 * sigma)),
        "within_3sigma": float(np.mean(radius <= 3 * sigma)),
    }


def position_systematics(image, reference):
    """Select training-fitted additive/multiplicative excess terms for POS_ERR."""
    train = np.flatnonzero(image.SPLIT.values == "train")
    validation = np.flatnonzero(image.SPLIT.values == "val")
    test = np.flatnonzero(image.SPLIT.values == "test")
    scale = float(np.nanmedian(image.POS_ERR.values[train]))

    def objective(rows):
        def value(parameters):
            return _position_score(image, reference, rows, parameters)["mean_nll"]

        return value

    candidates = {}
    for kind in KINDS:
        parameters = _fit(kind, objective(train), scale)
        validation_score = _position_score(image, reference, validation, parameters)
        npar = 0 if kind == "base" else 2 if kind == "both" else 1
        validation_score["bic"] = float(2 * validation_score["nll"] + npar * np.log(max(len(validation), 1)))
        candidates[kind] = {
            "parameters_train": {"add_arcsec": parameters[0], "fraction": parameters[1]},
            "validation": validation_score,
        }
    selected = min(candidates, key=lambda key: candidates[key]["validation"]["bic"])
    fitted = candidates[selected]["parameters_train"]
    final = (fitted["add_arcsec"], fitted["fraction"])
    return {
        "selected": selected,
        "add_arcsec": final[0],
        "fraction": final[1],
        "n_matches": int(len(image)),
        "test": _position_score(image, reference, test, final),
        "test_base": _position_score(image, reference, test, (0.0, 0.0)),
        "candidates": candidates,
    }


def _flux_arrays(image, reference, rows, band):
    """Finite (flux, residual, interval half-width, eRASS1 error) arrays of one band.

    The half-width is the side of the asymmetric 16-84% interval facing the eRASS1 flux.
    """
    flux = image[f"FLUX_{band}"].values[rows]
    truth = reference[f"ML_FLUX_{band}"].values[rows]
    ref_error = reference[f"ML_FLUX_ERR_{band}"].values[rows]
    lo = image[f"FLUX_{band}_LO"].values[rows]
    hi = image[f"FLUX_{band}_HI"].values[rows]
    good = (
        np.isfinite(flux)
        & np.isfinite(truth)
        & (truth > 0)
        & np.isfinite(ref_error)
        & (ref_error >= 0)
        & np.isfinite(lo)
        & np.isfinite(hi)
    )
    flux, truth, ref_error, lo, hi = flux[good], truth[good], ref_error[good], lo[good], hi[good]
    residual = flux - truth
    # Use the side of the asymmetric prediction interval pointing toward the reference value.
    base_error = np.where(residual >= 0, flux - lo, hi - flux)
    return flux, residual, np.maximum(base_error, np.finfo(float).tiny), ref_error


def _flux_score(image, reference, rows, band, parameters):
    """Gaussian likelihood and coverage of the flux residuals for (additive flux, fraction)."""
    add, mult = parameters
    flux, residual, base, ref_error = _flux_arrays(image, reference, rows, band)
    interval_variance = base**2 + (mult * flux) ** 2 + add**2
    sigma2 = interval_variance + ref_error**2
    nll = 0.5 * (np.log(sigma2) + residual**2 / sigma2)
    return {
        "n": int(len(residual)),
        "nll": float(nll.sum()),
        "mean_nll": float(nll.mean()),
        "within_1sigma": float(np.mean(np.abs(residual) <= np.sqrt(sigma2))),
        "coverage_16_84": float(np.mean(np.abs(residual) <= np.sqrt(interval_variance))),
    }


def flux_systematics(image, reference, position):
    """Select per-band excess terms using position-consistent unique matches.

    Only pairs within POS_Z_MAX of the calibrated positional error (from `position`, as returned by
    `position_systematics`) are kept, so chance coincidences do not inflate the flux terms. Each
    band then goes through the same train-fit / validation-BIC / test-score sequence as positions.

    Returns
    -------
    result : dict
        Per band: the selected kind, ``add_flux`` and ``fraction``, test scores with and without
        them, and all candidates.
    n : int
        Number of position-consistent pairs.
    """
    pos_sigma = np.sqrt(
        image.POS_ERR.values**2
        + (position["fraction"] * image.POS_ERR.values) ** 2
        + position["add_arcsec"] ** 2
        + reference.POS_ERR.values**2 / 2
    )
    reliable = image.MATCH_SEP.values <= POS_Z_MAX * pos_sigma
    image, reference = image[reliable].reset_index(drop=True), reference[reliable].reset_index(drop=True)
    train = np.flatnonzero(image.SPLIT.values == "train")
    validation = np.flatnonzero(image.SPLIT.values == "val")
    test = np.flatnonzero(image.SPLIT.values == "test")
    result = {}
    for band in BANDS:
        all_flux = image[f"FLUX_{band}"].values[train]
        scale = float(np.nanmedian(all_flux[np.isfinite(all_flux) & (all_flux > 0)]))

        def objective(rows):
            def value(parameters):
                return _flux_score(image, reference, rows, band, parameters)["mean_nll"]

            return value

        candidates = {}
        for kind in KINDS:
            parameters = _fit(kind, objective(train), scale)
            validation_score = _flux_score(image, reference, validation, band, parameters)
            npar = 0 if kind == "base" else 2 if kind == "both" else 1
            validation_score["bic"] = float(2 * validation_score["nll"] + npar * np.log(max(validation_score["n"], 1)))
            candidates[kind] = {
                "parameters_train": {"add_flux": parameters[0], "fraction": parameters[1]},
                "validation": validation_score,
            }
        selected = min(candidates, key=lambda key: candidates[key]["validation"]["bic"])
        fitted = candidates[selected]["parameters_train"]
        final = (fitted["add_flux"], fitted["fraction"])
        result[band] = {
            "selected": selected,
            "add_flux": final[0],
            "fraction": final[1],
            "n_matches": int(_flux_score(image, reference, np.arange(len(image)), band, final)["n"]),
            "test": _flux_score(image, reference, test, band, final),
            "test_base": _flux_score(image, reference, test, band, (0.0, 0.0)),
            "candidates": candidates,
        }
    return result, int(len(image))


def calibrate_uncertainties(image, reference, flux_limit):
    """Calibrate global DE systematic terms and return a serialisable report."""
    if not np.isfinite(flux_limit) or flux_limit <= 0:
        raise ValueError("A finite positive validation completeness limit is required")
    matched_image, matched_reference = unique_pairs(image, reference, flux_limit)
    if not {"train", "val", "test"}.issubset(set(matched_image.SPLIT)):
        raise ValueError("Uncertainty calibration requires associations in all three spatial subsets")
    position = position_systematics(matched_image, matched_reference)
    flux, n_flux_position_matches = flux_systematics(matched_image, matched_reference, position)
    return {
        "flux_limit": float(flux_limit),
        "flux_limit_source": "validation detections with adopted pipeline settings",
        "fit_split": "train",
        "selection_split": "val",
        "release_refit": False,
        "match_radius_arcmin": MATCH_ARCMIN,
        "flux_position_z_max": POS_Z_MAX,
        "n_unique_position_matches": int(len(matched_image)),
        "n_reliable_flux_matches": n_flux_position_matches,
        "position": position,
        "flux": flux,
    }


def apply_uncertainty_systematics(catalogue, calibration):
    """Apply selected global DE systematic terms to all catalogue uncertainty estimates."""
    cat = catalogue.copy()
    pos = calibration["position"]
    cat["POS_ERR"] = np.sqrt(
        cat.POS_ERR.values**2 + (pos["fraction"] * cat.POS_ERR.values) ** 2 + pos["add_arcsec"] ** 2
    )
    for band in BANDS:
        flux = cat[f"FLUX_{band}"].values
        lo_error = np.maximum(flux - cat[f"FLUX_{band}_LO"].values, 0.0)
        hi_error = np.maximum(cat[f"FLUX_{band}_HI"].values - flux, 0.0)
        sys = calibration["flux"][band]
        extra2 = sys["add_flux"] ** 2 + (sys["fraction"] * flux) ** 2
        cat[f"FLUX_{band}_LO"] = np.maximum(flux - np.sqrt(lo_error**2 + extra2), 0.0)
        cat[f"FLUX_{band}_HI"] = flux + np.sqrt(hi_error**2 + extra2)
    return cat


def test_coverage(catalogue, reference):
    """Coverage on the step-10 test association sample, without uncertainty-based cuts."""
    from epngcat.evaluate import match_detections

    test = catalogue[(catalogue.SPLIT == "test") & ~catalogue.IN_SNR_MASK].reset_index(drop=True)
    pairs = match_detections(test.rename(columns={"LII": "l", "BII": "b"}), reference)
    pt = pairs[(pairs.cat_EXT == 0) & pairs.cat_ML_FLUX_1.notna()]
    sigma = np.sqrt(pt.POS_ERR**2 + pt.cat_POS_ERR.fillna(0) ** 2 / 2)
    pos = {"n": len(pt)}
    pos.update({f"frac_within_{k}sigma": float((pt.sep <= k * sigma).mean()) for k in (1, 2, 3)})
    fluxes = {}
    for band in BANDS:
        good = pt[(pt.sep < 150) & (pt[f"cat_ML_FLUX_{band}"] > 0)]
        truth = good[f"cat_ML_FLUX_{band}"]
        fluxes[band] = {
            "n": len(good),
            "coverage_16_84": float(((good[f"FLUX_{band}_LO"] <= truth) & (truth <= good[f"FLUX_{band}_HI"])).mean()),
        }
    return {"positions": pos, "fluxes": fluxes}
