"""Empirical per-band sigmoid halos of saturated sources.

The press image contains an *observed* broad component around bright sources: a smooth plateau
and sigmoid-like transition into the sky.  It already includes the image smoothing and therefore
must not be convolved with the point-source PSF again.  This module deliberately keeps the first
fit independent of faint-source detection; after detection it repeats the fit while masking those
sources and allows the common bright-source centroid to move.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from epngcat.detect import SAT, linearize, robust_sigma

MIN_RADIUS = 0.2
# The press image has 2 arcmin pixels and an effective scene PSF FWHM of about 6 arcmin.  A
# narrower sigmoid transition is neither resolved nor physical; allowing it made the optimiser
# manufacture almost hard-edged dark rings (notably tile 5168).
MIN_WIDTH = 2.0
# A solution on the physical resolution boundary is a constrained, resolved model rather than a
# numerical failure.  Profile-improvement and flux checks below still have to accept it.
WIDTH_BOUNDARY_FACTOR = 1.0
CENTROID_STEP = 4.0  # patch pixels
MAX_CENTROID_SHIFT_ARCMIN = 6.0
NEIGHBOUR_MASK_ARCMIN = 12.0


def sigmoid(radius, amplitude, half_radius, width):
    """Positive flat-topped radial component with a smooth outer transition."""
    z = np.clip((np.asarray(radius) - half_radius) / width, -60.0, 60.0)
    return amplitude / (1.0 + np.exp(z))


def soft_clip(value, ceiling, width):
    """Smooth rendering ceiling used by the press-image saturation model."""
    return ceiling - width * np.logaddexp(0.0, (ceiling - np.asarray(value)) / width)


@dataclass
class BrightSourceFit:
    """Centroids and per-channel observed-sigmoid parameters of the bright sources in one patch.

    Arrays are per source, and per channel where two-dimensional, (nsrc, 3).

    Attributes
    ----------
    centers, initial_centers : ndarray, shape (nsrc, 2)
        Fitted and seed (x, y) positions in patch pixels.
    amplitude : ndarray
        Latent (unclipped) sigmoid amplitude, linearised image units.
    half_radius, width : ndarray
        Sigmoid half-amplitude radius and transition width, arcmin.
    baseline : ndarray
        Fitted sky at the source position, used when no background is passed to `component`.
    clip_level, clip_width : ndarray
        Clipping level of the stretch and width of the soft clip that renders the latent profile.
    success : ndarray of bool
        Channels whose fit is accepted; failed channels have zero amplitude after centroid
        refinement and fall back to the ordinary PSF model.
    """

    centers: np.ndarray
    amplitude: np.ndarray
    half_radius: np.ndarray
    width: np.ndarray
    baseline: np.ndarray
    clip_level: np.ndarray
    clip_width: np.ndarray
    success: np.ndarray
    initial_centers: np.ndarray

    def component(self, xs, ys, scale, background=None):
        """Evaluate the summed halo on channel-specific coordinates.

        ``xs`` and ``ys`` may be ordinary 2-D image grids or arrays of shape ``(3, n)`` for the
        registered native pixels used by :class:`epngcat.scene.Scene`.
        """
        xs, ys = np.asarray(xs), np.asarray(ys)
        background = None if background is None else np.asarray(background)
        if xs.ndim >= 2 and xs.shape[0] == 3 and xs.shape == ys.shape and xs.ndim == 2:
            out = np.zeros((xs.shape[1], 3))
            for i, (x0, y0) in enumerate(self.centers):
                for k in range(3):
                    r = np.hypot(xs[k] - x0, ys[k] - y0) * scale * 60.0
                    latent = sigmoid(r, self.amplitude[i, k], self.half_radius[i, k], self.width[i, k])
                    b = self.baseline[i, k] if background is None else background[:, k]
                    out[:, k] += soft_clip(b + latent, self.clip_level[i, k], self.clip_width[i, k]) - soft_clip(
                        b, self.clip_level[i, k], self.clip_width[i, k]
                    )
            return out
        out = np.zeros((*np.broadcast_shapes(xs.shape, ys.shape), 3))
        for i, (x0, y0) in enumerate(self.centers):
            r = np.hypot(xs - x0, ys - y0) * scale * 60.0
            for k in range(3):
                latent = sigmoid(r, self.amplitude[i, k], self.half_radius[i, k], self.width[i, k])
                b = self.baseline[i, k] if background is None else background[..., k]
                out[..., k] += soft_clip(b + latent, self.clip_level[i, k], self.clip_width[i, k]) - soft_clip(
                    b, self.clip_level[i, k], self.clip_width[i, k]
                )
        return out

    def render(self, npix, scale, background=None):
        """Halo cube on a square patch grid."""
        yy, xx = np.mgrid[:npix, :npix]
        return self.component(xx, yy, scale, background=background)


def _background(data, valid, xx, yy, center, outer, excluded):
    """Robust quadratic sky anchored in the outer part of a source-centred field."""
    x0, y0 = center
    radius = np.hypot(xx - x0, yy - y0)
    xn, yn = (xx - x0) / max(data.shape[1], 1), (yy - y0) / max(data.shape[0], 1)
    basis = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1)
    use = valid & (radius > 0.55 * outer) & (radius < outer)
    for x, y in excluded:
        use &= np.hypot(xx - x, yy - y) > 6.0
    if use.sum() < 30:
        use = valid & (radius > 0.4 * outer) & (radius < outer)
    result = np.empty_like(data, dtype=float)
    for k in range(3):
        x, y = basis[use], data[..., k][use]
        first, *_ = np.linalg.lstsq(x, y, rcond=None)
        noise = max(robust_sigma(y - x @ first), 1e-3)
        fit = optimize.least_squares(lambda q: (x @ q - y) / noise, first, loss="soft_l1", f_scale=2.0, max_nfev=300)
        result[..., k] = basis @ fit.x
    return result


def _crossing_radius(radius, profile, level, default):
    """Radius where a non-increasing profile first crosses ``level``."""
    below = np.flatnonzero(profile <= level)
    if not len(below):
        return float(default)
    j = int(below[0])
    if j == 0:
        return float(radius[0])
    x0, x1, y0, y1 = radius[j - 1], radius[j], profile[j - 1], profile[j]
    if y0 == y1:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _profile_initial(radius, observed, error, core_radius, outer):
    """Observable sigmoid seed from monotone radial-profile crossings.

    Saturation makes the latent central amplitude poorly identifiable, whereas the radii at which
    the *observed* excess falls through 90, 50 and 10 per cent are stable.  Use those crossings
    whenever possible and retain a core-size fallback for weak/noisy channels.
    """
    weights = 1.0 / np.maximum(np.asarray(error), 0.01) ** 2
    profile = optimize.isotonic_regression(observed, weights=weights, increasing=False).x
    profile = np.clip(profile, 0.0, None)
    n_inner = max(2, len(profile) // 4)
    peak = max(float(np.percentile(profile[:n_inner], 75)), 1e-4)
    fallback_half = np.clip(0.8 * core_radius, max(2.0, 1.15 * core_radius), 0.8 * outer)
    r50 = _crossing_radius(radius, profile, 0.5 * peak, fallback_half)
    r90 = _crossing_radius(radius, profile, 0.9 * peak, max(radius[0], r50 - core_radius))
    r10 = _crossing_radius(radius, profile, 0.1 * peak, min(outer, r50 + core_radius))
    width = (r10 - r90) / (2.0 * np.log(9.0))
    if not np.isfinite(width) or width <= 0:
        width = 0.35 * core_radius
    half = np.clip(r50, max(MIN_RADIUS, 1.15 * core_radius), 0.9 * outer)
    width = np.clip(width, MIN_WIDTH, 0.7 * outer)
    return peak, half, width


def _profile_fit_is_credible(observed, error, radius, source):
    """Require a fitted band to improve the profile without inventing excess flux."""
    error = np.maximum(np.asarray(error), 0.01)
    null_rss = float(np.sum((observed / error) ** 2))
    fit_rss = float(np.sum(((source - observed) / error) ** 2))
    improvement = null_rss - fit_rss
    # Annular medians represent surface brightness, hence the radius weighting for integrated flux.
    weight = np.maximum(np.asarray(radius), 1.0)
    observed_flux = float(np.sum(np.clip(observed, 0.0, None) * weight))
    model_flux = float(np.sum(np.clip(source, 0.0, None) * weight))
    flux_ok = model_flux <= 2.0 * observed_flux + 3.0 * float(np.sum(error * weight))
    overshot = np.mean(source - observed > 3.0 * error)
    # Quantised bright cores can have formally tiny annular errors, so a good smooth profile may
    # exceed individual medians by >3 sigma in several bins.  Reject only when a majority of the
    # radial profile is over-predicted; the integrated-flux condition carries the stronger guard.
    return bool(improvement > 9.0 and fit_rss < 0.95 * null_rss and flux_ok and overshot <= 0.5)


def _radial_fit(image, background, valid, radius, outer, core_radius, clip_level, initial=None):
    """Robust annular sigmoid fit after subtraction of an anchored background.

    The excess over `background` is binned in 4' annuli out to `outer` (radii in arcmin). A plain
    sigmoid is fitted from several seeds; its best solution then seeds the fit of the observed
    profile, i.e. the latent sigmoid rendered through the soft clip at `clip_level`.

    Returns
    -------
    params : ndarray
        (amplitude, half_radius, width, clip_width) of the clipped model.
    ok : bool
        Whether the fit is finite, positive, resolved (width above MIN_WIDTH) and passes
        `_profile_fit_is_credible`.
    """
    edges = np.arange(0.0, outer + 4.0, 4.0)
    rr, value, bvalue, error = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = valid & (radius >= lo) & (radius < hi)
        if use.sum() < 8:
            continue
        v = image[use]
        med = np.median(v)
        rr.append(0.5 * (lo + hi))
        value.append(med)
        bvalue.append(np.median(background[use]))
        error.append(max(1.4826 * np.median(np.abs(v - med)) / np.sqrt(use.sum()), 0.01))
    rr, value, bvalue, error = map(np.asarray, (rr, value, bvalue, error))
    if len(rr) < 6:
        return np.array([0.0, max(2.2 * core_radius, 2.0), max(core_radius, 1.0), clip_level]), False
    observed = value - bvalue
    peak, profile_half, profile_width = _profile_initial(rr, observed, error, core_radius, outer)
    half_min = max(MIN_RADIUS, 1.15 * core_radius)
    half0 = np.clip(profile_half, half_min, 0.8 * outer)
    seeds = [(peak, half0, profile_width)]
    if initial is not None:
        seeds.insert(0, tuple(np.maximum(initial, [1e-6, MIN_RADIUS, MIN_WIDTH])))
    for factor in (0.6, 1.0, 1.6):
        seeds.append((peak, np.clip(factor * half0, 1.0, 0.9 * outer), max(0.15 * factor * half0, 0.8)))
    best = None
    bounds = (
        [-14.0, np.log(half_min), np.log(MIN_WIDTH)],
        [25.0, np.log(min(3.0 * max(core_radius, MIN_RADIUS), 1.2 * outer)), np.log(0.7 * outer)],
    )
    for amplitude, half, width in seeds:

        def residual(q):
            profile = (sigmoid(rr, np.exp(q[0]), np.exp(q[1]), np.exp(q[2])) - observed) / error
            radius_prior = (q[1] - np.log(max(half0, MIN_RADIUS))) / 0.5
            return np.concatenate([profile, [radius_prior]])

        fit = optimize.least_squares(
            residual,
            np.clip(np.log([amplitude, half, width]), np.asarray(bounds[0]) + 1e-8, np.asarray(bounds[1]) - 1e-8),
            bounds=bounds,
            loss="soft_l1",
            f_scale=1.5,
            max_nfev=1000,
        )
        rss = np.sum(((sigmoid(rr, *np.exp(fit.x)) - observed) / error) ** 2)
        if best is None or rss < best[0]:
            best = rss, np.exp(fit.x), fit.success
    additive = best[1]
    best_counterfactual = None
    for factor in (1.0, 3.0, 10.0, 100.0):
        q0 = np.log([additive[0] * factor, additive[1], additive[2], max(0.03 * clip_level, 1e-4)])

        def counterfactual(q):
            latent = sigmoid(rr, np.exp(q[0]), np.exp(q[1]), np.exp(q[2]))
            transfer = np.exp(q[3])
            source = soft_clip(bvalue + latent, clip_level, transfer) - soft_clip(bvalue, clip_level, transfer)
            return (source - observed) / error

        fitted = optimize.least_squares(
            counterfactual,
            q0,
            bounds=(
                [-14.0, np.log(MIN_RADIUS), np.log(MIN_WIDTH), -10.0],
                [30.0, np.log(1.2 * outer), np.log(0.7 * outer), np.log(max(0.25 * clip_level, 1e-3))],
            ),
            loss="soft_l1",
            f_scale=1.5,
            max_nfev=1500,
        )
        rss = np.sum(counterfactual(fitted.x) ** 2)
        if best_counterfactual is None or rss < best_counterfactual[0]:
            best_counterfactual = rss, np.exp(fitted.x), fitted.success
    params = best_counterfactual[1]
    latent = sigmoid(rr, params[0], params[1], params[2])
    source = soft_clip(bvalue + latent, clip_level, params[3]) - soft_clip(bvalue, clip_level, params[3])
    credible = _profile_fit_is_credible(observed, error, rr, source)
    # ``least_squares.success`` is false when it merely exhausts ``max_nfev``; for clipped bright
    # profiles this is common even when the rendered solution is excellent.  The explicit profile
    # checks above are the acceptance criterion.
    resolved_width = params[2] > WIDTH_BOUNDARY_FACTOR * MIN_WIDTH
    return params, bool(np.isfinite(params).all() and params[0] > 0 and resolved_width and credible)


def fit_bright_sources(patch, stretch, centers, sat_area, neighbours=(), initial=None, refine_centroid=False):
    """Fit saturated-source halos, first alone and later with detected neighbours masked.

    Parameters
    ----------
    patch : Patch
        Patch holding the press-image RGB values, valid mask, grid size and pixel scale.
    stretch : dict
        Asinh stretch (`A`, `v0`) inverted by `linearize`.
    centers : array_like, shape (nsrc, 2)
        Seed (x, y) positions in patch pixels.
    sat_area : array_like
        Saturated area of each source in patch pixels; sets the core radius prior.
    neighbours : array_like, shape (m, 2)
        Detected sources masked within NEIGHBOUR_MASK_ARCMIN and excluded from the sky fit.
    initial : BrightSourceFit, optional
        Previous fit used as seed; its `initial_centers` remain the reference for the shift limit.
    refine_centroid : bool
        Also fit the common position of all channels, by at most CENTROID_STEP pixels per step;
        a source that ends more than MAX_CENTROID_SHIFT_ARCMIN from its seed is rejected.

    Returns
    -------
    BrightSourceFit
    """
    centers = np.asarray(centers, float).reshape(-1, 2).copy()
    area = np.asarray(sat_area, float)
    data = linearize(patch.rgb.astype(float), stretch)
    sat = patch.rgb >= SAT
    yy, xx = np.mgrid[: patch.npix, : patch.npix]
    amp = np.zeros((len(centers), 3))
    half = np.ones((len(centers), 3))
    width = np.ones((len(centers), 3))
    baseline = np.zeros((len(centers), 3))
    clip_levels = np.zeros((len(centers), 3))
    clip_width = np.ones((len(centers), 3))
    success = np.zeros((len(centers), 3), bool)
    initial_centers = centers.copy() if initial is None else initial.initial_centers.copy()
    neighbours = np.asarray(neighbours, float).reshape(-1, 2)

    for i, center in enumerate(centers):
        edge = min(center[0], center[1], patch.npix - 1 - center[0], patch.npix - 1 - center[1])
        arcmin_per_pixel = patch.scale * 60.0
        field_radius = patch.c * arcmin_per_pixel
        outer_arcmin = min((edge - 1.0) * arcmin_per_pixel, max(field_radius - 60.0, 60.0))
        outer_pix = max(outer_arcmin / arcmin_per_pixel, 8.0)
        excluded = np.concatenate([centers[np.arange(len(centers)) != i], neighbours], axis=0)
        background = _background(data, patch.valid, xx, yy, center, outer_pix, excluded)
        radius_pix = np.hypot(xx - center[0], yy - center[1])
        radius = radius_pix * patch.scale * 60.0
        core_radius = np.sqrt(max(area[i], 1.0) / np.pi) * patch.scale * 60.0
        neighbour_ok = np.ones_like(patch.valid)
        for x, y in excluded:
            neighbour_ok &= np.hypot(xx - x, yy - y) * patch.scale * 60.0 > NEIGHBOUR_MASK_ARCMIN
        for k in range(3):
            # This is an observed-domain sigmoid: the clipped plateau is useful information about
            # its amplitude and transition and is included in the robust radial initialization.
            valid = patch.valid & neighbour_ok & (radius < outer_arcmin)
            # The clipped radius is energy dependent. Count only saturated pixels near this
            # component so another bright source elsewhere in a large fitting patch cannot set
            # the sigmoid prior.
            local_sat = sat[..., k] & (radius < max(2.0 * core_radius, 10.0))
            channel_core = (
                np.sqrt(local_sat.sum() / np.pi) * patch.scale * 60.0 if local_sat.sum() >= 3 else core_radius
            )
            seed = (
                None if initial is None else [initial.amplitude[i, k], initial.half_radius[i, k], initial.width[i, k]]
            )
            clip_level = float(linearize(np.array([[[SAT, SAT, SAT]]], float), stretch)[0, 0, k])
            clip_levels[i, k] = clip_level
            q, ok = _radial_fit(
                data[..., k],
                background[..., k],
                valid,
                radius,
                outer_arcmin,
                channel_core,
                clip_level,
                seed,
            )
            amp[i, k], half[i, k], width[i, k], clip_width[i, k] = q
            baseline[i, k] = background[int(round(center[1])), int(round(center[0])), k]
            success[i, k] = ok

        if refine_centroid and success[i].any():
            sample = patch.valid & neighbour_ok & (radius_pix < outer_pix) & (((xx + yy) % 2) == 0)
            xref, yref = center
            q0 = np.concatenate(
                [
                    center,
                    np.log(np.clip(amp[i], 1e-6, None)),
                    np.log(np.clip(half[i], MIN_RADIUS, None)),
                    np.log(np.clip(width[i], MIN_WIDTH, None)),
                ]
            )
            lo = np.concatenate(
                [
                    [xref - CENTROID_STEP, yref - CENTROID_STEP],
                    np.full(3, -14.0),
                    np.log(np.maximum(0.6 * half[i], MIN_RADIUS)),
                    np.log(np.maximum(0.4 * width[i], MIN_WIDTH)),
                ]
            )
            hi = np.concatenate(
                [
                    [xref + CENTROID_STEP, yref + CENTROID_STEP],
                    np.full(3, 25.0),
                    np.log(np.minimum(1.8 * half[i], 1.2 * outer_arcmin)),
                    np.log(np.minimum(2.5 * width[i], 0.7 * outer_arcmin)),
                ]
            )

            def residual(q):
                r = np.hypot(xx - q[0], yy - q[1]) * patch.scale * 60.0
                model = background.copy()
                for k in range(3):
                    latent = sigmoid(r, np.exp(q[2 + k]), np.exp(q[5 + k]), np.exp(q[8 + k]))
                    model[..., k] = soft_clip(background[..., k] + latent, clip_levels[i, k], clip_width[i, k])
                pieces = []
                for k in range(3):
                    use = sample & ~sat[..., k]
                    noise = max(robust_sigma((data[..., k] - model[..., k])[use]), 0.02)
                    pieces.append((model[..., k][use] - data[..., k][use]) / noise)
                pieces.append((q[:2] - [xref, yref]) / 2.0)
                return np.concatenate(pieces)

            fit = optimize.least_squares(
                residual, np.clip(q0, lo + 1e-8, hi - 1e-8), bounds=(lo, hi), loss="soft_l1", f_scale=2.0, max_nfev=500
            )
            centers[i] = fit.x[:2]
            amp[i], half[i], width[i] = np.exp(fit.x[2:5]), np.exp(fit.x[5:8]), np.exp(fit.x[8:11])
            # Refinement may drive a previously acceptable transition back onto the resolution
            # floor.  Treat that channel as unresolved and use its Gaussian fallback.
            success[i] &= width[i] > WIDTH_BOUNDARY_FACTOR * MIN_WIDTH
            amp[i, ~success[i]] = 0.0
            shift = np.hypot(*(centers[i] - initial_centers[i])) * patch.scale * 60.0
            if shift > MAX_CENTROID_SHIFT_ARCMIN:
                # A large displacement normally means that the joint fit attached the halo to a
                # neighbour or diffuse structure.  Reject this source, restore its seed position,
                # and let the ordinary per-band Gaussian fallback handle its core.
                centers[i] = initial_centers[i]
                amp[i] = 0.0
                success[i] = False

    return BrightSourceFit(centers, amp, half, width, baseline, clip_levels, clip_width, success, initial_centers)
