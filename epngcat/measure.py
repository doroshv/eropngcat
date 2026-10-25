"""Per-detection measurements on a tile patch: positions (several estimators) and flux features.

Position estimators
  det      detection position (3x3 centroid of the matched-filtered image)
  cen      centroid of the linearised, background-subtracted, noise-weighted luminance (r < CEN_R)
  psf      PSF fit: common position, per-channel amplitude and background, calibrated PSF
           covariance (sky term + native-pixel Jacobian term), saturated pixels masked,
           neighbouring detections fitted simultaneously with fixed positions
  psfw     as psf, with a free PSF width scale (absorbs mildly extended emission)
  cenn, psfn, psfnw
           the same estimators applied to the native image pixels (centres mapped into the
           patch TAN plane) instead of the interpolated patch: no interpolation artefacts,
           and the Hammer shear is represented exactly by the pixel-centre geometry

Flux features (used by the flux models of step 09)
  A_<c>, A_<c>_w      PSF amplitudes (fixed / free width fit) in linearised units
  ap6_<c>, ap10_<c>   background-subtracted linearised aperture sums (r < 6', 10')
  sat_area, sat_pix   saturated-blob area from detection / saturated pixels within 10'
  raw/bkg/noise, width scale, local Jacobian (area, anisotropy), neighbour count, S/N
"""

import numpy as np
from scipy import ndimage, optimize

from epngcat.detect import SAT, linearize, psf_cov, sat_mask

CHANNELS = "RGB"
FIT_R = 10.0  # arcmin
BKG_PLANE = False  # fit a tilted (planar) background instead of a constant
CEN_R = 6.0
BKG_ANN = (16.0, 24.0)
NEIGH_R = 14.0  # arcmin: neighbours included in the PSF fit


def _gauss(dx, dy, icov):
    return np.exp(-0.5 * (icov[0, 0] * dx * dx + 2 * icov[0, 1] * dx * dy + icov[1, 1] * dy * dy))


def _linear_fit(x0, y0, others, xx, yy, data, wts, icov):
    """Given source positions, solve linear amplitudes/backgrounds per channel.

    xx, yy: pixel coordinates, either shared (n,) or per channel (3, n).
    Returns residuals and coefficients.
    """
    xx, yy = np.broadcast_to(xx, data.shape), np.broadcast_to(yy, data.shape)
    res, coefs = [], []
    for k in range(3):
        x, y = xx[k], yy[k]
        comps = [_gauss(x - x0, y - y0, icov)] + [_gauss(x - ox, y - oy, icov) for ox, oy in others]
        bkg = [np.ones_like(x)] + ([x - x.mean(), y - y.mean()] if BKG_PLANE else [])
        design = np.stack(comps + bkg, -1)
        w = wts[k]
        a = design * w[:, None]
        coef, *_ = np.linalg.lstsq(a, data[k] * w, rcond=None)
        res.append(a @ coef - data[k] * w)
        coefs.append(coef)
    return np.concatenate(res), coefs


def psf_fit(sub, x0, y0, others, cov, free_width=False):
    """Variable-projection PSF fit of one source, with neighbours at fixed positions.

    The position (and with `free_width` a PSF width scale in 0.5-4) is fitted non-linearly within
    3 px of (x0, y0); amplitudes and backgrounds are solved linearly at each step by `_linear_fit`.

    Parameters
    ----------
    sub : dict
        Pixels to fit: ``xx``, ``yy`` (shared (n,) or per channel (3, n)), ``data`` and ``wts``
        (3, n).
    x0, y0 : float
        Starting position, pixels.
    others : sequence of (x, y)
        Neighbouring sources fitted simultaneously with fixed positions.
    cov : ndarray, shape (2, 2)
        PSF covariance in pixels.
    free_width : bool
        Also fit a width scale of `cov`.

    Returns
    -------
    x, y : float
        Fitted position.
    amps, bkgs : list of float
        Per-channel source amplitude and (constant) background.
    width : float
        Width scale (1 if not fitted).
    chi2 : float
        Mean squared weighted residual.
    """

    def resid(p):
        c = cov * (p[2] ** 2 if free_width else 1.0)
        return _linear_fit(p[0], p[1], others, sub["xx"], sub["yy"], sub["data"], sub["wts"], np.linalg.inv(c))[0]

    p0 = [x0, y0] + ([1.0] if free_width else [])
    lo = [x0 - 3, y0 - 3] + ([0.5] if free_width else [])
    hi = [x0 + 3, y0 + 3] + ([4.0] if free_width else [])
    r = optimize.least_squares(resid, p0, bounds=(lo, hi), x_scale=0.5)
    c = cov * (r.x[2] ** 2 if free_width else 1.0)
    _, coefs = _linear_fit(r.x[0], r.x[1], others, sub["xx"], sub["yy"], sub["data"], sub["wts"], np.linalg.inv(c))
    chi2 = float(np.mean(r.fun**2))
    width = float(r.x[2]) if free_width else 1.0
    nbk = 3 if BKG_PLANE else 1
    return r.x[0], r.x[1], [cf[0] for cf in coefs], [cf[-nbk] for cf in coefs], width, chi2


def native_fits(nat, x0, y0, others, noise, bkg, cov, arcmin):
    """Centroid and PSF fits on native pixels near (x0, y0).

    nat = (x (3, n), y (3, n), lin (n, 3), sat (n,)): per-channel registered pixel positions.
    """
    nx, ny, nlin, nsat = nat
    out = {}
    # Per-channel aperture: channels are registered differently, a common pixel set would truncate
    # one channel's profile asymmetrically.
    rk = np.hypot(nx - x0, ny - y0) * arcmin
    wk = [np.where((rk[k] < CEN_R) & ~nsat, np.clip(nlin[:, k] - bkg[k], 0, None) / noise[k], 0) for k in range(3)]
    wsum = sum(w.sum() for w in wk)
    if wsum > 0:
        out["x_cenn"] = sum((wk[k] * nx[k]).sum() for k in range(3)) / wsum
        out["y_cenn"] = sum((wk[k] * ny[k]).sum() for k in range(3)) / wsum
    r = rk.min(0)
    m = (r < FIT_R) & ~nsat
    if m.sum() < 6:
        return out
    sub = {
        "xx": nx[:, m],
        "yy": ny[:, m],
        "data": nlin[m].T,
        "wts": np.stack([np.full(m.sum(), 1 / noise[k]) for k in range(3)]),
    }
    for name, tag, free in (("psfn", "_n", False), ("psfnw", "_nw", True)):
        try:
            px, py, amps, _, width, chi2 = psf_fit(sub, x0, y0, others, cov, free)
        except (ValueError, np.linalg.LinAlgError):
            continue
        out[f"x_{name}"], out[f"y_{name}"] = px, py
        for k, c in enumerate(CHANNELS):
            out[f"A_{c}{tag}"] = amps[k]
        out[f"chi2{tag}"] = chi2
        if free:
            out["width_n"] = width
    return out


def measure_patch(patch, det, stretch, psf, native=None):
    """Measure all detections (DataFrame with x, y, sat_area, snr) located in a patch.

    native: optional (x, y, rgb) of native image pixels in patch coordinates (see
    patches.native_pixels) for the native-pixel estimators.
    """
    lin = linearize(patch.rgb, stretch)
    sat = sat_mask(patch)
    usable = patch.valid & ~sat
    npix = patch.npix
    yy, xx = np.mgrid[:npix, :npix].astype(float)
    arcmin = patch.scale * 60
    cov = psf_cov(patch.jac, psf)
    ji = np.linalg.inv(patch.jac)
    jac_area = float(abs(np.linalg.det(ji)))
    ev = np.linalg.eigvalsh(ji @ ji.T)
    jac_anis = float(1 - ev[0] / ev[1])
    xs, ys = det.x.values, det.y.values
    if native is not None:
        nlin = linearize(native[2], stretch)
        nat = (native[0], native[1], nlin, (native[2] >= SAT).any(-1))
    rows = []
    for i, (x0, y0) in enumerate(zip(xs, ys)):
        r = np.hypot(xx - x0, yy - y0) * arcmin
        ann = (r > BKG_ANN[0]) & (r < BKG_ANN[1]) & patch.valid
        bkg = np.array([np.median(lin[..., k][ann]) if ann.any() else np.nan for k in range(3)])
        noise = np.array(
            [1.4826 * np.median(np.abs(lin[..., k][ann] - bkg[k])) if ann.any() else np.nan for k in range(3)]
        )
        # A blob large enough to swallow the annulus (Sco X-1, Virgo) leaves it clipped flat, so every
        # channel has MAD 0; falling back to the maximum would keep the zero and then divide by it.
        good = noise[np.isfinite(noise) & (noise > 0)]
        noise = np.where(noise > 0, noise, good.max() if good.size else 1.0)
        out = {
            "jac_area": jac_area,
            "jac_anis": jac_anis,
            "sat_area": det.sat_area.values[i],
            "snr": det.snr.values[i],
            "j11": patch.jac[0, 0],
            "j12": patch.jac[0, 1],
            "j21": patch.jac[1, 0],
            "j22": patch.jac[1, 1],
        }
        for k, c in enumerate(CHANNELS):
            out[f"bkg_{c}"], out[f"noise_{c}"] = bkg[k], noise[k]
            out[f"raw_{c}"] = float(ndimage.map_coordinates(patch.rgb[..., k], [[y0], [x0]], order=1)[0])
            out[f"rawbkg_{c}"] = float(np.median(patch.rgb[..., k][ann])) if ann.any() else np.nan
            for rad in (6, 10):
                m = (r < rad) & patch.valid
                out[f"ap{rad}_{c}"] = float((lin[..., k][m] - bkg[k]).sum())
        out["sat_pix"] = int((sat & (r < FIT_R)).sum())
        d = np.hypot(xs - x0, ys - y0) * arcmin
        nb = (d > 0) & (d < NEIGH_R)
        out["n_neigh"] = int(nb.sum())
        out["neigh_dist"] = float(d[d > 0].min()) if (d > 0).any() else np.inf

        # Centroid of noise-weighted linearised luminance.
        m = (r < CEN_R) & usable
        wl = sum(np.clip(lin[..., k] - bkg[k], 0, None) / noise[k] for k in range(3))[m]
        if wl.sum() > 0:
            out["x_cen"], out["y_cen"] = (wl * xx[m]).sum() / wl.sum(), (wl * yy[m]).sum() / wl.sum()
        else:
            out["x_cen"], out["y_cen"] = x0, y0

        others = list(zip(xs[nb], ys[nb]))
        if native is not None:
            out.update(native_fits(nat, x0, y0, others, noise, bkg, cov, arcmin))

        # PSF fits.
        m = (r < FIT_R) & usable
        if m.sum() < 12:
            rows.append(out)
            continue
        sub = {
            "xx": xx[m],
            "yy": yy[m],
            "data": np.stack([lin[..., k][m] for k in range(3)]),
            "wts": np.stack([np.full(m.sum(), 1 / noise[k]) for k in range(3)]),
        }
        for name, tag, free in (("psf", "", False), ("psfw", "_w", True)):
            try:
                px, py, amps, bks, width, chi2 = psf_fit(sub, x0, y0, others, cov, free)
            except (ValueError, np.linalg.LinAlgError):
                continue
            out[f"x_{name}"], out[f"y_{name}"] = px, py
            for k, c in enumerate(CHANNELS):
                out[f"A_{c}{tag}"] = amps[k]
                out[f"B_{c}{tag}"] = bks[k]
            out[f"chi2{tag}"] = chi2
            if free:
                out["width"] = width
        rows.append(out)
    return rows
