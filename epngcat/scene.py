"""Tile scene model: smooth background (tensor-product cubic B-spline) + PSF sources, per channel.

The fit uses the native image pixels (linearised, per-channel registered positions in the patch
TAN frame); saturated pixels are masked. Background knots are KNOT_ARCMIN apart, much wider than
the PSF (sigma ~4.6'), so large-scale diffuse emission is followed without absorbing sources. The
background coefficients carry a second-difference (P-spline) penalty, so knots with little or no
data are interpolated smoothly from their neighbours instead of ringing.

Scene.solve   linear solve for background coefficients and source amplitudes (fixed positions)
Scene.refine  alternates: linear solve -> per-source position fit on the image with background
              and all other sources subtracted -> final linear solve at the refined positions
Scene.render  evaluates background / source model on the patch grid
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import BSpline

KNOT_ARCMIN = 24.0
RIDGE = 1e-3  # Tikhonov weight on background coefficients (anchors the plane the curvature penalty leaves free)
SMOOTH = 0.1  # P-spline weight on the coefficient curvature, in units of the per-channel noise
POS_R = 10.0  # arcmin, pixels used for the per-source position refinement
MAX_STEP = 3.0  # patch px, bound of the position refinement
SAT_PASSES = 4  # active-set passes for the model truncated at the clipping level


def _knots(npix, spacing):
    """Uniform cubic B-spline knot vector covering pixels 0..npix-1 with the given knot spacing."""
    n = max(int(np.ceil(npix / spacing)), 1)
    inner = np.linspace(-0.5, npix - 0.5, n + 1)
    step = inner[1] - inner[0]
    return np.concatenate([inner[0] - step * np.arange(3, 0, -1), inner, inner[-1] + step * np.arange(1, 4)])


def curvature_rows(n1):
    """Second differences of an (n1, n1) coefficient grid along each axis, as rows over the flat grid."""
    d, i = np.diff(np.eye(n1), 2, axis=0), np.eye(n1)
    return np.concatenate([np.kron(d, i), np.kron(i, d)])


def bkg_basis(x, y, npix, spacing):
    """Dense tensor-product cubic B-spline design matrix for points (x, y)."""
    t = _knots(npix, spacing)
    lo, hi = t[3], t[-4]
    bx = BSpline.design_matrix(np.clip(x, lo, hi - 1e-9), t, 3).toarray()
    by = BSpline.design_matrix(np.clip(y, lo, hi - 1e-9), t, 3).toarray()
    return (bx[:, :, None] * by[:, None, :]).reshape(len(x), -1)


def quad(x, y, x0, y0, icov):
    """PSF quadratic form (r/sigma)^2 at points (x, y) for a source at (x0, y0)."""
    dx, dy = x - x0, y - y0
    return icov[0, 0] * dx * dx + 2 * icov[0, 1] * dx * dy + icov[1, 1] * dy * dy


def gauss(x, y, x0, y0, icov):
    """Unit-peak elliptical Gaussian."""
    return np.exp(-0.5 * quad(x, y, x0, y0, icov))


class Scene:
    """Background + PSF model of one patch, fitted to its native image pixels.

    Parameters
    ----------
    npix : int
        Side of the square patch grid on which the background knots are laid out and models are
        rendered.
    scale : float
        Patch pixel size in degrees.
    nx, ny : ndarray, shape (3, n)
        Positions of the n native pixels in the patch frame, per channel (each channel has its own
        registration).
    lin : ndarray, shape (n, 3)
        Linearised pixel values.
    sat : ndarray of bool, shape (n,) or (n, 3)
        Clipped pixels, for all channels at once or per channel.
    noise : sequence of float
        Per-channel pixel noise; it weights the fit and scales the curvature penalty.
    cov : ndarray, shape (2, 2)
        PSF covariance in patch pixels.
    censored : bool
        Treat clipped pixels as lower bounds (see `solve`) rather than dropping them.
    """

    def __init__(self, npix, scale, nx, ny, lin, sat, noise, cov, censored=True):
        self.npix, self.scale = npix, scale
        self.nx, self.ny, self.lin = nx, ny, lin  # (3, n), (3, n), (n, 3)
        sat = np.asarray(sat, bool)
        if sat.ndim == 1:
            sat = np.broadcast_to(sat[:, None], self.lin.shape)
        if sat.shape != self.lin.shape:
            raise ValueError(f"sat has shape {sat.shape}, expected {self.lin.shape}")
        self.use = ~sat
        self.censored = bool(censored)
        self.noise = noise
        self.icov = np.linalg.inv(cov)
        self.spacing = KNOT_ARCMIN / (scale * 60)
        self.B = [bkg_basis(nx[k], ny[k], npix, self.spacing) for k in range(3)]
        self.pen = curvature_rows(len(_knots(npix, self.spacing)) - 4)

    def _design(self, k, xs, ys, m):
        """Design matrix of channel k (sources then background) on the pixels selected by `m`."""
        g = (
            np.stack([gauss(self.nx[k, m], self.ny[k, m], x0, y0, self.icov) for x0, y0 in zip(xs, ys)], -1)
            if len(xs)
            else np.zeros((int(m.sum()), 0))
        )
        return np.concatenate([g, self.B[k][m]], 1)

    def _lstsq(self, k, xs, ys, m):
        """Penalised least squares for channel k on the pixels selected by `m`."""
        nsrc, nb = len(xs), self.B[0].shape[1]
        a = self._design(k, xs, ys, m) / self.noise[k]
        # Penalise coefficient curvature, not amplitude: where the patch has no pixels (off the sky
        # ellipse at the l = 180 cut, or near the poles) the basis is unconstrained by the data and
        # a plain ridge lets it ring to tens of times the sky level, bleeding four knots into the
        # tile.
        p = np.concatenate([np.sqrt(RIDGE) * np.eye(nb), (np.sqrt(SMOOTH) / self.noise[k]) * self.pen])
        reg = np.zeros((len(p), nsrc + nb))
        reg[:, nsrc:] = p
        sol, *_ = np.linalg.lstsq(
            np.concatenate([a, reg]),
            np.concatenate([self.lin[m, k] / self.noise[k], np.zeros(len(p))]),
            rcond=None,
        )
        return sol

    def solve(self, xs, ys):
        """Linear solve for source amplitudes (nsrc, 3) and background coefficients (3, nb).

        A clipped pixel is a lower bound, not a missing measurement: the sky there is *at least* the
        clipping level. Dropping those pixels outright leaves a saturated source's amplitude with no
        constraint but the Gaussian's own tail just outside the core, which at r_core/sigma of 3-5
        is 1e-3 to 1e-7 of the peak - a lever long enough to fit noise. It showed: Cyg X-2 came back
        at 50432 in red against a clipping level of 6.4, and 4U 1636-536 at *minus* 7.2, which
        renders as a bright red core because subtracting it adds flux.

        So the clipped pixels are kept, with the model truncated at the clipping level: each one
        constrains the fit only while the model is still below it, and drops out once satisfied.
        `min(model, clip)` is piecewise linear, so alternating the solve with that active set
        converges in a few passes and each pass is the same linear problem as before.
        """
        nsrc, nb = len(xs), self.B[0].shape[1]
        amps, coefs = np.zeros((nsrc, 3)), np.zeros((3, nb))
        for k in range(3):
            use = self.use[:, k]
            clipped = ~use
            active = np.ones(len(use), bool) if self.censored else use.copy()
            for _ in range(SAT_PASSES):
                sol = self._lstsq(k, xs, ys, active)
                if not self.censored or not clipped.any():
                    break
                model = self._design(k, xs, ys, clipped) @ sol
                # a clipped pixel stays in the fit only while the model is below the level it records
                nxt = use.copy()
                nxt[clipped] = model < self.lin[clipped, k]
                if (nxt == active).all():
                    break
                active = nxt
            amps[:, k], coefs[k] = sol[:nsrc], sol[nsrc:]
        return amps, coefs

    def fit_background(self, mask=None):
        """Fit only the smooth background, optionally on an anchored outer-pixel mask."""
        nb = self.B[0].shape[1]
        coefs = np.zeros((3, nb))
        mask = np.ones(len(self.lin), bool) if mask is None else np.asarray(mask, bool)
        for k in range(3):
            use = self.use[:, k] & mask
            a = self.B[k][use] / self.noise[k]
            p = np.concatenate([np.sqrt(RIDGE) * np.eye(nb), (np.sqrt(SMOOTH) / self.noise[k]) * self.pen])
            coefs[k], *_ = np.linalg.lstsq(
                np.concatenate([a, p]),
                np.concatenate([self.lin[use, k] / self.noise[k], np.zeros(len(p))]),
                rcond=None,
            )
        return coefs

    def solve_sources(self, xs, ys, coefs):
        """Fit PSF amplitudes while holding an outer-anchored background fixed."""
        amps = np.zeros((len(xs), 3))
        if not len(xs):
            return amps
        for k in range(3):
            use = self.use[:, k]
            g = np.stack([gauss(self.nx[k, use], self.ny[k, use], x0, y0, self.icov) for x0, y0 in zip(xs, ys)], -1)
            target = self.lin[use, k] - self.B[k][use] @ coefs[k]
            amps[:, k], *_ = np.linalg.lstsq(g, target, rcond=None)
        return amps

    def fill_missing_source_amplitudes(self, xs, ys, amps, coefs, missing):
        """Fit PSFs only for source/channel pairs not represented by another component.

        A saturated source can have a valid sigmoid fit in only some bands. Such a source is kept
        out of the ordinary joint PSF solve to avoid double-counting the successful sigmoid bands,
        but the failed bands still need a source model. Fit those amplitudes jointly against the
        residual left by the background and all already-modelled sources.
        """
        xs, ys, amps = np.asarray(xs), np.asarray(ys), np.asarray(amps).copy()
        missing = np.asarray(missing, bool)
        for k in range(3):
            rows = np.flatnonzero(missing[:, k])
            if not len(rows):
                continue
            use = self.use[:, k]
            current = self.model(xs, ys, amps, coefs, k, self.nx[k], self.ny[k])
            g = np.stack(
                [gauss(self.nx[k, use], self.ny[k, use], xs[i], ys[i], self.icov) for i in rows],
                -1,
            )
            fitted, *_ = np.linalg.lstsq(g, self.lin[use, k] - current[use], rcond=None)
            amps[rows, k] = np.maximum(fitted, 0.0)
        return amps

    def model(self, xs, ys, amps, coefs, k, x, y, which="all"):
        """Model of channel k at points (x, y).

        `xs`, `ys`, `amps` (nsrc, 3) and `coefs` (3, nb) are as returned by `solve`; `which` is
        "all", "bkg" (background only) or "src" (sources only).
        """
        out = np.zeros(len(x))
        if which in ("all", "bkg"):
            out += bkg_basis(x, y, self.npix, self.spacing) @ coefs[k]
        if which in ("all", "src"):
            for (x0, y0), a in zip(zip(xs, ys), amps[:, k]):
                out += a * gauss(x, y, x0, y0, self.icov)
        return out

    def refine(self, xs, ys, fixed=None, iterations=1, fixed_coefs=None):
        """Alternate linear solves and per-source position fits.

        Each source is moved by at most MAX_STEP patch pixels from its input position, using the
        pixels within POS_R that are unclipped in every channel.

        Parameters
        ----------
        xs, ys : array_like
            Initial source positions in patch pixels.
        fixed : array_like of bool, optional
            Sources whose position is kept (e.g. saturated cores, which are better localised by the
            saturated pixels than by the masked-core PSF fit).
        iterations : int
            Number of solve / position-fit rounds before the final solve.
        fixed_coefs : ndarray, shape (3, nb), optional
            Background coefficients to hold fixed (see `solve_sources`); fitted jointly if None.

        Returns
        -------
        xs, ys : ndarray
            Refined positions.
        amps : ndarray, shape (nsrc, 3)
            Source amplitudes.
        coefs : ndarray, shape (3, nb)
            Background coefficients.
        """
        xs, ys = np.array(xs, float), np.array(ys, float)
        fixed = np.zeros(len(xs), bool) if fixed is None else np.asarray(fixed, bool)
        x_in, y_in = xs.copy(), ys.copy()
        arcmin = self.scale * 60
        for _ in range(iterations):
            if fixed_coefs is None:
                amps, coefs = self.solve(xs, ys)
            else:
                coefs = np.asarray(fixed_coefs)
                amps = self.solve_sources(xs, ys, coefs)
            full = np.stack([self.model(xs, ys, amps, coefs, k, self.nx[k], self.ny[k]) for k in range(3)], -1)
            for i in range(len(xs)):
                own = np.stack(
                    [amps[i, k] * gauss(self.nx[k], self.ny[k], xs[i], ys[i], self.icov) for k in range(3)], -1
                )
                resid = self.lin - full + own  # this source plus noise only
                # A shared position is constrained only by pixels usable in every channel. This is
                # conservative near a channel-dependent clipped core and avoids indexing one
                # channel's data with another channel's saturation mask.
                m = self.use.all(1) & (np.hypot(self.nx[1] - xs[i], self.ny[1] - ys[i]) * arcmin < POS_R)
                if fixed[i] or m.sum() < 6 or amps[i].max() <= 0:
                    continue

                def res(p, m=m, resid=resid):
                    out = []
                    for k in range(3):
                        g = gauss(self.nx[k, m], self.ny[k, m], p[0], p[1], self.icov)
                        a = max((g @ resid[m, k]) / max(g @ g, 1e-12), 0.0)
                        out.append((a * g - resid[m, k]) / self.noise[k])
                    return np.concatenate(out)

                lo = [x_in[i] - MAX_STEP, y_in[i] - MAX_STEP]
                hi = [x_in[i] + MAX_STEP, y_in[i] + MAX_STEP]
                p0 = np.clip([xs[i], ys[i]], np.add(lo, 1e-6), np.subtract(hi, 1e-6))
                xs[i], ys[i] = optimize.least_squares(res, p0, bounds=(lo, hi), x_scale=0.5).x
        if fixed_coefs is None:
            amps, coefs = self.solve(xs, ys)
        else:
            coefs = np.asarray(fixed_coefs)
            amps = self.solve_sources(xs, ys, coefs)
        return xs, ys, amps, coefs

    def render(self, xs, ys, amps, coefs):
        """Background and source model images (npix, npix, 3) on the patch grid (channel-registered frame)."""
        iy, ix = np.mgrid[: self.npix, : self.npix].astype(float)
        x, y = ix.ravel(), iy.ravel()
        bkg = np.stack([self.model(xs, ys, amps, coefs, k, x, y, "bkg") for k in range(3)], -1)
        src = np.stack([self.model(xs, ys, amps, coefs, k, x, y, "src") for k in range(3)], -1)
        shape = (self.npix, self.npix, 3)
        return bkg.reshape(shape), src.reshape(shape)
