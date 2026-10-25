"""Sky tiling (~2x2 deg tiles) and tangent-plane reprojection of the image onto patches.

Tiles: rows of 2 deg in b; each row is split into an even number of equal l-ranges starting at
L_SPLIT, so tile edges follow the DE/RU boundary and no tile is wider than 2 deg. Polar rows
become wedges that meet at the pole; every patch is a gnomonic projection centred on its tile,
so nothing degenerates there.

Patch pixel arrays are indexed [iy, ix] with xi = (ix - c) * scale, eta = (iy - c) * scale,
xi pointing to increasing l and eta to increasing b.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from epngcat.config import L_SPLIT
from epngcat.image import defect_mask, load_rgb, sky_mask

# The asinh stretch is so compressed near the ceiling that a pixel a few counts below 255 is
# already clipped in practice: 250 masks the near-saturated cores that 254.5 let through, which
# otherwise enter the fits at face value and bias bright amplitudes ~30% low.
SAT = 250.0  # raw value treated as saturated
ROW_H = 2.0
TILE_W = 2.0
NROW = int(180 / ROW_H)


def row_ntiles(row):
    """Return the number of tiles in a row (even, width <= 2 deg at the row edge nearest the equator)."""
    b_lo, b_hi = -90 + row * ROW_H, -90 + (row + 1) * ROW_H
    bmin = 0.0 if b_lo < 0 < b_hi else min(abs(b_lo), abs(b_hi))
    return 2 * int(np.ceil(180 * np.cos(np.radians(bmin)) / TILE_W))


NTILE = np.array([row_ntiles(r) for r in range(NROW)])
ROW_OFFSET = np.concatenate([[0], np.cumsum(NTILE)[:-1]])


def tile_of(l, b):
    """Tile id for Galactic positions."""
    row = np.clip(((np.asarray(b) + 90) // ROW_H).astype(int), 0, NROW - 1)
    n = NTILE[row]
    col = np.minimum((((np.asarray(l) - L_SPLIT) % 360) / 360 * n).astype(int), n - 1)
    return ROW_OFFSET[row] + col


def tile_table():
    """All tiles: id, row, col, l/b bounds and centre."""
    rows = np.repeat(np.arange(NROW), NTILE)
    cols = np.concatenate([np.arange(n) for n in NTILE])
    n = NTILE[rows]
    lo = (L_SPLIT + cols / n * 360) % 360
    width = 360 / n
    b_lo = -90 + rows * ROW_H
    return {
        "tile": np.arange(len(rows)),
        "row": rows,
        "col": cols,
        "l_lo": lo,
        "l_width": width,
        "b_lo": b_lo,
        "b_hi": b_lo + ROW_H,
        "l_c": (lo + width / 2) % 360,
        "b_c": b_lo + ROW_H / 2,
    }


def tan_forward(l, b, lc, bc):
    """Gnomonic projection -> (xi, eta) in degrees."""
    l, b, lc, bc = map(np.radians, (l, b, lc, bc))
    dl = l - lc
    cosc = np.sin(bc) * np.sin(b) + np.cos(bc) * np.cos(b) * np.cos(dl)
    xi = np.cos(b) * np.sin(dl) / cosc
    eta = (np.cos(bc) * np.sin(b) - np.sin(bc) * np.cos(b) * np.cos(dl)) / cosc
    return np.degrees(xi), np.degrees(eta)


def tan_inverse(xi, eta, lc, bc):
    """Inverse gnomonic projection -> (l, b) in degrees."""
    xi, eta, lc, bc = np.radians(xi), np.radians(eta), np.radians(lc), np.radians(bc)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(rho > 0, np.arcsin(np.cos(c) * np.sin(bc) + eta * np.sin(c) * np.cos(bc) / rho), bc)
    l = lc + np.arctan2(xi * np.sin(c), rho * np.cos(bc) * np.cos(c) - eta * np.sin(bc) * np.sin(c))
    return np.degrees(l) % 360, np.degrees(b)


def in_tile(l, b, t):
    """Flag positions (l, b) inside the core of tile row `t` (a mapping from tile_table)."""
    dl = (np.asarray(l) - t["l_lo"]) % 360
    return (dl < t["l_width"]) & (np.asarray(b) >= t["b_lo"]) & (np.asarray(b) < t["b_hi"])


@dataclass
class Patch:
    """Tangent-plane cutout of the all-sky image."""

    lc: float
    bc: float
    scale: float  # deg / patch pixel
    npix: int
    rgb: np.ndarray  # (npix, npix, 3)
    valid: np.ndarray  # (npix, npix) bool, all interpolation input pixels on the sky
    jac: np.ndarray  # (2, 2) d(px, py)/d(xi, eta) [image px / patch px] at the centre
    sat: np.ndarray | None = None  # (npix, npix) bool, nearest native pixel saturated in any channel

    @property
    def c(self):
        """Index of the tangent point."""
        return (self.npix - 1) / 2

    def pix2sky(self, ix, iy):
        """Patch pixel -> Galactic (l, b)."""
        return tan_inverse(
            (np.asarray(ix) - self.c) * self.scale, (np.asarray(iy) - self.c) * self.scale, self.lc, self.bc
        )

    def sky2pix(self, l, b):
        """Galactic (l, b) -> patch pixel."""
        xi, eta = tan_forward(l, b, self.lc, self.bc)
        return xi / self.scale + self.c, eta / self.scale + self.c


def core_halfwidth(t, n=64):
    """Largest |xi|, |eta| (deg) of the tile core boundary in the tile's tangent plane."""
    s = np.linspace(0, 1, n)
    ls = np.concatenate(
        [t["l_lo"] + s * t["l_width"]] * 2 + [np.full(n, t["l_lo"]), np.full(n, t["l_lo"] + t["l_width"])]
    )
    bs = np.concatenate([np.full(n, t["b_lo"]), np.full(n, t["b_hi"]), t["b_lo"] + s * ROW_H, t["b_lo"] + s * ROW_H])
    xi, eta = tan_forward(ls, bs, t["l_c"], t["b_c"])
    return float(max(np.abs(xi).max(), np.abs(eta).max()))


def local_jacobian(proj, lc, bc, scale, h=0.5):
    """d(px, py)/d(ix, iy) at the tangent point [image px per patch px].

    One-sided differences; per axis the smaller one is kept, so a patch centred next to a join
    between the image halves (or the l = 180 edge) does not difference across the jump.
    """
    cols = []
    for dxi, deta in [(1, 0), (0, 1)]:
        steps = np.array([-h, 0.0, h])
        l, b = tan_inverse(steps * dxi * scale, steps * deta * scale, lc, bc)
        px, py = proj.sky2pix(l, b)
        d = np.stack([np.diff(px), np.diff(py)]) / h  # columns: backward, forward
        cols.append(d[:, np.argmin(np.hypot(*d))])
    return np.stack(cols, -1)


_VALID = None


def _valid_map():
    """Usable image pixels: on the sky, away from its rim, and not frozen by an image defect.

    Defects go here rather than into the saturation mask: a saturated blob is turned into a
    detection by detect_patch, and a stuck column would become a spurious source.
    """
    global _VALID
    if _VALID is None:
        ok = ndimage.binary_erosion(sky_mask(), iterations=1) & ~defect_mask()
        _VALID = ok.astype(np.float32)
    return _VALID


def make_patch(proj, lc, bc, halfwidth, scale, order=1):
    """Reproject the image onto a tangent-plane patch centred at (lc, bc).

    `halfwidth` [deg] is rounded up to whole patch pixels. `order` is the spline order of the
    interpolation (1 = bilinear, no ringing around saturated sources).
    """
    rgb = load_rgb()
    npix = 2 * int(np.ceil(halfwidth / scale)) + 1
    c = (npix - 1) / 2
    iy, ix = np.mgrid[0:npix, 0:npix].astype(float)
    l, b = tan_inverse((ix - c) * scale, (iy - c) * scale, lc, bc)
    out = np.empty((npix, npix, 3), np.float32)
    sat = np.zeros((npix, npix), bool)
    for k in range(3):  # each channel sampled at its own registered position
        cx, cy = proj.sky2pix_channel(l, b, k)
        cc = [cy.ravel(), cx.ravel()]
        out[..., k] = ndimage.map_coordinates(rgb[..., k], cc, order=order, mode="nearest").reshape(npix, npix)
        # Saturation from the nearest native pixel: interpolation would hide single saturated pixels.
        sat |= ndimage.map_coordinates(rgb[..., k], cc, order=0, mode="nearest").reshape(npix, npix) >= SAT
    px, py = proj.sky2pix(l, b)
    coords = np.stack([py.ravel(), px.ravel()])
    valid = ndimage.map_coordinates(_valid_map(), coords, order=1, mode="constant", cval=0).reshape(npix, npix) > 0.999
    jac = local_jacobian(proj, lc, bc, scale)
    return Patch(float(lc), float(bc), scale, npix, out, valid, jac, sat)


def native_pixels(proj, patch):
    """Native image pixels covering a patch, with their centres in patch pixel coordinates.

    Every native pixel hit by a patch sample (and its 8 neighbours) is included, so tiles that
    straddle a join or the l = 180 edge collect pixels from both sides. Channels are registered
    separately, so positions are per channel. Returns (iy, ix, x (3, n), y (3, n), rgb (n, 3));
    pixels are kept if their (R-channel) centre lies on the sky and inside the patch.
    """
    iy, ix = np.mgrid[0 : patch.npix, 0 : patch.npix]
    l, b = patch.pix2sky(ix.ravel(), iy.ravel())
    px, py = proj.sky2pix(l, b)
    rgb = load_rgb()
    ny, nx = rgb.shape[:2]
    cx, cy = np.round(px).astype(int), np.round(py).astype(int)
    off = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    ax = (cx[:, None] + off[None, :, 0]).ravel()
    ay = (cy[:, None] + off[None, :, 1]).ravel()
    keep = (ax >= 0) & (ay >= 0) & (ax < nx) & (ay < ny)
    flat = np.unique(ay[keep] * nx + ax[keep])
    ny_, nx_ = flat // nx, flat % nx
    on_sky = _valid_map()[ny_, nx_] > 0
    ny_, nx_ = ny_[on_sky], nx_[on_sky]
    xs, ys = np.full((3, len(nx_)), np.nan), np.full((3, len(nx_)), np.nan)
    for k in range(3):
        pl, pb = proj.pix2sky_channel(nx_.astype(float), ny_.astype(float), k)
        ok = np.isfinite(pl)
        xs[k, ok], ys[k, ok] = patch.sky2pix(pl[ok], pb[ok])
    lo, hi = -0.5, patch.npix - 0.5
    inside = np.isfinite(xs).all(0) & (xs[0] >= lo) & (ys[0] >= lo) & (xs[0] <= hi) & (ys[0] <= hi)
    ny_, nx_ = ny_[inside], nx_[inside]
    return ny_, nx_, xs[:, inside], ys[:, inside], rgb[ny_, nx_]
