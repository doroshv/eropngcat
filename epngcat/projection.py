"""All-sky (Hammer-Aitoff / Mollweide) projections between Galactic (l, b) and image pixels.

Normalised plane: x in [-2*sqrt2, 2*sqrt2], y in [-sqrt2, sqrt2] for both projections.
Pixel mapping (image rows increase downwards, l increases to the left)::

    [u, v] = [-ax * x, -ay * y]  rotated by `rot`,  px = x0 + u', py = y0 + v'
"""

from dataclasses import asdict, dataclass

import numpy as np

SQ2 = np.sqrt(2.0)


def _wrap(lam):
    return (lam + np.pi) % (2 * np.pi) - np.pi


def _mollweide_theta(phi, tol=1e-12, maxiter=50):
    """Solve 2t + sin 2t = pi sin(phi).

    Plain Newton stalls at the poles (the derivative 2 + 2cos 2t -> 0), so we start from the
    asymptotic pole solution d = (3 pi (1 - |sin phi|) / 4)^(1/3), t = sign(phi)(pi/2 - d), which is
    accurate there, and clip every Newton step to the valid range.
    """
    phi = np.asarray(phi, dtype=float)
    s = np.sin(phi)
    rhs = np.pi * s
    d = np.cbrt(0.75 * np.pi * (1.0 - np.abs(s)))
    t = np.where(np.abs(phi) > 1.0, np.sign(phi) * (np.pi / 2 - d), phi)
    for _ in range(maxiter):
        f = 2 * t + np.sin(2 * t) - rhs
        fp = 2 + 2 * np.cos(2 * t)
        step = np.where(fp > 1e-15, f / np.maximum(fp, 1e-15), 0.0)
        t = np.clip(t - step, -np.pi / 2, np.pi / 2)
        if np.all(np.abs(step) < tol):
            break
    return t


def forward_norm(lam, phi, kind):
    """(lambda, phi) [rad, lambda relative to centre] -> normalised (x, y)."""
    lam = _wrap(lam)
    if kind == "hammer":
        den = np.sqrt(1 + np.cos(phi) * np.cos(lam / 2))
        return 2 * SQ2 * np.cos(phi) * np.sin(lam / 2) / den, SQ2 * np.sin(phi) / den
    if kind == "mollweide":
        t = _mollweide_theta(phi)
        return 2 * SQ2 / np.pi * lam * np.cos(t), SQ2 * np.sin(t)
    raise ValueError(kind)


def inverse_norm(x, y, kind):
    """Normalised (x, y) -> (lambda, phi) [rad]; NaN outside the ellipse."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = (x / (2 * SQ2)) ** 2 + (y / SQ2) ** 2
    inside = r2 <= 1.0
    if kind == "hammer":
        z = np.sqrt(np.clip(1 - (x / 4) ** 2 - (y / 2) ** 2, 0, None))
        lam = 2 * np.arctan2(z * x, 2 * (2 * z**2 - 1))
        phi = np.arcsin(np.clip(z * y, -1, 1))
    elif kind == "mollweide":
        t = np.arcsin(np.clip(y / SQ2, -1, 1))
        phi = np.arcsin(np.clip((2 * t + np.sin(2 * t)) / np.pi, -1, 1))
        ct = np.cos(t)
        # At the pole every lambda maps to x = 0: return lambda = 0 there instead of 0/0.
        lam = np.where(ct > 1e-12, np.pi * x / (2 * SQ2 * np.where(ct > 1e-12, ct, 1)), 0.0)
        inside &= np.abs(lam) <= np.pi + 1e-12
    else:
        raise ValueError(kind)
    return np.where(inside, lam, np.nan), np.where(inside, phi, np.nan)


@dataclass
class AllSkyProjection:
    """Mapping between Galactic coordinates and pixels of the all-sky image."""

    kind: str = "hammer"
    x0: float = 0.0
    y0: float = 0.0
    ax: float = 1.0
    ay: float = 1.0
    rot: float = 0.0  # radians
    l0: float = 0.0  # deg, Galactic longitude at image centre

    def sky2pix(self, l, b):
        """Galactic (l, b) [deg] -> pixel (px, py) (0-based, pixel centres at integers)."""
        x, y = forward_norm(np.radians(np.asarray(l) - self.l0), np.radians(b), self.kind)
        u, v = -self.ax * x, -self.ay * y
        c, s = np.cos(self.rot), np.sin(self.rot)
        return self.x0 + c * u - s * v, self.y0 + s * u + c * v

    def pix2sky(self, px, py):
        """Pixel (px, py) -> Galactic (l, b) [deg]; NaN outside the sky ellipse."""
        du, dv = np.asarray(px) - self.x0, np.asarray(py) - self.y0
        c, s = np.cos(self.rot), np.sin(self.rot)
        u, v = c * du + s * dv, -s * du + c * dv
        lam, phi = inverse_norm(-u / self.ax, -v / self.ay, self.kind)
        return (np.degrees(lam) + self.l0) % 360.0, np.degrees(phi)

    def to_dict(self):
        """Serialise parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Deserialise parameters."""
        return cls(**d)


@dataclass
class SplitProjection:
    """Image made of two separately projected halves joined at the central meridian.

    Sky side: 0 <= l - l0 < 180 (left half of the image, l0 = `de.l0`) uses `ru`, the rest `de`.
    Pixel side: pixels left of the `de` central meridian use `ru`.
    The eRASS1-DE footprint boundary (l = 179.944 / 359.944) differs from this geometric split by
    0.056 deg (~0.6 px); positions inside that sliver are mapped by the geometric rule.
    """

    de: AllSkyProjection
    ru: AllSkyProjection
    # Per-half, per-channel (R, G, B) pixel shifts (dx, dy) of the channel image relative to the
    # projection (the DE-half blue channel is displaced by ~0.8 px in x).
    shift_de: tuple = ((0.0, 0.0),) * 3
    shift_ru: tuple = ((0.0, 0.0),) * 3

    def _left_pix(self, px, py):
        c, s = np.cos(self.de.rot), np.sin(self.de.rot)
        return c * (np.asarray(px) - self.de.x0) + s * (np.asarray(py) - self.de.y0) < 0

    def left_sky(self, l):
        """Flag Galactic longitudes drawn in the left (Russian) half."""
        return ((np.asarray(l) - self.de.l0) % 360.0) < 180.0

    def sky2pix(self, l, b):
        """Galactic (l, b) [deg] -> pixel (px, py)."""
        left = self.left_sky(l)
        (xd, yd), (xr, yr) = self.de.sky2pix(l, b), self.ru.sky2pix(l, b)
        return np.where(left, xr, xd), np.where(left, yr, yd)

    def sky2pix_channel(self, l, b, k):
        """Galactic (l, b) -> pixel position of the source in channel k (0=R, 1=G, 2=B)."""
        px, py = self.sky2pix(l, b)
        left = self.left_sky(l)
        dx = np.where(left, self.shift_ru[k][0], self.shift_de[k][0])
        dy = np.where(left, self.shift_ru[k][1], self.shift_de[k][1])
        return px + dx, py + dy

    def pix2sky_channel(self, px, py, k):
        """Pixel of channel k -> Galactic (l, b)."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        left = self._left_pix(px, py)
        dx = np.where(left, self.shift_ru[k][0], self.shift_de[k][0])
        dy = np.where(left, self.shift_ru[k][1], self.shift_de[k][1])
        return self.pix2sky(px - dx, py - dy)

    def pix2sky(self, px, py):
        """Pixel (px, py) -> Galactic (l, b) [deg]; NaN outside the half's sky region."""
        left = self._left_pix(px, py)
        (ld, bd), (lr, br) = self.de.pix2sky(px, py), self.ru.pix2sky(px, py)
        l, b = np.where(left, lr, ld), np.where(left, br, bd)
        # A pixel must map back into its own half, otherwise it lies in a gap/overlap.
        bad = left != self.left_sky(l)
        return np.where(bad, np.nan, l), np.where(bad, np.nan, b)

    def to_dict(self):
        """Serialise parameters."""
        return {
            "de": self.de.to_dict(),
            "ru": self.ru.to_dict(),
            "shift_de": [list(v) for v in self.shift_de],
            "shift_ru": [list(v) for v in self.shift_ru],
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialise parameters."""
        zero = ((0.0, 0.0),) * 3
        return cls(
            AllSkyProjection.from_dict(d["de"]),
            AllSkyProjection.from_dict(d["ru"]),
            tuple(map(tuple, d.get("shift_de", zero))),
            tuple(map(tuple, d.get("shift_ru", zero))),
        )


def load_alignment():
    """Return the fitted SplitProjection from step 03."""
    from epngcat.config import OUT, load_json

    return SplitProjection.from_dict(load_json(OUT / "03_align" / "alignment.json")["projection"])
