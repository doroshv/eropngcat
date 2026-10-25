"""Step 02: coarse alignment from the outline of the sky ellipse.

The ellipse gives centre, semi-axes and rotation; both Hammer-Aitoff and Mollweide share the same
outline, so the projection type is decided in step 03 using sources.

Outputs (out/02_align_coarse/): alignment_coarse.json, outline.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, optimize

from epngcat.config import save_json, step_dir
from epngcat.image import load_rgb, sky_mask
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import SQ2, AllSkyProjection

OUTDIR = step_dir("02_align_coarse")


def fit_ellipse(ex, ey):
    """Least-squares ellipse (centre, semi-axes, rotation) through edge points."""

    def resid(p):
        x0, y0, a, b, t = p
        c, s = np.cos(t), np.sin(t)
        u, v = c * (ex - x0) + s * (ey - y0), -s * (ex - x0) + c * (ey - y0)
        # Approximately geometric distance for near-circular normalised radius.
        return (np.hypot(u / a, v / b) - 1) * np.sqrt(a * b)

    p0 = [ex.mean(), ey.mean(), np.ptp(ex) / 2, np.ptp(ey) / 2, 0.0]
    res = optimize.least_squares(resid, p0, loss="soft_l1")
    return res.x, resid(res.x)


def main():
    """Run step 02."""
    mask = sky_mask()
    edge = mask & ~ndimage.binary_erosion(mask)
    ey, ex = np.nonzero(edge)
    # Edge pixels lie half a pixel inside the true boundary.
    (x0, y0, a, b, rot), r = fit_ellipse(ex.astype(float), ey.astype(float))
    a, b = a + 0.5, b + 0.5
    proj = {
        kind: AllSkyProjection(kind=kind, x0=x0, y0=y0, ax=a / (2 * SQ2), ay=b / SQ2, rot=rot, l0=0.0).to_dict()
        for kind in ("hammer", "mollweide")
    }
    info = {"x0": x0, "y0": y0, "a": a, "b": b, "rot_deg": np.degrees(rot), "axis_ratio": a / b}
    info["edge_rms_px"] = float(np.sqrt(np.mean(r**2)))
    info["n_edge"] = int(len(ex))
    save_json({"ellipse": info, "projections": proj}, OUTDIR / "alignment_coarse.json")
    print(info)

    img = load_rgb()
    t = np.linspace(0, 2 * np.pi, 2000)
    fig, axs = plt.subplots(1, 3, figsize=figure_size(1, 3))
    axs[0].imshow(img.astype(np.uint8))
    axs[0].plot(
        x0 + a * np.cos(t) * np.cos(rot) - b * np.sin(t) * np.sin(rot),
        y0 + a * np.cos(t) * np.sin(rot) + b * np.sin(t) * np.cos(rot),
        "c-",
        lw=0.5,
    )
    for ax, (sx, sy) in zip(axs[1:], [(slice(800, 1120), slice(0, 200)), (slice(0, 150), slice(1780, 1990))]):
        ax.imshow(img[sx, sy].astype(np.uint8), extent=(sy.start - 0.5, sy.stop - 0.5, sx.stop - 0.5, sx.start - 0.5))
        ax.plot(
            x0 + a * np.cos(t) * np.cos(rot) - b * np.sin(t) * np.sin(rot),
            y0 + a * np.cos(t) * np.sin(rot) + b * np.sin(t) * np.cos(rot),
            "c-",
            lw=1,
        )
        ax.set_xlim(sy.start, sy.stop)
        ax.set_ylim(sx.stop, sx.start)
    axs[1].set_title("left edge (l=180)")
    axs[2].set_title("north pole")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "outline.pdf")


if __name__ == "__main__":
    main()
