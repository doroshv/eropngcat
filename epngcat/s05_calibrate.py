"""Step 05: calibrate the display stretch and the PSF on training-split sources.

Stretch: the image is a non-linear rendering of three bands. For each channel c we look for an
inverse L(v) such that L(peak) - L(background) is proportional to the catalogue band flux
independently of the background level. Families: linear, power (gamma), exp (log stretch),
sinh (asinh stretch). Scale parameters are fitted on one half of the calibration sources and the
family is chosen on the other half. Channel <-> band: R ~ P1 (0.2-0.5), G ~ P2 (0.5-1),
B ~ P3 (1-2 keV).

PSF: elliptical Gaussians fitted to unsaturated sources in local TAN patches (linearised
luminance). The covariance is modelled as C = t2 * I + s2 * J^-1 J^-T: a sky-space PSF plus a
native-pixel term that stretches sources where the Hammer grid is sheared.

Outputs (out/05_calibrate/): calibration.json, sources.parquet, stretch.pdf, psf.pdf
"""

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from epngcat import s03_align_refine as a3
from epngcat.config import SEED, WORKERS, save_json, step_dir
from epngcat.detect import SAT, SCALE, linearize, robust_sigma
from epngcat.evaluate import catalog
from epngcat.patches import make_patch
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import load_alignment

OUTDIR = step_dir("05_calibrate")
F_RANGE = (2e-13, 1e-10)
ISO_ARCMIN = 12
CHANNELS = "RGB"
BAND_OF = {"R": "ML_FLUX_P1", "G": "ML_FLUX_P2", "B": "ML_FLUX_P3"}
A_GRID = np.geomspace(5, 3000, 80)
V0_GRID = np.arange(-90, 6, 3.0)


def families():
    """Inverse-stretch families L(v; A, v0)."""
    return {
        "linear": lambda v, a, v0: v - v0,
        "pow": lambda v, a, v0: (np.clip(v - v0, 0, None) / 255.0) ** (a / 100.0),
        "exp": lambda v, a, v0: np.exp((v - v0) / a),
        "sinh": lambda v, a, v0: np.sinh((v - v0) / a),
    }


def select_sources():
    """Isolated point sources on training tiles (outside known SNRs, see evaluate.catalog)."""
    cat = catalog("train")
    a3.ISO_RADIUS = ISO_ARCMIN / 60
    e = cat[(cat.EXT == 0) & cat.ML_FLUX_1.between(*F_RANGE)]
    iso = a3.isolated(cat.LII.values, cat.BII.values, cat.ML_FLUX_1.values, e.LII, e.BII, e.ML_FLUX_1)
    return e[iso].reset_index(drop=True)


def measure_raw(args):
    """Raw peak (bilinear at the catalogue position), annulus background and saturation."""
    l, b = args
    pa = make_patch(load_alignment(), l, b, 0.45, SCALE)
    if not pa.valid.all():
        return None
    yy, xx = np.mgrid[: pa.npix, : pa.npix] - pa.c
    r = np.hypot(xx, yy) * SCALE * 60
    ann = (r > 16) & (r < 24)
    return {
        **{f"peak_{c}": pa.rgb[r <= 1.0, k].mean() for k, c in enumerate(CHANNELS)},
        **{f"bkg_{c}": np.median(pa.rgb[ann, k]) for k, c in enumerate(CHANNELS)},
        "sat": bool((pa.rgb[r < 6] >= SAT).any()),
    }


def fit_psf(args):
    """Elliptical Gaussian fit on linearised luminance; returns covariance and pixel-term matrix."""
    l, b, stretch = args
    pa = make_patch(load_alignment(), l, b, 0.5, SCALE)
    if not pa.valid.all() or (pa.rgb[np.hypot(*(np.mgrid[: pa.npix, : pa.npix] - pa.c)) < 4] >= SAT).any():
        return None
    img = linearize(pa.rgb, stretch).sum(-1)
    yy, xx = np.mgrid[: pa.npix, : pa.npix] - pa.c
    r = np.hypot(xx, yy)
    bk = np.median(img[r > 11])
    m = r < 8

    def resid(p):
        a, x0, y0, sxx, syy, rho = p
        dx, dy = xx[m] - x0, yy[m] - y0
        z = (dx**2 / sxx**2 - 2 * rho * dx * dy / (sxx * syy) + dy**2 / syy**2) / (1 - rho**2)
        return a * np.exp(-z / 2) + bk - img[m]

    p0 = [max(img[r < 1].mean() - bk, 1e-6), 0, 0, 2, 2, 0]
    try:
        res = optimize.least_squares(resid, p0, bounds=([0, -3, -3, 0.3, 0.3, -0.95], [np.inf, 3, 3, 15, 15, 0.95]))
    except ValueError:
        return None
    a, x0, y0, sxx, syy, rho = res.x
    ji = np.linalg.inv(pa.jac)
    mm = ji @ ji.T
    return {"cxx": sxx**2, "cyy": syy**2, "cxy": rho * sxx * syy, "mxx": mm[0, 0], "myy": mm[1, 1], "mxy": mm[0, 1]}


def pmap(fn, items):
    """Parallel map preserving order."""
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=32))


def stretch_scatter(fn, a, v0, peak, bkg, logf):
    """Robust scatter of log F - log(L(peak) - L(bkg)); non-positive increments count as outliers."""
    d = fn(peak, a, v0) - fn(bkg, a, v0)
    y = np.where(d > 0, logf - np.log10(np.where(d > 0, d, 1)), np.nan)
    ok = np.isfinite(y)
    if ok.mean() < 0.8:
        return np.inf, np.nan
    med = np.median(y[ok])
    # Treat failed increments as residuals of 1 dex so they are penalised, not ignored.
    res = np.where(ok, y - med, 1.0)
    return 1.4826 * np.median(np.abs(res)), med


def fit_stretch(src, fit_mask):
    """Grid-fit each family per channel on fit_mask; evaluate on the rest."""
    out = {}
    for name, fn in families().items():
        ch_res = {}
        for c in CHANNELS:
            with np.errstate(divide="ignore"):
                logf = np.log10(src[BAND_OF[c]].values)
            peak, bkg = src[f"peak_{c}"].values, src[f"bkg_{c}"].values
            good = np.isfinite(logf) & (peak < SAT)
            tr, ho = good & fit_mask, good & ~fit_mask
            a_grid = [1.0] if name == "linear" else A_GRID
            best = min(
                ((stretch_scatter(fn, a, v0, peak[tr], bkg[tr], logf[tr])[0], a, v0) for a in a_grid for v0 in V0_GRID),
                key=lambda t: t[0],
            )
            s_ho, zp = stretch_scatter(fn, best[1], best[2], peak[ho], bkg[ho], logf[ho])
            # Residual trend with background level (dex per 100 raw units) on the hold-out half.
            d = fn(peak[ho], best[1], best[2]) - fn(bkg[ho], best[1], best[2])
            ok = d > 0
            trend = np.polyfit(bkg[ho][ok], logf[ho][ok] - np.log10(d[ok]), 1)[0] * 100
            ch_res[c] = {
                "A": best[1],
                "v0": best[2],
                "scatter_fit": best[0],
                "scatter_holdout": s_ho,
                "bkg_trend_dex_per_100": trend,
                "zeropoint": zp,
            }
        out[name] = ch_res
    return out


def main():
    """Run step 05."""
    src = select_sources()
    raw = pmap(measure_raw, list(zip(src.LII, src.BII)))
    keep = np.array([r is not None for r in raw])
    src = pd.concat([src[keep].reset_index(drop=True), pd.DataFrame([r for r in raw if r])], axis=1)
    rng = np.random.default_rng(SEED)
    src["fit_half"] = rng.random(len(src)) < 0.5
    print("calibration sources:", len(src), "saturated:", int(src.sat.sum()))

    fams = fit_stretch(src, src.fit_half.values)
    for name, v in fams.items():
        print(
            name,
            {
                c: (round(x["A"], 1), x["v0"], round(x["scatter_holdout"], 3), round(x["bkg_trend_dex_per_100"], 3))
                for c, x in v.items()
            },
        )
    best = min(fams, key=lambda k: np.mean([fams[k][c]["scatter_holdout"] for c in CHANNELS]))
    if best != "sinh":
        raise RuntimeError(f"detection code implements the sinh inverse only, but {best} fits best")
    # One common black level keeps the linearisation a single transform.
    v0 = float(np.median([fams[best][c]["v0"] for c in CHANNELS]))
    stretch = {"A": [fams[best][c]["A"] for c in CHANNELS], "v0": v0}

    shapes = pmap(fit_psf, [(l, b, stretch) for l, b in zip(src.LII, src.BII)])
    ps = pd.DataFrame([s if s else {} for s in shapes], index=src.index)
    src = pd.concat([src, ps], axis=1)
    ok = src.cxx.notna()
    d = src[ok]
    a_mat = np.concatenate(
        [np.c_[d.mxx, np.ones(len(d))], np.c_[d.myy, np.ones(len(d))], np.c_[d.mxy, np.zeros(len(d))]]
    )
    y = np.concatenate([d.cxx, d.cyy, d.cxy])
    # Robust linear fit (soft-L1) of C = s2 * M + t2 * I.
    res = optimize.least_squares(lambda p: a_mat @ p - y, [0.3, 6.0], loss="soft_l1", f_scale=2.0)
    s2, t2 = res.x
    psf_models = {}
    for name, cols in [("pixel_only", [0]), ("sky_only", [1]), ("sky_plus_pixel", [0, 1])]:
        coef = optimize.least_squares(
            lambda p: a_mat[:, cols] @ p - y, np.ones(len(cols)), loss="soft_l1", f_scale=2.0
        ).x
        psf_models[name] = {"coef": coef.tolist(), "robust_rms": robust_sigma(y - a_mat[:, cols] @ coef)}
    psf = {"t2": float(t2), "s2": float(s2), "sky_sigma_arcmin": float(np.sqrt(t2) * SCALE * 60)}
    print("psf", psf, psf_models)

    save_json(
        {
            "stretch": stretch,
            "stretch_family": best,
            "stretch_families": fams,
            "psf": psf,
            "psf_models": psf_models,
            "n_sources": int(len(src)),
            "n_psf": int(ok.sum()),
        },
        OUTDIR / "calibration.json",
    )
    src.to_parquet(OUTDIR / "sources.parquet")

    plot_results(src, best, fams, d, s2, t2)


def plot_results(src, best, fams, d, s2, t2):
    """Render diagnostics separately from the expensive analysis."""
    fn = families()[best]
    fig, axs = plt.subplots(2, 3, figsize=figure_size(2, 3))
    for k, c in enumerate(CHANNELS):
        p = fams[best][c]
        dlin = fn(src[f"peak_{c}"], p["A"], p["v0"]) - fn(src[f"bkg_{c}"], p["A"], p["v0"])
        ax = axs[0, k]
        sc = ax.scatter(src[BAND_OF[c]], dlin, c=src[f"bkg_{c}"], s=3, vmax=150)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(rf"$F_{{\mathrm{{{BAND_OF[c].removeprefix('ML_FLUX_')}}}}}$ [erg s$^{{-1}}$ cm$^{{-2}}$]")
        ax.set_ylabel(f"linearised peak increment ({c})")
        ax.set_title(f"{c}: {best} A={p['A']:.0f}, scatter {p['scatter_holdout']:.3f} dex")
        fig.colorbar(sc, ax=ax, label="raw background")
        ax = axs[1, k]
        for name, v in fams.items():
            ax.bar(name, v[c]["scatter_holdout"])
        ax.set_ylabel("hold-out robust scatter [dex]")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "stretch.pdf")

    fig, axs = plt.subplots(1, 2, figsize=figure_size(1, 2))
    tr = d.cxx + d.cyy
    anis_meas = np.hypot(d.cxx - d.cyy, 2 * d.cxy) / tr
    anis_pix = np.hypot(d.mxx - d.myy, 2 * d.mxy) / (d.mxx + d.myy)
    mod = psf_cov_components(d, s2, t2)
    anis_mod = np.hypot(mod[0] - mod[1], 2 * mod[2]) / (mod[0] + mod[1])
    axs[0].scatter(anis_mod, anis_meas, s=3, c=anis_pix)
    axs[0].plot([0, 1], [0, 1], "k-")
    axs[0].set_xlabel("model ellipticity")
    axs[0].set_ylabel("measured ellipticity")
    axs[1].scatter(d.ML_FLUX_1, np.sqrt(tr / 2) * SCALE * 60, s=3)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$F$ (0.2–2.3 keV)")
    axs[1].set_ylabel("rms width [arcmin] (flat if stretch is right)")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "psf.pdf")


def psf_cov_components(d, s2, t2):
    """Model covariance components (xx, yy, xy) for rows of d."""
    return t2 + s2 * d.mxx.values, t2 + s2 * d.myy.values, s2 * d.mxy.values


if __name__ == "__main__":
    main()
