"""Step 03: refine the image<->sky alignment; one Hammer-Aitoff projection per half.

The two halves of the image were projected separately (a single projection leaves ~1.5 px
residuals on the Russian side), so each half gets its own fit of (x0, y0, ax, ay, rot, l0):

* DE half: bright (F > 3e-12), point-like, isolated eRASS1-DE sources.
* RU half: point-like ROSAT BSC sources (no eRASS1-RU catalogue exists).
* The BSC fit is repeated on the DE half to show it reproduces the eRASS1 answer.
* Mollweide and low-order polynomial distortions are tested as alternatives on the DE half.
* Sources inside known supernova remnants (SNRcat) are not used.
* Every fit uses a random training half; statistics are quoted on the hold-out half. XMMSL3
  gives a further independent check.

Positions are image centroids of background-subtracted R+G luminance in pixel space (the
projection is locally affine, so the pixel centroid maps onto the sky centroid). Blue is left out
because it is not registered with R and G in the DE half: after the projection fit, per-channel
shifts are measured for each half from unsaturated sources (linearised channel centroids).

Outputs (out/03_align/): alignment.json, align_sources_{de,bsc}.parquet, residuals.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy.spatial import cKDTree

from epngcat.config import OUT, SEED, is_german, load_json, save_json, step_dir
from epngcat.evaluate import in_snr
from epngcat.image import load_rgb
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import AllSkyProjection, SplitProjection, forward_norm
from epngcat.sky import radec_unit, sep_arcsec

OUTDIR = step_dir("03_align")
FLUX_MIN = 3e-12
ISO_RADIUS = 20 / 60  # deg: no neighbour brighter than 10% of the source within this radius
EDGE_GAP = 1.0  # deg from the joins (l = 0 / 180) where a source may be split between halves
MIN_PEAK = 20  # luminance above background for a usable centroid
PARAMS = ["x0", "y0", "ax", "ay", "rot", "l0"]


def away_from_joins(l):
    """Flag longitudes further than EDGE_GAP from l = 0 and l = 180."""
    d = np.abs(((np.asarray(l) + 90) % 180) - 90)
    return d > EDGE_GAP


def isolated(l, b, flux, sel_l, sel_b, sel_flux, frac=0.1):
    """No reference neighbour brighter than frac * flux within ISO_RADIUS (self excluded)."""
    tree = cKDTree(radec_unit(l, b))
    chord = 2 * np.sin(np.radians(ISO_RADIUS) / 2)
    nbs = tree.query_ball_point(radec_unit(sel_l, sel_b), chord)
    return np.array([np.sum(flux[nb] > frac * f) <= 1 for nb, f in zip(nbs, sel_flux)])


def erass1_sources(flux_min=FLUX_MIN, flux_max=np.inf, frac=0.1):
    """Bright isolated eRASS1-DE point sources."""
    cat = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    cat = cat[np.isfinite(cat.ML_FLUX_1)]
    br = cat[cat.ML_FLUX_1.between(flux_min, flux_max) & (cat.EXT == 0) & away_from_joins(cat.LII)]
    br = br[~in_snr(br.LII.values, br.BII.values)]
    iso = isolated(cat.LII.values, cat.BII.values, cat.ML_FLUX_1.values, br.LII, br.BII, br.ML_FLUX_1, frac)
    return pd.DataFrame({"l": br.LII[iso], "b": br.BII[iso], "flux": br.ML_FLUX_1[iso]}).reset_index(drop=True)


def bsc_sources(count_min=0.1):
    """Isolated point-like ROSAT BSC sources, all sky, away from the Galactic plane and joins."""
    bsc = pd.read_parquet(OUT / "01_catalogs" / "rass_bsc.parquet")
    g = SkyCoord(bsc.RA.values, bsc.DEC.values, unit="deg").galactic
    df = pd.DataFrame({"l": g.l.deg, "b": g.b.deg, "flux": bsc.FLUX.values})
    iso = isolated(df.l.values, df.b.values, df.flux.values, df.l, df.b, df.flux)
    keep = (
        iso
        & ~in_snr(df.l.values, df.b.values)
        & (bsc.Extent.values == 0)
        & (bsc.Count.values > count_min)
        & (np.abs(df.b) > 3)
        & away_from_joins(df.l)
    )
    return df[keep].reset_index(drop=True)


def xmmsl_sources():
    """Bright clean XMMSL3 detections (independent check)."""
    x = pd.read_parquet(OUT / "01_catalogs" / "xmmsl3.parquet")
    x = x[(x.FLUX_B6 > 3e-12) & (x.EXT_B6 == 0) & away_from_joins(x.LII) & ~in_snr(x.LII.values, x.BII.values)]
    return pd.DataFrame({"l": x.LII.values, "b": x.BII.values, "flux": x.FLUX_B6.values})


def centroid(lum, px, py, search=4, rad=2.5, ann=(6, 10)):
    """Peak search then background-subtracted centroid around it (pixel coordinates).

    Returns (n, 3): centroid x, centroid y, mean peak luminance above background (NaN if unusable).
    """
    ny, nx = lum.shape
    out = np.full((len(px), 3), np.nan)
    yy, xx = np.mgrid[-ann[1] : ann[1] + 1, -ann[1] : ann[1] + 1]
    rr = np.hypot(xx, yy)
    m = rr <= rad
    for k, (x, y) in enumerate(zip(px, py)):
        ix, iy = int(round(x)), int(round(y))
        pad = ann[1] + search
        if not (pad <= ix < nx - pad and pad <= iy < ny - pad):
            continue
        w = lum[iy - search : iy + search + 1, ix - search : ix + search + 1]
        j, i = np.unravel_index(np.argmax(w), w.shape)
        cx, cy = ix - search + i, iy - search + j
        cut = lum[cy - ann[1] : cy + ann[1] + 1, cx - ann[1] : cx + ann[1] + 1]
        bkg = np.median(cut[(rr >= ann[0]) & (rr <= ann[1])])
        wt = np.clip(cut[m] - bkg, 0, None)
        if wt.sum() > 0:
            out[k] = cx + (wt * xx[m]).sum() / wt.sum(), cy + (wt * yy[m]).sum() / wt.sum(), cut[rr <= 1].mean() - bkg
    return out


def fit_projection(start, l, b, mx, my):
    """Robust least-squares fit of projection parameters to measured pixel positions."""

    def resid(p):
        px, py = AllSkyProjection(kind=start.kind, **dict(zip(PARAMS, p))).sky2pix(l, b)
        return np.concatenate([px - mx, py - my])

    p0 = [getattr(start, k) for k in PARAMS]
    res = optimize.least_squares(resid, p0, loss="soft_l1", f_scale=1.0, x_scale=[1, 1, 1, 1, 1e-3, 1e-2])
    return AllSkyProjection(kind=start.kind, **dict(zip(PARAMS, res.x)))


def iterate_fit(lum, proj, src, train):
    """Centroid with current predictions and refit, three times. Returns (proj, measurements)."""
    for it in range(3):
        m = centroid(lum, *proj.sky2pix(src.l.values, src.b.values), search=4 if it == 0 else 2)
        tr = train & np.isfinite(m[:, 0]) & (m[:, 2] > MIN_PEAK)
        proj = fit_projection(proj, src.l.values[tr], src.b.values[tr], m[tr, 0], m[tr, 1])
    m = centroid(lum, *proj.sky2pix(src.l.values, src.b.values), search=2)
    return proj, m


def robust_stats(s):
    """Median (with bootstrap standard error) / 68% / 90% of separations [arcsec]."""
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return {"n": 0}
    boot = np.median(np.random.default_rng(SEED).choice(s, (500, len(s))), axis=1)
    pct = np.percentile(s, [50, 68, 90])
    return {"n": len(s), "median": pct[0], "median_se": boot.std(), "p68": pct[1], "p90": pct[2]}


def poly_terms(x, y, order):
    """Monomials x^i y^j with 1 <= i + j <= order."""
    return np.stack([x**i * y**j for n in range(1, order + 1) for i in range(n + 1) for j in [n - i]], -1)


def distortion_holdout(proj, src, m, train, ok, order):
    """Hold-out separations after a polynomial pixel-offset correction fitted on the training set."""
    l, b = src.l.values, src.b.values
    px, py = proj.sky2pix(l, b)
    dx, dy = m[:, 0] - px, m[:, 1] - py
    if order:
        t = poly_terms(*forward_norm(np.radians(l - proj.l0), np.radians(b), proj.kind), order)
        tr = train & ok
        dx = dx - t @ np.linalg.lstsq(t[tr], dx[tr], rcond=None)[0]
        dy = dy - t @ np.linalg.lstsq(t[tr], dy[tr], rcond=None)[0]
    ho = ~train & ok
    # Map the corrected measured pixel through the exact inverse to get a sky separation.
    ml, mb = proj.pix2sky(px[ho] + dx[ho], py[ho] + dy[ho])
    return robust_stats(sep_arcsec(l[ho], b[ho], ml, mb))


def evaluate(split, lum, src):
    """Per-half positional statistics of a split projection against a reference set."""
    m = centroid(lum, *split.sky2pix(src.l.values, src.b.values), search=2)
    ok = np.isfinite(m[:, 0]) & (m[:, 2] > MIN_PEAK)
    ml, mb = split.pix2sky(m[:, 0], m[:, 1])
    s = np.where(ok, sep_arcsec(src.l.values, src.b.values, ml, mb), np.nan)
    left = split.left_sky(src.l.values)
    return {"de": robust_stats(s[~left]), "ru": robust_stats(s[left])}, s


def channel_offsets(split, rgb, lin, l, b, left_half, shift):
    """Median per-channel centroid offset (native px) from positions predicted with `shift`."""
    sel = split.left_sky(l) == left_half
    yy, xx = np.mgrid[-4:5, -4:5]
    rr = np.hypot(xx, yy)
    big = np.hypot(*np.mgrid[-8:9, -8:9]) > 6
    offs = []
    for ls, bs in zip(l[sel], b[sel]):
        row = []
        for k in range(3):
            x = split.sky2pix(np.array([ls]), np.array([bs]))[0][0] + shift[k][0]
            y = split.sky2pix(np.array([ls]), np.array([bs]))[1][0] + shift[k][1]
            ix, iy = int(round(x)), int(round(y))
            if not (8 <= ix < rgb.shape[1] - 8 and 8 <= iy < rgb.shape[0] - 8):
                break
            if (rgb[iy - 4 : iy + 5, ix - 4 : ix + 5][rr <= 1.5] >= 250).any():
                break
            bk = np.median(lin[iy - 8 : iy + 9, ix - 8 : ix + 9][big], 0)[k]
            w = np.clip(lin[iy - 4 : iy + 5, ix - 4 : ix + 5, k] - bk, 0, None) * (rr <= 2.5)
            if w.sum() <= 0.3:
                break
            row += [(w * xx).sum() / w.sum() + ix - x, (w * yy).sum() / w.sum() + iy - y]
        else:
            offs.append(row)
    offs = np.array(offs)
    med = np.median(offs, 0).reshape(3, 2)
    se = (1.2533 * offs.std(0) / np.sqrt(len(offs))).reshape(3, 2)
    return med, se, len(offs)


def channel_shifts(split, rgb, l, b, left_half, iterations=4):
    """Per-channel registration shifts (native px) of one half, by iterated centroiding.

    The centroid aperture is centred on the predicted position, so a displaced channel is
    truncated asymmetrically and its offset is underestimated (the DE blue channel measures 0.82 px
    in the first pass and converges to ~1.0 px). Iterating with the current shift applied removes
    that bias; the last residual is reported as a convergence check.

    Sources need an unsaturated core; channels are linearised with the display-stretch scale of the
    exploratory calibration (sinh, A ~ 100-200) - the centroid hardly depends on it, and step 05
    (which needs registered patches) comes later.
    """
    lin = np.sinh(rgb / np.array([90.0, 140.0, 190.0]))
    shift = np.zeros((3, 2))
    for _ in range(iterations):
        med, se, n = channel_offsets(split, rgb, lin, l, b, left_half, shift)
        shift = shift + med
    return shift, med, se, n


def main():
    """Run step 03."""
    coarse = load_json(OUT / "02_align_coarse" / "alignment_coarse.json")["projections"]
    rgb = load_rgb()
    lum = rgb[..., 0] + rgb[..., 1]
    rng = np.random.default_rng(SEED)
    er, bsc, xmm = erass1_sources(), bsc_sources(), xmmsl_sources()
    er = er[is_german(er.l)].reset_index(drop=True)
    er_train = rng.random(len(er)) < 0.5
    bsc_train = rng.random(len(bsc)) < 0.5
    left = (bsc.l.values % 360) < 180
    report = {"n_sources": {"erass1_de": len(er), "bsc_de": int((~left).sum()), "bsc_ru": int(left.sum())}}
    print(report)

    # DE half from eRASS1: projection type and distortion comparison.
    comparison = {}
    fits = {}
    for kind in ("hammer", "mollweide"):
        proj, m = iterate_fit(lum, AllSkyProjection.from_dict(coarse[kind]), er, er_train)
        ok = np.isfinite(m[:, 0]) & (m[:, 2] > MIN_PEAK)
        fits[kind] = (proj, m)
        for order in range(5):
            comparison[f"{kind}_poly{order}"] = distortion_holdout(proj, er, m, er_train, ok, order)
            print(f"DE {kind} poly{order}", comparison[f"{kind}_poly{order}"])
    de, m_de = fits["hammer"]
    ho0 = comparison["hammer_poly0"]
    better = {k: v for k, v in comparison.items() if v["median"] < ho0["median"] - ho0["median_se"]}
    assert not better, f"projection model beyond plain Hammer-Aitoff needed: {better}"

    # RU half from BSC (and the same on DE as a method cross-check).
    ru, _ = iterate_fit(lum, AllSkyProjection.from_dict(coarse["hammer"]), bsc[left], bsc_train[left])
    de_bsc, _ = iterate_fit(lum, AllSkyProjection.from_dict(coarse["hammer"]), bsc[~left], bsc_train[~left])
    single = SplitProjection(de, de)
    split = SplitProjection(de, ru)

    # Per-channel registration, per half (relative to the R+G projection).
    faint = erass1_sources(flux_min=7e-13, flux_max=3e-11, frac=0.05)
    faint = faint[is_german(faint.l)]
    bsc_all = bsc_sources(count_min=0.15)
    sh_de, res_de, se_de, n_de = channel_shifts(split, rgb, faint.l.values, faint.b.values, False)
    sh_ru, res_ru, se_ru, n_ru = channel_shifts(split, rgb, bsc_all.l.values, bsc_all.b.values, True)
    split = SplitProjection(de, ru, tuple(map(tuple, sh_de)), tuple(map(tuple, sh_ru)))
    report["channel_shifts"] = {
        "de": {"shift_px": sh_de.tolist(), "residual_px": res_de.tolist(), "stderr_px": se_de.tolist(), "n": n_de},
        "ru": {"shift_px": sh_ru.tolist(), "residual_px": res_ru.tolist(), "stderr_px": se_ru.tolist(), "n": n_ru},
    }
    print("channel shifts", report["channel_shifts"])

    xd, yd = de.sky2pix(er.l.values, er.b.values)
    xb, yb = de_bsc.sky2pix(er.l.values, er.b.values)
    report["de_bsc_vs_erass1_projection_max_offset_px"] = float(np.hypot(xd - xb, yd - yb).max())
    er_ho = er[~er_train]
    bsc_ho = bsc[~bsc_train]
    for name, sp in [("single_projection", single), ("split_projection", split)]:
        report[name] = {
            "erass1_holdout": evaluate(sp, lum, er_ho)[0]["de"],
            "bsc_holdout": evaluate(sp, lum, bsc_ho)[0],
            "xmmsl3": evaluate(sp, lum, xmm)[0],
        }
        print(name, report[name])
    report["projection_comparison_de_erass1_holdout"] = comparison
    save_json({"projection": split.to_dict(), "report": report}, OUTDIR / "alignment.json")
    er.assign(train=er_train, mx=m_de[:, 0], my=m_de[:, 1], peak=m_de[:, 2]).to_parquet(
        OUTDIR / "align_sources_de.parquet"
    )
    bsc.assign(train=bsc_train).to_parquet(OUTDIR / "align_sources_bsc.parquet")

    plot_results(lum, bsc_ho, single, split)


def plot_results(lum, bsc_ho, single, split):
    """Render diagnostics separately from the expensive analysis."""
    # Diagnostics: pixel residuals of BSC hold-out sources for single vs split projection.
    fig, axs = plt.subplots(2, 2, figsize=figure_size(2, 2))
    for row, (name, sp) in enumerate([("single (DE) projection", single), ("split projection", split)]):
        px, py = sp.sky2pix(bsc_ho.l.values, bsc_ho.b.values)
        m = centroid(lum, px, py, search=3)
        ok = np.isfinite(m[:, 0]) & (m[:, 2] > MIN_PEAK)
        dx, dy = m[:, 0] - px, m[:, 1] - py
        for col, (x, lab) in enumerate([(bsc_ho.l.values, r"$l$ [deg]"), (bsc_ho.b.values, r"$b$ [deg]")]):
            ax = axs[row, col]
            ax.scatter(x[ok], dx[ok], s=2, label="dx")
            ax.scatter(x[ok], dy[ok], s=2, label="dy")
            ax.set_ylim(-3, 3)
            ax.set_xlabel(lab)
            ax.set_ylabel("measured - model [px]")
            ax.set_title(f"BSC hold-out, {name}")
            ax.legend(markerscale=4)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "residuals.pdf")


if __name__ == "__main__":
    main()
