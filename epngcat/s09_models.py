"""Step 09: select position and flux models on the DE half (train -> val).

1. Input: detections + measurements on all DE tiles from step 08 (the test split is only used
   in step 10).
2. Counterparts: brightest eRASS1 source (F >= F_FLOOR) within MATCH_R; model training/selection
   uses point-like counterparts within TRAIN_SEP (fewer wrong associations), outside SNRs.
3. Positions: estimators det, cen, psf, psfw, cenn, psfn, psfnw, scene (scene-model refit with
   B-spline background), each optionally with an astrometric correction (sky-frame offset
   model per half, linear in image coordinates, fitted on train), plus a blend choosing per S/N
   bin the estimator and correction with the lowest median separation on train. Best = lowest
   median separation on val.
   Position error model: Rayleigh sigma in S/N bins, interpolated in log S/N and forced
   non-increasing, with a separate constant for saturated sources.
4. Fluxes (log10 F, 0.2-2.3 keV and bands P1-P3): parametric PSF-amplitude and aperture
   calibrations, ridge regression on engineered features, sklearn HistGradientBoosting and
   XGBoost (hyper-parameters tuned with an internal split of train), all trained on train and
   compared on val (robust scatter). Flux uncertainty: quantile gradient boosting (16/84%).

Outputs (out/09_models/): de_matched.parquet, position_models.csv, flux_models.csv,
models.pkl, selection.json, positions.pdf, fluxes.pdf
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import HistGradientBoostingRegressor

from epngcat.config import OUT, SEED, save_json, step_dir
from epngcat.evaluate import catalog, match_detections
from epngcat.fluxmodels import HGBFlux, ParametricFlux, RidgeFlux, XGBFlux, features, internal_split
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import forward_norm, load_alignment
from epngcat.sky import sep_arcsec

OUTDIR = step_dir("09_models")
TRAIN_SEP = 150.0  # arcsec
POS_EST = ["det", "cen", "psf", "psfw", "cenn", "psfn", "psfnw", "scene"]
SNR_EDGES = [0, 5, 6, 8, 10, 15, 25, np.inf]  # bins of the per-S/N blended estimator (saturated: own bin)
BANDS = {"1": "cat_ML_FLUX_1", "P1": "cat_ML_FLUX_P1", "P2": "cat_ML_FLUX_P2", "P3": "cat_ML_FLUX_P3"}


def with_counterparts(det):
    """Attach counterparts per split (catalogue of the same split, SNRs removed)."""
    parts = []
    for split in ("train", "val", "test"):
        d = det[(det.split == split) & ~det.in_snr]
        parts.append(match_detections(d.reset_index(drop=True), catalog(split)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- positions


MIN_SHIFT_PAIRS = 50
JOIN_GAP = 1.0  # deg


def image_xy(split_proj, l, b):
    """Normalised Hammer coordinates (x, y) of positions, using the projection of their half."""
    left = split_proj.left_sky(l)
    x, y = np.zeros(len(l)), np.zeros(len(l))
    for half, sel in ((split_proj.de, ~left), (split_proj.ru, left)):
        x[sel], y[sel] = forward_norm(np.radians(l[sel] - half.l0), np.radians(b[sel]), half.kind)
    return x, y


def sky_offsets(l, b, cl, cb):
    """Offsets catalogue - estimate [arcsec]: (dl cos b, db)."""
    dl = ((cl - l + 180) % 360 - 180) * np.cos(np.radians(cb)) * 3600
    return dl, (cb - b) * 3600


def fit_sky_shift(split_proj, d, est):
    """Per-half sky-frame astrometric correction, linear in image coordinates.

    (dl cos b, db) = c0 + c1 x + c2 y, robust (soft-L1) fit on pairs further than JOIN_GAP from the
    image joins. The measured offsets are close to constant on the sky (not in pixels), with mild
    trends across the image. A half with fewer than MIN_SHIFT_PAIRS pairs (in practice the RU half,
    which has no training catalogue) gets no correction.
    """
    l, b = d[f"l_{est}"].values, d[f"b_{est}"].values
    ok = np.isfinite(l) & (np.abs(((d.cat_LII.values + 90) % 180) - 90) > JOIN_GAP)
    dl, db = sky_offsets(l, b, d.cat_LII.values, d.cat_BII.values)
    x, y = image_xy(split_proj, np.where(ok, l, 0), np.where(ok, b, 0))
    left = split_proj.left_sky(np.where(ok, l, 0))
    out = {}
    for half, sel in (("de", ok & ~left), ("ru", ok & left)):
        if sel.sum() < MIN_SHIFT_PAIRS:
            out[half] = None
            continue
        a = np.c_[np.ones(sel.sum()), x[sel], y[sel]]
        out[half] = {
            comp: optimize.least_squares(
                lambda c, v=v: a @ c - v[sel], np.zeros(3), loss="soft_l1", f_scale=30.0
            ).x.tolist()
            for comp, v in (("dl", dl), ("db", db))
        }
    return out


def bin_intercepts(split_proj, d, est, model):
    """Median residual (after the global sky model) per half, used as per-S/N-bin intercept."""
    l, b = apply_sky_shift(split_proj, d[f"l_{est}"].values, d[f"b_{est}"].values, model)
    ok = np.isfinite(l) & (np.abs(((d.cat_LII.values + 90) % 180) - 90) > JOIN_GAP)
    dl, db = sky_offsets(l, b, d.cat_LII.values, d.cat_BII.values)
    left = split_proj.left_sky(np.where(ok, l, 0))
    out = {}
    for half, sel in (("de", ok & ~left), ("ru", ok & left)):
        enough = sel.sum() >= MIN_SHIFT_PAIRS and model.get(half) is not None
        out[half] = [float(np.median(dl[sel])), float(np.median(db[sel]))] if enough else [0.0, 0.0]
    return out


def apply_sky_shift(split_proj, l, b, model, intercept=None):
    """Apply a sky-frame correction model (and optional per-half intercepts) to positions."""
    l, b = np.asarray(l, float).copy(), np.asarray(b, float).copy()
    ok = np.isfinite(l)
    x, y = image_xy(split_proj, np.where(ok, l, 0), np.where(ok, b, 0))
    left = split_proj.left_sky(np.where(ok, l, 0))
    dl, db = np.zeros(len(l)), np.zeros(len(l))
    for half, sel in (("de", ok & ~left), ("ru", ok & left)):
        m = model.get(half)
        if m is None:
            continue
        a = np.c_[np.ones(sel.sum()), x[sel], y[sel]]
        dl[sel], db[sel] = a @ m["dl"], a @ m["db"]
        if intercept is not None:
            dl[sel] += intercept[half][0]
            db[sel] += intercept[half][1]
    cosb = np.maximum(np.cos(np.radians(b)), 0.05)
    l[ok] = (l[ok] + dl[ok] / cosb[ok] / 3600) % 360
    b[ok] = np.clip(b[ok] + db[ok] / 3600, -90, 90)
    return l, b


def position_stats(sep):
    """Median / 68% / 90% separation [arcsec] and number."""
    s = sep[np.isfinite(sep)]
    return {"n": len(s), "median": np.median(s), "p68": np.percentile(s, 68), "p90": np.percentile(s, 90)}


def snr_bin(d):
    """S/N bin index of detections (saturated blobs get their own bin)."""
    return np.where(d.sat_area.values > 0, len(SNR_EDGES) - 1, np.digitize(d.snr.values, SNR_EDGES[1:-1]))


def corrected_positions(d, split_proj, est, corr, shifts, bin_key=None):
    """Positions of one estimator; NaN where the estimator failed.

    corr: False (none), "sky" (per-half sky-frame model of the estimator) or "skybin" (the same
    plus a per-half intercept fitted within S/N bin `bin_key`, stored as shifts["<est>@<bin>"]).
    """
    l, b = d[f"l_{est}"].values, d[f"b_{est}"].values
    if not corr:
        return l, b
    intercept = shifts[f"{est}@{bin_key}"] if corr == "skybin" else None
    return apply_sky_shift(split_proj, l, b, shifts[est], intercept)


def blend_choice(tr, split_proj, shifts):
    """Per S/N bin, the (estimator, correction) pair with the lowest median train separation.

    Bin intercepts are added to `shifts` in place (bright and faint sources can be offset differently).
    """
    bins = snr_bin(tr)
    choice = {}
    for k in np.unique(bins):
        t = tr[bins == k]
        best = None
        for est in POS_EST:
            shifts[f"{est}@{k}"] = bin_intercepts(split_proj, t, est, shifts[est])
            for corr in (False, "sky", "skybin"):
                l, b = corrected_positions(t, split_proj, est, corr, shifts, k)
                med = np.nanmedian(np.where(np.isnan(l), np.inf, sep_arcsec(l, b, t.cat_LII.values, t.cat_BII.values)))
                if best is None or med < best[0]:
                    best = (med, est, corr)
        choice[int(k)] = [best[1], best[2]]
    return choice


def select_positions(m, split_proj):
    """Compare position estimators (with/without astrometric correction, and the S/N blend) on val."""
    tr = m[(m.split == "train") & (m.cat_EXT == 0) & (m.sep < TRAIN_SEP)]
    va = m[(m.split == "val") & (m.cat_EXT == 0) & m.cat_ML_FLUX_1.notna()]
    shifts = {est: fit_sky_shift(split_proj, tr, est) for est in POS_EST}
    blend = blend_choice(tr, split_proj, shifts)
    rows = []
    sat = va.sat_area.values > 0
    for est, corr in [(e, c) for e in POS_EST for c in (False, "sky")] + [("blend", None)]:
        if est == "blend":
            l, b, _ = final_position(va, split_proj, ("blend", blend), shifts)
            fail = 0.0
        else:
            l, b = corrected_positions(va, split_proj, est, corr, shifts)
            fail = np.isnan(l).mean()
            l, b = np.where(np.isnan(l), va.l_det, l), np.where(np.isnan(l), va.b_det, b)  # fallback to det
        s = sep_arcsec(l, b, va.cat_LII.values, va.cat_BII.values)
        rows.append(
            {
                "estimator": est,
                "corrected": corr,
                "fail_frac": fail,
                **position_stats(s),
                "median_nonsat": np.median(s[~sat]),
                "median_sat": np.median(s[sat]) if sat.any() else np.nan,
                "median_snr_gt10": np.median(s[va.snr.values > 10]),
            }
        )
    table = pd.DataFrame(rows).sort_values("median")
    return table, shifts, blend


def final_position(d, split_proj, choice, shifts):
    """Chosen estimator with correction; falls back to the detection position where it failed.

    choice: (estimator, corrected) or ("blend", {snr_bin: [estimator, corrected]}).
    """
    est, corr = choice
    if est == "blend":
        l, b = np.full(len(d), np.nan), np.full(len(d), np.nan)
        bins = snr_bin(d)
        for k, (e, c) in corr.items():
            sel = bins == int(k)
            if sel.any():
                l[sel], b[sel] = corrected_positions(d[sel], split_proj, e, c, shifts, int(k))
    else:
        l, b = corrected_positions(d, split_proj, est, corr, shifts)
    fb = ~np.isfinite(l)
    return np.where(fb, d.l_det, l), np.where(fb, d.b_det, b), fb


def fit_pos_error(snr, sat, sep, catpos):
    """Binned sigma(snr) for unsaturated, constant for saturated (Rayleigh medians).

    sigma drops steeply where the selected estimator switches from centroids to profile fits,
    which sqrt(a^2 + (b/snr)^2) cannot follow (it overestimated the brightest bins by ~1.5x), so
    the binned values are interpolated in log S/N instead, forced non-increasing with S/N.
    """
    # eRASS1 POS_ERR is the radial 1-sigma error, so its expected contribution to radial
    # separation squared is POS_ERR^2 (per-axis sigma = POS_ERR/sqrt(2)).
    sep2 = np.sqrt(np.clip(sep**2 - catpos**2, 0, None))
    q = pd.qcut(snr[~sat], 10)
    grp = pd.DataFrame({"q": q, "snr": snr[~sat], "sep": sep2[~sat]}).groupby("q", observed=True)
    s_med = grp.snr.median().values
    sig = np.minimum.accumulate(grp.sep.median().values / np.sqrt(2 * np.log(2)))
    sig_sat = float(np.median(sep2[sat]) / np.sqrt(2 * np.log(2))) if sat.any() else float(sig[-1])
    return {
        "sat": sig_sat,
        "bins_snr": s_med.tolist(),
        "bins_sigma": sig.tolist(),
    }


def pos_error(model, snr, sat):
    """1-sigma per-axis positional uncertainty [arcsec]."""
    # np.interp holds the end bins constant outside the calibrated S/N range.
    s = np.interp(np.log(np.asarray(snr, float)), np.log(model["bins_snr"]), model["bins_sigma"])
    return np.where(sat, model["sat"], s)


# ---------------------------------------------------------------- fluxes


def flux_stats(y, p):
    """Robust scatter, bias and outlier fraction of log residuals."""
    r = p - y
    return {
        "n": len(r),
        "bias": float(np.median(r)),
        "scatter": float(1.4826 * np.median(np.abs(r - np.median(r)))),
        "rms": float(np.sqrt(np.mean(r**2))),
        "out_0.3dex": float(np.mean(np.abs(r - np.median(r)) > 0.3)),
    }


def flux_models():
    """Candidate flux models (name -> factory)."""
    return {
        "param_psf_native": lambda: ParametricFlux(["A_R_n", "A_G_n", "A_B_n"]),
        "param_psf_scene": lambda: ParametricFlux(["A_R_s", "A_G_s", "A_B_s"]),
        "param_aperture": lambda: ParametricFlux(["ap6_R", "ap6_G", "ap6_B"]),
        "ridge": RidgeFlux,
        "hgb": HGBFlux,
        "xgb": XGBFlux,
    }


def train_flux(tr, va, target):
    """Fit every flux model for one band; return results table and fitted models."""
    rows, fitted = [], {}
    ftr, fva = features(tr), features(va)
    ytr, yva = np.log10(tr[target].values), np.log10(va[target].values)
    gtr, gva = np.isfinite(ytr), np.isfinite(yva)
    inner = internal_split(tr[gtr])
    for name, factory in flux_models().items():
        m = factory()
        args = (ftr[gtr], ytr[gtr], inner) if name in ("hgb", "xgb") else (ftr[gtr], ytr[gtr])
        m.fit(*args)
        p = m.predict(fva[gva])
        sat = va.sat_area.values[gva] > 0
        rows.append(
            {
                "model": name,
                **flux_stats(yva[gva], p),
                "scatter_sat": flux_stats(yva[gva][sat], p[sat])["scatter"] if sat.sum() > 5 else np.nan,
                "scatter_nonsat": flux_stats(yva[gva][~sat], p[~sat])["scatter"],
                "params": getattr(m, "params", None),
            }
        )
        fitted[name] = m
        print(target, rows[-1], flush=True)
    return pd.DataFrame(rows).sort_values("scatter"), fitted


def quantile_models(tr, target):
    """16% / 84% quantile gradient boosting for flux uncertainties."""
    f, y = features(tr), np.log10(tr[target].values)
    g = np.isfinite(y)
    return {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.05, max_iter=400, max_leaf_nodes=15, random_state=SEED
        ).fit(f[g], y[g])
        for q in (0.16, 0.84)
    }


def main():
    """Run step 09."""
    split_proj = load_alignment()
    det = pd.read_parquet(OUT / "08_de_pipeline" / "de_pipeline.parquet")
    print("DE detections:", det.split.value_counts().to_dict())
    m = with_counterparts(det)

    # Positions.
    ptab, shifts, blend = select_positions(m, split_proj)
    ptab.to_csv(OUTDIR / "position_models.csv", index=False)
    print(ptab.to_string())
    best = ptab.iloc[0]
    choice = ("blend", blend) if best.estimator == "blend" else (best.estimator, best.corrected or False)
    m["l_fin"], m["b_fin"], m["pos_fallback"] = final_position(m, split_proj, choice, shifts)
    m["sep_fin"] = sep_arcsec(m.l_fin, m.b_fin, m.cat_LII, m.cat_BII)
    va = m[(m.split == "val") & (m.cat_EXT == 0) & m.cat_ML_FLUX_1.notna()]
    pos_err = fit_pos_error(va.snr.values, va.sat_area.values > 0, va.sep_fin.values, va.cat_POS_ERR.fillna(0).values)
    print("position error model", pos_err)

    # Fluxes: train on reliable point-like associations.
    good = (m.cat_EXT == 0) & (m.sep_fin < TRAIN_SEP)
    tr, va = m[good & (m.split == "train")], m[good & (m.split == "val")]
    results, fitted, quant = {}, {}, {}
    for band, target in BANDS.items():
        tab, models = train_flux(tr, va, target)
        tab.insert(0, "band", band)
        results[band], fitted[band] = tab, models
        quant[band] = quantile_models(tr, target)
    ftab = pd.concat(results.values(), ignore_index=True)
    ftab.to_csv(OUTDIR / "flux_models.csv", index=False)
    best_flux = results["1"].iloc[0].model
    print(ftab.drop(columns="params").to_string())

    # Quantile intervals are too narrow out of sample: widen them around the best prediction by a
    # per-band factor that gives 68% coverage on val (checked independently on test in step 10).
    coverage, widen = {}, {}
    for band, target in BANDS.items():
        y = np.log10(va[target].where(va[target] > 0).values)
        g = np.isfinite(y)
        fv = features(va[g])
        lo, hi = (quant[band][q].predict(fv) for q in (0.16, 0.84))
        p = fitted[band][best_flux].predict(fv)
        dlo, dhi = np.clip(p - lo, 1e-3, None), np.clip(hi - p, 1e-3, None)

        def cover(k, y=y[g], p=p, dlo=dlo, dhi=dhi):
            return np.mean((y >= p - k * dlo) & (y <= p + k * dhi))

        widen[band] = float(optimize.brentq(lambda k: cover(k) - 0.6827, 0.2, 20.0))
        coverage[band] = {
            "frac_in_16_84_raw": float(np.mean((y[g] >= lo) & (y[g] <= hi))),
            "widen_factor": widen[band],
            "median_halfwidth_dex": float(np.median(widen[band] * (dlo + dhi) / 2)),
        }
    selection = {
        "position": {
            "estimator": choice[0],
            "corrected": choice[1],
            "sky_shift_models": shifts,
            "error_model": pos_err,
            "val": best.to_dict(),
        },
        "flux": {
            "model": best_flux,
            "val": results["1"].iloc[0].drop("params").to_dict(),
            "quantile_coverage_val": coverage,
        },
        "train_sep_arcsec": TRAIN_SEP,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
    }
    save_json(selection, OUTDIR / "selection.json")
    with open(OUTDIR / "models.pkl", "wb") as fh:
        pickle.dump(
            {
                "flux": {b: fitted[b][best_flux] for b in BANDS},
                "quantiles": quant,
                "widen": widen,
                "position": selection["position"],
                "features": list(features(va.head(1)).columns),
            },
            fh,
        )
    m.to_parquet(OUTDIR / "de_matched.parquet")

    plot_results(m, va, pos_err, fitted, best_flux, results)


def plot_results(m, va, pos_err, fitted, best_flux, results):
    """Render diagnostics separately from the expensive analysis."""
    # Plots.
    fig, axs = plt.subplots(1, 2, figsize=figure_size(1, 2))
    vv = m[(m.split == "val") & (m.cat_EXT == 0) & m.cat_ML_FLUX_1.notna()]
    for est in POS_EST:
        s = np.sort(sep_arcsec(vv[f"l_{est}"], vv[f"b_{est}"], vv.cat_LII, vv.cat_BII).dropna())
        axs[0].plot(s, np.arange(len(s)) / len(s), label=est)
    axs[0].set_xlim(0, 300)
    axs[0].set_xlabel("separation [arcsec]")
    axs[0].set_ylabel("cumulative fraction (val)")
    axs[0].legend()
    x = np.geomspace(3, 300, 100)
    axs[1].scatter(vv.snr, vv.sep_fin, s=2, alpha=0.3)
    axs[1].plot(x, pos_error(pos_err, x, False) * np.sqrt(2 * np.log(2)), "r-", label="model median")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("S/N")
    axs[1].set_ylabel("separation [arcsec]")
    axs[1].legend()
    fig.tight_layout()
    save_figure(fig, OUTDIR / "positions.pdf")

    fig, axs = plt.subplots(4, 2, figsize=figure_size(4, 2))
    axs = axs.reshape(2, 4)
    for k, (band, target) in enumerate(BANDS.items()):
        y = np.log10(va[target].where(va[target] > 0).values)
        g = np.isfinite(y)
        p = fitted[band][best_flux].predict(features(va[g]))
        ax = axs[0, k]
        ax.scatter(y[g], p, s=3, c=va.sat_area.values[g] > 0, cmap="coolwarm")
        ax.plot([-15, -9], [-15, -9], "k-")
        ax.set_xlim(-14.5, -9.5)
        ax.set_ylim(-14.5, -9.5)
        ax.set_xlabel(rf"$\log_{{10}} F_{{\mathrm{{eRASS1}},{band}}}$")
        ax.set_ylabel(rf"$\log_{{10}} F_{{\mathrm{{image}},{band}}}$")
        ax = axs[1, k]
        t = results[band]
        ax.bar(t.model, t.scatter)
        ax.set_ylabel("val robust scatter [dex]")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "fluxes.pdf")


if __name__ == "__main__":
    main()
