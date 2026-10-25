"""Step 10: downstream source-model validation, conditional on global image registration.

* detection: reliability (overall, vs S/N), chance-corrected completeness vs flux, detection limit
  (F50 / F90) overall and vs |b| and background brightness
* positions: separation percentiles vs S/N; calibration of POS_ERR (Rayleigh check)
* fluxes: bias / scatter vs catalogue flux for all bands; saturated vs not; coverage of the
  16-84% interval

Also writes the image catalogue of the whole DE half (train/val entries are in-sample for the
models, flagged by SPLIT).

Outputs (out/10_validate_de/): de_catalog.parquet, metrics.json, validation_report.md, *.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epngcat.apply_models import apply
from epngcat.config import OUT, save_json, step_dir
from epngcat.evaluate import (
    F_FLOOR,
    MATCH_R,
    brightest_within,
    catalog,
    completeness,
    flux_at_completeness,
    match_detections,
    shifted,
)
from epngcat.plotting import figure_size, save_figure
from epngcat.s09_models import BANDS
from epngcat.sky import sep_arcsec

OUTDIR = step_dir("10_validate_de")
FLUX_BINS = np.geomspace(1e-14, 1e-10, 25)
SNR_BINS = [0, 5, 6, 8, 10, 15, 25, 1e9]


def rel_numbers(det, cat):
    """Chance-corrected reliability of a detection subset."""
    real = brightest_within(det.LII.values, det.BII.values, cat, MATCH_R, F_FLOOR) >= 0
    fake = brightest_within(*shifted(det.LII.values, det.BII.values), cat, MATCH_R, F_FLOOR) >= 0
    f, pc = real.mean(), fake.mean()
    return float((f - pc) / (1 - pc)), int(len(det))


def detection_metrics(c, cat):
    """Reliability and completeness summaries."""
    out = {"reliability": rel_numbers(c, cat)[0], "n": len(c)}
    grp = pd.cut(np.where(c.SATURATED, 1e8, c.SNR), SNR_BINS)
    out["reliability_vs_snr"] = {str(k): rel_numbers(g, cat) for k, g in c.groupby(grp, observed=True) if len(g) > 20}
    det = c.rename(columns={"LII": "l", "BII": "b"})
    comp, n = completeness(det, cat, FLUX_BINS)
    out["completeness"] = {"bins": FLUX_BINS.tolist(), "comp": comp.tolist(), "n": n.tolist()}
    out["F50"] = flux_at_completeness(0.5, comp, FLUX_BINS, n)
    out["F90"] = flux_at_completeness(0.9, comp, FLUX_BINS, n)
    from sklearn.isotonic import IsotonicRegression

    good = n > 0
    x = np.log10(np.sqrt(FLUX_BINS[1:] * FLUX_BINS[:-1]))[good]
    out["completeness_plateau"] = float(IsotonicRegression().fit(x, comp[good], sample_weight=n[good]).predict(x).max())
    out["F50_vs_absb"] = {}
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 90)]:
        sel = cat.BII.abs().between(lo, hi)
        dsel = det[det.b.abs().between(lo - 1, hi + 1)]
        cc, nn = completeness(dsel, cat[sel], FLUX_BINS)
        out["F50_vs_absb"][f"{lo}-{hi}"] = flux_at_completeness(0.5, cc, FLUX_BINS, nn)
    return out


def main():
    """Run step 10."""
    det = pd.read_parquet(OUT / "08_de_pipeline" / "de_pipeline.parquet")
    cat_all = apply(det)
    cat_all["SPLIT"] = det.split.values
    cat_all["IN_SNR_MASK"] = det.in_snr.values
    cat_all.to_parquet(OUTDIR / "de_catalog.parquet")

    test = cat_all[(cat_all.SPLIT == "test") & ~cat_all.IN_SNR_MASK].reset_index(drop=True)
    cat = catalog("test")
    validation = cat_all[(cat_all.SPLIT == "val") & ~cat_all.IN_SNR_MASK].reset_index(drop=True)
    metrics = {
        "detection": detection_metrics(test, cat),
        "validation_detection": detection_metrics(validation, catalog("val")),
    }

    # Positions and fluxes against counterparts.
    m = match_detections(test.rename(columns={"LII": "l", "BII": "b"}), cat)
    pt = m[(m.cat_EXT == 0) & m.cat_ML_FLUX_1.notna()].copy()
    pt["sep"] = sep_arcsec(pt.l, pt.b, pt.cat_LII, pt.cat_BII)
    tot_err = np.sqrt(pt.POS_ERR**2 + pt.cat_POS_ERR.fillna(0) ** 2 / 2)
    ratio = pt.sep / tot_err
    snr_grp = pd.cut(np.where(pt.SATURATED, 1e8, pt.SNR), SNR_BINS)
    metrics["positions"] = {
        "n": len(pt),
        "median_arcsec": float(pt.sep.median()),
        "p68_arcsec": float(pt.sep.quantile(0.68)),
        "p90_arcsec": float(pt.sep.quantile(0.9)),
        "by_snr": pt.groupby(snr_grp, observed=True)
        .sep.describe(percentiles=[0.5, 0.68, 0.9])[["count", "50%", "68%", "90%"]]
        .round(1)
        .to_dict(orient="index"),
        # For a 2D Gaussian with per-axis sigma, 39.3% lie within 1 sigma and 86.5% within 2 sigma.
        "frac_within_1sigma": float((ratio < 1).mean()),
        "frac_within_2sigma": float((ratio < 2).mean()),
        "frac_within_3sigma": float((ratio < 3).mean()),
    }
    metrics["positions"]["by_snr"] = {str(k): v for k, v in metrics["positions"]["by_snr"].items()}

    fl = {}
    good = pt.sep < 150
    for band in BANDS:
        y = np.log10(pt[f"cat_ML_FLUX_{band}"].where(pt[f"cat_ML_FLUX_{band}"] > 0))
        p = np.log10(pt[f"FLUX_{band}"])
        ok = good & y.notna()
        r = (p - y)[ok]
        inside = (
            (pt[f"FLUX_{band}_LO"] <= pt[f"cat_ML_FLUX_{band}"]) & (pt[f"cat_ML_FLUX_{band}"] <= pt[f"FLUX_{band}_HI"])
        )[ok]
        by_flux = (
            pd.DataFrame({"r": r, "f": y[ok]})
            .groupby(pd.cut(y[ok], [-15, -13, -12.5, -12, -11.5, -11, -8]), observed=True)
            .r
        )
        fl[band] = {
            "n": int(ok.sum()),
            "bias_dex": float(r.median()),
            "scatter_dex": float(1.4826 * (r - r.median()).abs().median()),
            "rms_dex": float(np.sqrt((r**2).mean())),
            "frac_outlier_0.3dex": float(((r - r.median()).abs() > 0.3).mean()),
            "coverage_16_84": float(inside.mean()),
            "scatter_saturated": float(1.4826 * (r[pt.SATURATED[ok]] - r[pt.SATURATED[ok]].median()).abs().median()),
            "scatter_unsaturated": float(
                1.4826 * (r[~pt.SATURATED[ok]] - r[~pt.SATURATED[ok]].median()).abs().median()
            ),
            "by_catalog_flux": {
                str(k): {
                    "n": int(len(v)),
                    "bias": float(v.median()),
                    "scatter": float(1.4826 * (v - v.median()).abs().median()),
                }
                for k, v in by_flux
                if len(v) > 5
            },
        }
    metrics["fluxes"] = fl
    save_json(metrics, OUTDIR / "metrics.json")

    plot_results(metrics, ratio, pt, good, fl)

    write_report(metrics)
    print(
        metrics["detection"]["F50"],
        metrics["detection"]["reliability"],
        metrics["positions"]["median_arcsec"],
        {b: (v["bias_dex"], v["scatter_dex"]) for b, v in fl.items()},
    )


def plot_results(metrics, ratio, pt, good, fl):
    """Render validation diagnostics from saved results and matched test sources."""
    # Figures.
    d = metrics["detection"]
    n = np.asarray(d["completeness"]["n"])
    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.1))
    ax.stairs(np.where(n > 0, np.asarray(d["completeness"]["comp"]), np.nan), FLUX_BINS, baseline=None)
    ax.axhline(0.5, c="k", lw=0.5)
    ax.axhline(0.9, c="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{\mathrm{eRASS1}}$ [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel("completeness (test, point sources)")
    f90 = f"F90={d['F90']:.2e}" if np.isfinite(d["F90"]) else "90% completeness not reached"
    ax.set_title(f"F50={d['F50']:.2e}\n{f90}")
    save_figure(fig, OUTDIR / "detection_positions.pdf")

    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.1))
    rs = dict(sorted(d["reliability_vs_snr"].items(), key=lambda item: float(item[0].strip("()[]").split(",")[0])))
    ax.stairs([v[0] for v in rs.values()], np.arange(len(rs) + 1) - 0.5, baseline=None)
    labels = []
    for interval in rs:
        lo, hi = (float(v) for v in interval.strip("()[]").split(","))
        labels.append(rf"$>{lo:g}$" if hi >= 1e8 else rf"$({lo:g}, {hi:g}]$")
    ax.set_xticks(range(len(rs)), labels, rotation=30)
    ax.set_xlabel(r"S/N bin")
    ax.set_ylabel("reliability")
    ax.set_title(f"overall reliability {d['reliability']:.3f} (N={d['n']})")
    save_figure(fig, OUTDIR / "reliability.pdf")

    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.1))
    ax.hist(ratio.clip(upper=6), bins=60, density=True, alpha=0.6, label=r"$\Delta r / \sigma_{\mathrm{combined}}$")
    x = np.linspace(0, 6, 200)
    ax.plot(x, x * np.exp(-(x**2) / 2), "r-", label="Rayleigh(1)")
    ax.legend()
    ax.set_xlabel(r"$\Delta r / \sigma_{\mathrm{combined}}$")
    ax.set_ylabel("Probability density")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "position_calibration.pdf")

    fig, axs = plt.subplots(2, 2, figsize=figure_size(2, 2))
    axs = axs.ravel()
    for ax, band in zip(axs, BANDS):
        y = np.log10(pt[f"cat_ML_FLUX_{band}"].where(pt[f"cat_ML_FLUX_{band}"] > 0))
        p = np.log10(pt[f"FLUX_{band}"])
        ok = good & y.notna()
        ax.scatter(y[ok], p[ok], s=3, c=pt.SATURATED[ok], cmap="coolwarm")
        ax.plot([-15, -9], [-15, -9], "k-")
        ax.set_xlim(-14.5, -9.5)
        ax.set_ylim(-14.5, -9.5)
        ax.set_xlabel(rf"$\log_{{10}} F_{{\mathrm{{eRASS1}},{band}}}$")
        ax.set_ylabel(rf"$\log_{{10}} F_{{\mathrm{{image}},{band}}}$")
        ax.set_title(f"bias {fl[band]['bias_dex']:+.3f}, scatter {fl[band]['scatter_dex']:.3f} dex")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "fluxes.pdf")


def write_report(mt):
    """Markdown validation report."""
    d, p, f = mt["detection"], mt["positions"], mt["fluxes"]
    lines = [
        "# Validation report: image-based catalogue, eRASS1-DE test split",
        "",
        "All numbers are from the test split (spatial blocks held out from downstream source-model tuning and",
        "model fitting), conditional on global image registration and fixed empirical halo safeguards.",
        "Sources inside known supernova remnants are excluded. Reference: eRASS1 Main",
        "v1.2 (counterpart = brightest source with F >= 1e-13 within 4 arcmin; chance matches corrected",
        "with positions displaced by 15 arcmin).",
        "",
        "## Detection",
        "",
        f"- detections: {d['n']}, chance-corrected reliability: {d['reliability']:.3f}",
        f"- 50% / 90% completeness flux (0.2-2.3 keV, point sources): {d['F50']:.2e} / {d['F90']:.2e} erg/s/cm2"
        f" (monotone completeness curve plateaus at {d['completeness_plateau']:.0%})",
        "- 50% completeness flux vs |b|: " + ", ".join(f"{k} deg: {v:.2e}" for k, v in d["F50_vs_absb"].items()),
        "",
        "| S/N bin | reliability | N |",
        "|---|---|---|",
    ]
    lines += [
        f"| {k.replace('100000000.0', 'saturated')} | {v[0]:.3f} | {v[1]} |" for k, v in d["reliability_vs_snr"].items()
    ]
    lines += [
        "",
        "## Positions",
        "",
        f"- matched point sources: {p['n']}",
        f"- separation median / 68% / 90%: {p['median_arcsec']:.0f} / {p['p68_arcsec']:.0f} / {p['p90_arcsec']:.0f} arcsec",
        f"- POS_ERR calibration (expected 39% / 86% / 99%): {p['frac_within_1sigma']:.0%} within 1 sigma, "
        f"{p['frac_within_2sigma']:.0%} within 2 sigma, {p['frac_within_3sigma']:.0%} within 3 sigma",
        "",
        "| S/N bin | N | median | 68% | 90% |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {k.replace('100000000.0', 'saturated')} | {v['count']:.0f} | {v['50%']:.0f} | {v['68%']:.0f} | {v['90%']:.0f} |"
        for k, v in p["by_snr"].items()
    ]
    lines += [
        "",
        "## Fluxes (log10 image / eRASS1)",
        "",
        "| band | N | bias [dex] | scatter [dex] | rms [dex] | >0.3 dex | 16-84% coverage | scatter sat / unsat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {b} | {v['n']} | {v['bias_dex']:+.3f} | {v['scatter_dex']:.3f} | {v['rms_dex']:.3f} | "
        f"{v['frac_outlier_0.3dex']:.1%} | {v['coverage_16_84']:.0%} | {v['scatter_saturated']:.3f} / {v['scatter_unsaturated']:.3f} |"
        for b, v in f.items()
    ]
    lines += ["", "Band 1 by eRASS1 flux (log bins):", "", "| log F bin | N | bias | scatter |", "|---|---|---|---|"]
    lines += [
        f"| {k} | {v['n']} | {v['bias']:+.3f} | {v['scatter']:.3f} |" for k, v in f["1"]["by_catalog_flux"].items()
    ]
    lines += ["", "Figures: `detection_positions.pdf`, `fluxes.pdf`.", ""]
    (OUTDIR / "validation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
