"""Step 12: external validation of the DE and RU image catalogues with XMMSL3 and the ROSAT BSC.

External sources: point-like, outside SNR masks, above EXT_FLUX_MIN (well above the image
detection limit, so that incompleteness does not bias the flux comparison). XMMSL3 detections of
the same source are merged (median flux). eRASS1 itself is compared with the same external
samples (DE half) as a reference for the scatter expected from source variability and band
conversions alone.

Per catalogue and half:
  * detection fraction of external sources (chance-corrected)
  * separations (median, 68%, 90%) and mean offset (dl cos b, db)
  * log flux ratio catalogue / external: median, robust scatter, bootstrap error of the median
The RU-DE difference of the median flux ratio (image catalogues) estimates a residual flux-scale
offset of the RU half.

High-energy catalogues (MAXI/GSC 4-10 keV, RXTE/XSS 3-20 keV, MAXI/SSC 0.7-7 keV): completeness of
the image catalogues (and of eRASS1 in the DE half, as reference) for their sources, using the
SIMBAD position of the identification within HARD_ID_R where available, otherwise the catalogue
position within 3 sigma (chance-corrected, capped at HARD_MAX_R). Split by half, |b| and flux.
MAXI/SSC soft-band fluxes (0.7-1.85 keV) are also compared with image P2+P3 (0.5-2 keV) fluxes.

Outputs (out/12_external/): external.json, hard.json, external_report.md, external.pdf, hard.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from epngcat.config import OUT, SEED, is_german, load_json, save_json, step_dir
from epngcat.evaluate import in_snr, shifted
from epngcat.plotting import figure_size, save_figure
from epngcat.sky import crossmatch

OUTDIR = step_dir("12_external")
EXT_FLUX_MIN = 2e-12
MATCH_IMG = 180.0  # arcsec, image catalogues
MATCH_ERASS = 60.0  # arcsec, eRASS1
EXT_BANDS = {"xmmsl3": "0.2-2 keV", "rass_bsc": "0.1-2.4 keV"}
HARD_ID_R = 4.0  # arcmin
HARD_MAX_R = 30.0  # arcmin
HARD = {  # name: (flux column used for binning, label)
    "maxi_gsc": ("flux_4_10", "MAXI/GSC 4-10 keV flux"),
    "rxte_xss": ("count_rate_1", "RXTE/XSS 3-8 keV rate"),
    "maxi_ssc": ("sb_flux", "MAXI/SSC 0.7-1.85 keV flux"),
}


def hard_match(h, lon, lat):
    """Match flags of a hard catalogue against positions (lon, lat): (matched, chance, used_id, index)."""
    has_id = h.l_id.notna().values
    ln = np.where(has_id, h.l_id, h.l)
    bn = np.where(has_id, h.b_id, h.b)
    rad = np.where(has_id, HARD_ID_R, np.minimum(3 * h.pos_err_arcmin.values, HARD_MAX_R))
    matched, chance = np.zeros(len(h), bool), np.zeros(len(h), bool)
    idx = np.full(len(h), -1)
    ls, bs = shifted(ln, bn, 60.0)
    for r in np.unique(rad):
        sel = rad == r
        i, _ = crossmatch(ln[sel], bn[sel], lon, lat, r * 60)
        matched[sel], idx[sel] = i >= 0, i
        ic, _ = crossmatch(ls[sel], bs[sel], lon, lat, r * 60)
        chance[sel] = ic >= 0
    return matched, chance, has_id, idx


def completeness_stats(matched, chance, sel):
    """Chance-corrected completeness of a subset."""
    if sel.sum() == 0:
        return {"n": 0}
    f, pc = matched[sel].mean(), chance[sel].mean()
    return {
        "n": int(sel.sum()),
        "raw": float(f),
        "chance": float(pc),
        "completeness": float((f - pc) / (1 - pc)) if pc < 1 else float("nan"),
    }


def hard_catalogs(img, er):
    """Completeness of image catalogues (and eRASS1) for high-energy catalogue sources."""
    res, curves = {}, {}
    for name, (fcol, _) in HARD.items():
        h = pd.read_parquet(OUT / "01_catalogs" / f"{name}.parquet")
        pos_l = h.l_id.fillna(h.l).values
        pos_b = h.b_id.fillna(h.b).values
        h = h[~in_snr(pos_l, pos_b) & (np.abs(((pos_l + 90) % 180) - 90) > 0.5)].reset_index(drop=True)
        pos_l = h.l_id.fillna(h.l).values
        pos_b = h.b_id.fillna(h.b).values
        half = np.where(is_german(pos_l), "DE", "RU")
        m_img, c_img, has_id, idx = hard_match(h, img.LII.values, img.BII.values)
        m_er, c_er, _, _ = hard_match(h, er.LII.values, er.BII.values)
        flux = h[fcol].values.astype(float)
        edges = np.nanpercentile(flux[flux > 0], [0, 25, 50, 75, 100])
        r = {}
        for hf in ("DE", "RU"):
            base = half == hf
            r[f"image_{hf}"] = {
                "all": completeness_stats(m_img, c_img, base),
                "identified": completeness_stats(m_img, c_img, base & has_id),
                "abs_b_gt10": completeness_stats(m_img, c_img, base & (np.abs(pos_b) > 10)),
                "abs_b_lt10": completeness_stats(m_img, c_img, base & (np.abs(pos_b) <= 10)),
                "by_flux_quartile": [
                    completeness_stats(m_img, c_img, base & (flux >= lo) & (flux <= hi))
                    for lo, hi in zip(edges[:-1], edges[1:])
                ],
            }
        de = half == "DE"
        r["erass1_DE"] = {
            "all": completeness_stats(m_er, c_er, de),
            "identified": completeness_stats(m_er, c_er, de & has_id),
        }
        r["flux_quartile_edges"] = edges.tolist()
        missed = h[~m_img & has_id]
        r["missed_identified"] = [
            {"name": str(a), "id": str(b), "l": float(c), "b": float(d)}
            for a, b, c, d in zip(missed.name, missed.id_name, missed.l_id, missed.b_id)
        ][:40]
        if name == "maxi_ssc":  # soft band overlaps eROSITA: compare fluxes (identified matches only)
            ok = m_img & has_id & (flux > 0)
            j = idx[ok]
            ratio = np.log10(img.FLUX_P2.values[j] + img.FLUX_P3.values[j]) - np.log10(flux[ok])
            sat = img.SATURATED.values[j]
            hh = half[ok]
            r["flux_comparison"] = {}
            for hf in ("DE", "RU"):
                v = ratio[hh == hf]
                if len(v) > 3:
                    r["flux_comparison"][f"image_{hf}"] = {
                        "n": int(len(v)),
                        "median_dex": float(np.median(v)),
                        "scatter_dex": float(1.4826 * np.median(np.abs(v - np.median(v)))),
                        "frac_saturated": float(sat[hh == hf].mean()),
                    }
            m2, _, _, j2 = hard_match(h, er.LII.values, er.BII.values)
            ok2 = m2 & has_id & (flux > 0) & de
            v = np.log10(er.ML_FLUX_P2.values[j2[ok2]] + er.ML_FLUX_P3.values[j2[ok2]]) - np.log10(flux[ok2])
            v = v[np.isfinite(v)]
            r["flux_comparison"]["erass1_DE"] = {
                "n": int(len(v)),
                "median_dex": float(np.median(v)),
                "scatter_dex": float(1.4826 * np.median(np.abs(v - np.median(v)))),
            }
            curves[name + "_flux"] = (flux[ok], ratio, hh)
        res[name] = r
        curves[name] = (flux, m_img, half, has_id)
    return res, curves


def external(name):
    """Clean external sample with l, b, flux."""
    if name == "xmmsl3":
        x = pd.read_parquet(OUT / "01_catalogs" / "xmmsl3.parquet")
        x = x[x.EXT_B6 == 0]
        g = x.groupby("UNIQUE_SRCNAME")
        df = pd.DataFrame({"l": g.LII.median(), "b": g.BII.median(), "flux": g.FLUX_B6.median(), "n_obs": g.size()})
    else:
        x = pd.read_parquet(OUT / "01_catalogs" / "rass_bsc.parquet")
        x = x[x.Extent == 0]
        c = SkyCoord(x.RA.values, x.DEC.values, unit="deg").galactic
        df = pd.DataFrame({"l": c.l.deg, "b": c.b.deg, "flux": x.FLUX.values, "n_obs": 1})
    df = df[(df.flux > EXT_FLUX_MIN) & ~in_snr(df.l.values, df.b.values)].reset_index(drop=True)
    # Sources in the thin DE/RU footprint sliver or in the cropped image edge are ambiguous.
    df = df[np.abs(((df.l + 90) % 180) - 90) > 0.5]
    df["half"] = np.where(is_german(df.l), "DE", "RU")
    return df.reset_index(drop=True)


def compare(ext, cat, lcol, bcol, fcol, radius):
    """Match stats of one catalogue against an external sample."""
    idx, sep = crossmatch(ext.l.values, ext.b.values, cat[lcol].values, cat[bcol].values, radius)
    _, sep_c = crossmatch(*shifted(ext.l.values, ext.b.values, 30.0), cat[lcol].values, cat[bcol].values, radius)
    ok = idx >= 0
    f, pc = ok.mean(), (sep_c < np.inf).mean()
    out = {
        "n_ext": len(ext),
        "match_frac": float(f),
        "chance_frac": float(pc),
        "detected_frac": float((f - pc) / (1 - pc)) if pc < 1 else np.nan,
    }
    if ok.sum() < 5:
        return out
    m = cat.iloc[idx[ok]]
    s = sep[ok]
    dl = ((m[lcol].values - ext.l.values[ok] + 180) % 360 - 180) * np.cos(np.radians(ext.b.values[ok])) * 3600
    db = (m[bcol].values - ext.b.values[ok]) * 3600
    r = np.log10(m[fcol].values) - np.log10(ext.flux.values[ok])
    r = r[np.isfinite(r)]
    boot = np.median(np.random.default_rng(SEED).choice(r, (1000, len(r))), axis=1)
    out.update(
        {
            "n_match": int(ok.sum()),
            "sep_median": float(np.median(s)),
            "sep_p68": float(np.percentile(s, 68)),
            "sep_p90": float(np.percentile(s, 90)),
            "offset_dl_cosb": float(np.median(dl)),
            "offset_db": float(np.median(db)),
            "flux_ratio_median_dex": float(np.median(r)),
            "flux_ratio_median_err": float(boot.std()),
            "flux_ratio_scatter_dex": float(1.4826 * np.median(np.abs(r - np.median(r)))),
        }
    )
    return out


def main():
    """Run step 12."""
    de = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    ru = pd.read_parquet(OUT / "11_catalog_ru" / "ru_catalog.parquet")
    img = pd.concat([de, ru], ignore_index=True)
    img = img[~img.IN_SNR_MASK].reset_index(drop=True)
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)].reset_index(drop=True)

    res, pairs = {}, {}
    for name in EXT_BANDS:
        ext = external(name)
        res[name] = {}
        for half in ("DE", "RU"):
            e = ext[ext.half == half].reset_index(drop=True)
            res[name][f"image_{half}"] = compare(e, img, "LII", "BII", "FLUX_1", MATCH_IMG)
            idx, _ = crossmatch(e.l.values, e.b.values, img.LII.values, img.BII.values, MATCH_IMG)
            pairs[(name, half)] = (e.flux.values[idx >= 0], img.FLUX_1.values[idx[idx >= 0]])
        e = ext[ext.half == "DE"].reset_index(drop=True)
        res[name]["erass1_DE"] = compare(e, er, "LII", "BII", "ML_FLUX_1", MATCH_ERASS)
        idx, _ = crossmatch(e.l.values, e.b.values, er.LII.values, er.BII.values, MATCH_ERASS)
        pairs[(name, "eRASS1")] = (e.flux.values[idx >= 0], er.ML_FLUX_1.values[idx[idx >= 0]])
        a, b = res[name]["image_RU"], res[name]["image_DE"]
        res[name]["ru_minus_de_flux_offset_dex"] = a["flux_ratio_median_dex"] - b["flux_ratio_median_dex"]
        res[name]["ru_minus_de_flux_offset_err"] = float(
            np.hypot(a["flux_ratio_median_err"], b["flux_ratio_median_err"])
        )
        # Scatter attributable to the image pipeline once variability/band conversion is removed.
        res[name]["image_excess_scatter_DE"] = float(
            np.sqrt(max(b["flux_ratio_scatter_dex"] ** 2 - res[name]["erass1_DE"]["flux_ratio_scatter_dex"] ** 2, 0))
        )
    offs = np.array([res[n]["ru_minus_de_flux_offset_dex"] for n in EXT_BANDS])
    errs = np.array([res[n]["ru_minus_de_flux_offset_err"] for n in EXT_BANDS])
    w = 1 / errs**2
    comb = float((offs * w).sum() / w.sum())
    comb_err = float(1 / np.sqrt(w.sum()))
    res["ru_minus_de_flux_offset_combined"] = {"dex": comb, "err": comb_err, "significance": abs(comb) / comb_err}
    save_json(res, OUTDIR / "external.json")
    print(res)
    hard, curves = hard_catalogs(img, er)
    save_json(hard, OUTDIR / "hard.json")
    print(hard)
    plot_hard(curves)

    plot_results(pairs)
    write_report(res)


def plot_results(pairs):
    """Render the pre-correction external flux comparison from matched pairs."""
    fig, axs = plt.subplots(1, 2, figsize=figure_size(1, 2))
    for ax, name in zip(axs, EXT_BANDS):
        for key, col in [("eRASS1", "k"), ("DE", "C0"), ("RU", "C3")]:
            fe, fc = pairs[(name, key)]
            ax.scatter(
                fe,
                fc,
                marker=".",
                s=4,
                linewidths=0,
                c=col,
                alpha=0.5,
                label=f"{'eRASS1 (DE)' if key == 'eRASS1' else 'image ' + key} N={len(fe)}",
            )
        ax.plot([1e-13, 1e-9], [1e-13, 1e-9], "k-", lw=0.8)
        ax.set(xscale="log", yscale="log")
        label = "XMMSL3" if name == "xmmsl3" else "BSC"
        units = r"$[\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$"
        ax.set_xlabel(rf"$F_{{\mathrm{{{label}}}}}$ ({EXT_BANDS[name]})" + "\n" + units)
        ax.set_ylabel(r"$F_{\mathrm{cat}}$ (0.2–2.3 keV)" + "\n" + units)
        ax.legend(markerscale=3)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "external.pdf")


def replot():
    """Recreate pre-correction pairs from saved products without recalibration."""
    de = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    ru = pd.read_parquet(OUT / "11_catalog_ru" / "ru_catalog.parquet")
    image = pd.concat([de, ru], ignore_index=True)
    image = image[~image.IN_SNR_MASK]
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)]
    pairs = {}
    for name in EXT_BANDS:
        ext = external(name)
        for key, half, cat, fcol, radius in (
            ("DE", "DE", image, "FLUX_1", MATCH_IMG),
            ("RU", "RU", image, "FLUX_1", MATCH_IMG),
            ("eRASS1", "DE", er, "ML_FLUX_1", MATCH_ERASS),
        ):
            ref = ext[ext.half == half]
            idx, _ = crossmatch(ref.l.values, ref.b.values, cat.LII.values, cat.BII.values, radius)
            pairs[(name, key)] = (ref.flux.values[idx >= 0], cat[fcol].values[idx[idx >= 0]])
    plot_results(pairs)


def plot_hard(curves):
    """Plot detection fraction vs high-energy flux and the SSC flux comparison."""
    fig, axs = plt.subplots(2, 2, figsize=figure_size(2, 2))
    axs = axs.ravel()
    for ax, (name, (fcol, label)) in zip(axs, HARD.items()):
        flux, m, half, has_id = curves[name]
        good = flux > 0
        bins = np.geomspace(np.nanmin(flux[good]), np.nanmax(flux[good]) * 1.001, 9)
        for hf, col in (("DE", "C0"), ("RU", "C3")):
            sel = good & (half == hf) & has_id
            k = np.digitize(flux[sel], bins)
            frac = [m[sel][k == i].mean() if (k == i).sum() > 2 else np.nan for i in range(1, len(bins))]
            ax.stairs(frac, bins, baseline=None, color=col, label=f"image {hf} (identified, N={sel.sum()})")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(label)
        ax.set_ylabel("detected fraction")
        ax.set_title(name)
        ax.legend()
    f, r, hh = curves["maxi_ssc_flux"]
    for hf, col in (("DE", "C0"), ("RU", "C3")):
        axs[3].scatter(f[hh == hf], r[hh == hf], s=12, c=col, label=f"image {hf}")
    axs[3].set_xscale("log")
    axs[3].set_xlabel("MAXI/SSC 0.7-1.85 keV flux")
    axs[3].set_ylabel(r"$\log_{10}(F_{\mathrm{image}} / F_{\mathrm{SSC}})$")
    axs[3].axhline(0, c="k", lw=0.5)
    axs[3].legend()
    fig.tight_layout()
    save_figure(fig, OUTDIR / "hard.pdf")


def write_report(res):
    """Markdown summary."""
    lines = [
        "# External validation (XMMSL3, ROSAT BSC)",
        "",
        f"External point sources with F > {EXT_FLUX_MIN:.0e} erg/s/cm2 outside SNR masks. Image catalogues",
        f'matched within {MATCH_IMG:.0f}", eRASS1 within {MATCH_ERASS:.0f}" (reference for intrinsic variability and',
        "band-conversion scatter). Flux ratio = log10(catalogue 0.2-2.3 keV / external band flux); a",
        "constant offset is expected from the different bands, what matters is DE vs RU and the scatter.",
        "",
    ]
    for name in EXT_BANDS:
        lines += [
            f"## {name} ({EXT_BANDS[name]})",
            "",
            "| catalogue | N ext | detected | median sep | 90% sep | offset dl cosb / db | log ratio | scatter |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key in ("erass1_DE", "image_DE", "image_RU"):
            v = res[name][key]
            lines.append(
                f'| {key} | {v["n_ext"]} | {v["detected_frac"]:.0%} | {v.get("sep_median", np.nan):.0f}" | '
                f'{v.get("sep_p90", np.nan):.0f}" | {v.get("offset_dl_cosb", np.nan):+.0f}" / {v.get("offset_db", np.nan):+.0f}" | '
                f"{v.get('flux_ratio_median_dex', np.nan):+.3f} +- {v.get('flux_ratio_median_err', np.nan):.3f} | "
                f"{v.get('flux_ratio_scatter_dex', np.nan):.3f} |"
            )
        lines += [
            "",
            f"- RU - DE flux offset: {res[name]['ru_minus_de_flux_offset_dex']:+.3f} +- "
            f"{res[name]['ru_minus_de_flux_offset_err']:.3f} dex",
            f"- image-pipeline scatter beyond eRASS1's own (DE, quadrature): {res[name]['image_excess_scatter_DE']:.3f} dex",
            "",
        ]
    c = res["ru_minus_de_flux_offset_combined"]
    lines += [
        "## Combined",
        "",
        f"RU - DE flux-scale offset: {c['dex']:+.3f} +- {c['err']:.3f} dex ({c['significance']:.1f} sigma)",
        "",
    ]
    hard = load_json(OUTDIR / "hard.json")
    lines += [
        "## High-energy catalogues (completeness)",
        "",
        f"Identified sources: SIMBAD position of the identification, match within {HARD_ID_R:.0f}'; others: catalogue",
        f"position within 3 sigma (max {HARD_MAX_R:.0f}'), chance-corrected with positions displaced by 1 deg.",
        "",
        "| catalogue | half | N | completeness (all) | identified | abs(b)>10 | abs(b)<10 | by flux quartile (low -> high) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in HARD:
        for hf in ("DE", "RU"):
            v = hard[name][f"image_{hf}"]
            q = " / ".join(f"{x.get('completeness', np.nan):.0%}" for x in v["by_flux_quartile"])
            lines.append(
                f"| {name} | {hf} | {v['all']['n']} | {v['all'].get('completeness', np.nan):.0%} | "
                f"{v['identified'].get('completeness', np.nan):.0%} | {v['abs_b_gt10'].get('completeness', np.nan):.0%} | "
                f"{v['abs_b_lt10'].get('completeness', np.nan):.0%} | {q} |"
            )
        e = hard[name]["erass1_DE"]
        lines.append(
            f"| {name} | eRASS1 (DE) | {e['all']['n']} | {e['all'].get('completeness', np.nan):.0%} | "
            f"{e['identified'].get('completeness', np.nan):.0%} | | | |"
        )
    fc = hard["maxi_ssc"]["flux_comparison"]
    lines += [
        "",
        "MAXI/SSC soft-band flux comparison, log10(F(0.5-2 keV) / F_SSC(0.7-1.85 keV)):",
        "",
        "| catalogue | N | median | scatter |",
        "|---|---|---|---|",
    ]
    lines += [f"| {k} | {v['n']} | {v['median_dex']:+.3f} | {v['scatter_dex']:.3f} |" for k, v in fc.items()]
    lines += ["", "Figures: `external.pdf`, `hard.pdf`; lists of missed identified sources in `hard.json`.", ""]
    (OUTDIR / "external_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
