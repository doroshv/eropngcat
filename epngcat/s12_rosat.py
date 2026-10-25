"""Step 12 external validation: final all-sky catalogue against the ROSAT 2RXS catalogue.

2RXS (Boller+ 2016, J/A+A/588/A103) is the second processing of the ROSAT all-sky survey: ~135000
sources over the whole sky in 0.1-2.4 keV, reaching ~1e-13 erg/s/cm2. Two properties make it the
right external check for the *final* catalogue, which step 12's ROSAT comparison (the 1RXS bright
source catalogue, 18811 sources above ~2e-12) cannot provide:

  * it covers both halves uniformly, unlike eRASS1-DE, so DE and RU are measured against the same
    yardstick and any RU-DE difference is in the image catalogue, not in the reference;
  * it reaches the image catalogue's own completeness limit, so the completeness curve is resolved
    instead of saturating at 100% as it does for the bright source catalogue.

It is not a flux calibrator: ROSAT and eROSITA differ in band (0.1-2.4 vs 0.2-2.3 keV) and epoch
(1990/91 vs 2019/20), so the absolute flux ratio carries a band term and ~30 years of variability.
Only differences between catalogues measured on the *same* 2RXS sources are meaningful, which is
why image DE - image RU (RU flux scale) and image DE - eRASS1 DE (image calibration) are the
numbers quoted, not the ratio itself.

Count rates are converted with the HR1-dependent ECF of Schmitt+ (1995), the same conversion step
01 applies to the bright source catalogue (`rass_flux`); the 2RXS spectral-fit fluxes exist only
for the bright end and are not used.

2RXS is cut to ExiML >= EXI_MIN (the catalogue's own 6.5 limit leaves ~30% spurious sources,
which would depress the completeness denominator) and to screening flag Sflag = 0. Sources inside
SNR masks, within 0.5 deg of the image joins, and extended ones (for positions and fluxes) are
excluded, as in the other step-12 external checks.

Outputs (out/12_external/): rosat_2rxs.json, rosat_2rxs_report.md, rosat_2rxs.pdf,
rosat_2rxs_flux.pdf, rosat_2rxs_ratio.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epngcat.config import DATA, OUT, SEED, is_german, save_json, step_dir
from epngcat.evaluate import in_snr, shifted
from epngcat.plotting import figure_size, save_figure
from epngcat.s01_prepare_catalogs import rass_flux
from epngcat.sky import crossmatch

OUTDIR = step_dir("12_external")
VIZIER = "J/A+A/588/A103"
COLS = ["2RXS", "GLON", "GLAT", "ExiML", "CRate", "e_CRate", "ExpTime", "Ext", "ExtML", "HR1", "Sflag"]
EXI_MIN = 10.0  # Boller+ 2016: ~30% spurious above the catalogue limit of 6.5, a few % above 10
RADIUS = 120.0  # arcsec; set by the 2RXS position error, so the same for every catalogue compared
SHIFT = 30.0  # arcmin, for the chance-match rate
FLUX_EDGES = np.array([0, 3e-13, 1e-12, 3e-12, 1e-11, np.inf])
FLUX_BRIGHT = 3e-12  # both halves are ~85% complete above this, so flux ratios there are unbiased


def load(exi_min=EXI_MIN, *, geometry=True):
    """2RXS clean sample with Galactic position, 0.1-2.4 keV flux and half (cached in data/).

    exi_min defaults to the clean cut used here; step 17 passes the catalogue's own 6.5 limit,
    because for a counterpart search even a marginal 2RXS detection counts.
    Number counts pass geometry=False and apply their common sky mask separately.
    """
    cache = DATA / "rass_2rxs.parquet"
    if not cache.exists():
        from astroquery.vizier import Vizier

        DATA.mkdir(exist_ok=True)
        t = Vizier(columns=COLS, row_limit=-1).get_catalogs(VIZIER)[0]
        t.to_pandas()[COLS].to_parquet(cache)
    df = pd.read_parquet(cache)
    hr1 = df.HR1.values
    out = pd.DataFrame(
        {
            "name": df["2RXS"].astype(str).str.strip().values,
            "l": df.GLON.values,
            "b": df.GLAT.values,
            "exi": df.ExiML.values,
            "rate": df.CRate.values,
            # HR1 is undefined for sources with no counts in one band; 0 is the middle of the ECF
            # range (8.3 of 5.7-13.6 e-12), and such sources are faint, so the effect is small.
            "flux": rass_flux(df.CRate.values, np.where(np.isfinite(hr1), hr1, 0.0)),
            "hr1_missing": ~np.isfinite(hr1),
            "extended": (df.Ext.fillna(0).values > 0) & (df.ExtML.fillna(0).values > 10),
        }
    )
    clean = (out.exi >= exi_min) & (df.Sflag.fillna(0).values == 0) & np.isfinite(out.flux) & (out.flux > 0)
    out = out[clean.values].reset_index(drop=True)
    if geometry:
        out = out[~in_snr(out.l.values, out.b.values) & (np.abs(((out.l + 90) % 180) - 90) > 0.5)]
    out = out.reset_index(drop=True)
    out["half"] = np.where(is_german(out.l), "DE", "RU")
    return out


def match(lon1, lat1, lon2, lat2):
    """Nearest match of each set-1 position in set 2, plus the chance-match flags."""
    idx, sep = crossmatch(lon1, lat1, lon2, lat2, RADIUS)
    ic, _ = crossmatch(*shifted(lon1, lat1, SHIFT), lon2, lat2, RADIUS)
    return idx, sep, ic >= 0


def completeness(matched, chance, sel):
    """Chance-corrected matched fraction of a subset."""
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    f, pc = matched[sel].mean(), chance[sel].mean()
    return {"n": n, "completeness": float((f - pc) / (1 - pc)), "raw": float(f), "chance": float(pc)}


def boot_median(v, n=2000):
    """Median, bootstrap standard error and robust scatter."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"n": int(len(v))}
    b = np.median(np.random.default_rng(SEED).choice(v, (n, len(v))), axis=1)
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "err": float(b.std()),
        "scatter": float(1.4826 * np.median(np.abs(v - np.median(v)))),
    }


def analyse(rx, cats):
    """Completeness, positions and flux ratios of each catalogue against 2RXS."""
    res, extra = {}, {}
    for key, (cat, lcol, bcol, fcol) in cats.items():
        r = rx[rx.half == key.split("_")[1]].reset_index(drop=True)
        idx, sep, chance = match(r.l.values, r.b.values, cat[lcol].values, cat[bcol].values)
        ok = idx >= 0
        out = {
            "completeness_all": completeness(ok, chance, np.ones(len(r), bool)),
            "completeness_point": completeness(ok, chance, ~r.extended.values),
            "completeness_vs_flux": {
                f"{lo:.0e}-{hi:.0e}": completeness(ok, chance, (r.flux.values >= lo) & (r.flux.values < hi))
                for lo, hi in zip(FLUX_EDGES[:-1], FLUX_EDGES[1:])
            },
            "completeness_abs_b_lt10": completeness(ok, chance, np.abs(r.b.values) < 10),
            "completeness_abs_b_gt10": completeness(ok, chance, np.abs(r.b.values) >= 10),
        }
        pt = ok & ~r.extended.values
        m = cat.iloc[idx[pt]]
        dl = ((m[lcol].values - r.l.values[pt] + 180) % 360 - 180) * np.cos(np.radians(r.b.values[pt])) * 3600
        db = (m[bcol].values - r.b.values[pt]) * 3600
        out["positions"] = {
            "n": int(pt.sum()),
            "median": float(np.median(sep[pt])),
            "p68": float(np.percentile(sep[pt], 68)),
            "p90": float(np.percentile(sep[pt], 90)),
            "offset_dl_cosb": boot_median(dl),
            "offset_db": boot_median(db),
        }
        ratio = np.log10(m[fcol].values) - np.log10(r.flux.values[pt])
        out["flux_ratio"] = boot_median(ratio)
        out["flux_ratio_bright"] = boot_median(ratio[r.flux.values[pt] > FLUX_BRIGHT])
        res[key] = out
        extra[key] = {
            "flux": r.flux.values,
            "ok": ok,
            "chance": chance,
            "sep": sep[pt],
            "rx_flux": r.flux.values[pt],
            "cat_flux": m[fcol].values,
            "ratio": ratio,
            "names": r.name.values[pt],
            "cat_idx": idx[pt],
        }

    # RU flux scale. The two halves match different fractions of 2RXS, so their matched samples
    # have different flux distributions and the all-source difference carries a selection term;
    # above FLUX_BRIGHT both halves are ~85% complete, so that version is the one to quote.
    for k in ("flux_ratio", "flux_ratio_bright"):
        u, v = res["image_RU"][k], res["image_DE"][k]
        res["image_RU_minus_image_DE" + k[10:]] = {
            "dex": u["median"] - v["median"],
            "err": float(np.hypot(u["err"], v["err"])),
        }
    # Image vs eRASS1 on the *same* 2RXS sources - eRASS1 matches nearly all of them and the image
    # only the brighter third, so this pairing is the only unbiased way to compare the two.
    e, i = extra["erass1_DE"], extra["image_DE"]
    _, ie, ii = np.intersect1d(e["names"], i["names"], return_indices=True)
    res["image_minus_erass1_DE_paired"] = boot_median(i["ratio"][ii] - e["ratio"][ie])
    split = cats["image_DE"][0].SPLIT.values[i["cat_idx"][ii]]
    res["image_minus_erass1_DE_paired_test"] = boot_median((i["ratio"][ii] - e["ratio"][ie])[split == "test"])
    return res, extra


def reverse(img, rx):
    """Fraction of image sources with a 2RXS counterpart, per half (chance-corrected)."""
    out = {}
    for half in ("DE", "RU"):
        c = img[half == img.HALF]
        r = rx[rx.half == half]
        idx, _, chance = match(c.LII.values, c.BII.values, r.l.values, r.b.values)
        out[half] = completeness(idx >= 0, chance, np.ones(len(c), bool))
        out[half]["n_2rxs"] = int(len(r))
    return out


def main():
    """Run the ROSAT 2RXS part of step 12 external validation."""
    img = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    img = img[(img.SNR_NAME.isna() | (img.SNR_NAME == "")) & np.isfinite(img.FLUX_1)].reset_index(drop=True)
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)].reset_index(drop=True)
    rx = load()
    print(f"2RXS clean: {len(rx)} (ExiML >= {EXI_MIN:.0f}), {rx.half.value_counts().to_dict()}")

    cats = {
        "image_DE": (img, "LII", "BII", "FLUX_1"),
        "image_RU": (img, "LII", "BII", "FLUX_1"),
        "erass1_DE": (er, "LII", "BII", "ML_FLUX_1"),
    }
    res, extra = analyse(rx, cats)
    res["image_with_2rxs_counterpart"] = reverse(img, rx)
    res["n_2rxs"] = {"all": len(rx), **rx.half.value_counts().to_dict()}
    save_json(res, OUTDIR / "rosat_2rxs.json")
    for k in cats:
        print(f"  {k:10s} completeness {res[k]['completeness_point']['completeness']:.1%}", res[k]["flux_ratio"])
    print("  RU - DE (bright):", res["image_RU_minus_image_DE_bright"])
    print("  image - eRASS1 (paired):", res["image_minus_erass1_DE_paired"])
    plot(extra)
    write_report(res)


def plot(extra):
    """Completeness vs 2RXS flux, flux-flux and flux ratio vs 2RXS flux."""
    colors = {"image_DE": "C0", "image_RU": "C3", "erass1_DE": "k"}
    panels = [plt.subplots(figsize=figure_size()) for _ in range(3)]
    axs = [ax for _, ax in panels]
    for key, e in reversed(extra.items()):  # eRASS1 first: it has most points, keep it underneath
        comp = []
        for lo, hi in zip(FLUX_EDGES[:-1], FLUX_EDGES[1:]):
            sel = (e["flux"] >= lo) & (e["flux"] < hi)
            if sel.sum() > 5:
                pc = e["chance"][sel].mean()
                comp.append((e["ok"][sel].mean() - pc) / (1 - pc))
            else:
                comp.append(np.nan)
        axs[0].stairs(
            comp,
            np.clip(FLUX_EDGES, 1e-13, 1e-10),
            baseline=None,
            color=colors[key],
            label=f"{key.replace('_', ' ')} (N={len(e['flux'])})",
        )
        axs[1].scatter(
            e["rx_flux"], e["cat_flux"], marker=".", s=3, c=colors[key], alpha=0.3, lw=0, label=key.replace("_", " ")
        )
        k = np.digitize(e["rx_flux"], FLUX_EDGES)
        axs[2].scatter(e["rx_flux"], e["ratio"], marker=".", s=3, c=colors[key], alpha=0.25, lw=0)
        medians = [np.median(e["ratio"][k == i]) if (k == i).sum() >= 10 else np.nan for i in range(1, len(FLUX_EDGES))]
        axs[2].stairs(medians, np.clip(FLUX_EDGES, 1e-13, 1e-10), baseline=None, color=colors[key])
    grid = np.array([1e-13, 1e-9])
    axs[1].plot(grid, grid, "k-", lw=1)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_ylabel("completeness for 2RXS sources (chance-corrected)")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$F_{\mathrm{cat}}$ (0.2–2.3 keV) [erg s$^{-1}$ cm$^{-2}$]")
    axs[1].legend(markerscale=4)
    axs[2].axhline(0, c="k", lw=1)
    axs[2].set_ylim(-1.5, 1.5)
    axs[2].set_ylabel(r"$\log_{10}(F_{\mathrm{cat}} / F_{\mathrm{2RXS}})$")
    axs[2].set_title("Median residuals per flux bin")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$F_{\mathrm{2RXS}}$ (0.1–2.4 keV) [erg s$^{-1}$ cm$^{-2}$]")
        ax.grid(False)
    axs[0].legend()
    for (fig, _), name in zip(panels, ("rosat_2rxs", "rosat_2rxs_flux", "rosat_2rxs_ratio")):
        save_figure(fig, OUTDIR / f"{name}.pdf")


def replot():
    """Reconstruct plot arrays from cached inputs, without bootstrap analysis."""
    img = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    img = img[(img.SNR_NAME.isna() | (img.SNR_NAME == "")) & np.isfinite(img.FLUX_1)]
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)]
    rx = load()
    extra = {}
    for key, cat, fcol in (("image_DE", img, "FLUX_1"), ("image_RU", img, "FLUX_1"), ("erass1_DE", er, "ML_FLUX_1")):
        ref = rx[rx.half == key.split("_")[1]]
        idx, sep, chance = match(ref.l.values, ref.b.values, cat.LII.values, cat.BII.values)
        ok = idx >= 0
        pt = ok & ~ref.extended.values
        flux = cat[fcol].values[idx[pt]]
        extra[key] = {
            "flux": ref.flux.values,
            "ok": ok,
            "chance": chance,
            "rx_flux": ref.flux.values[pt],
            "cat_flux": flux,
            "ratio": np.log10(flux / ref.flux.values[pt]),
        }
    plot(extra)


def fmt(c):
    """Completeness table cell."""
    return f"{c['completeness']:.0%} ({c['n']})" if c.get("n") else "-"


def write_report(res):
    """Markdown summary."""
    keys = ["image_DE", "image_RU", "erass1_DE"]
    L = [
        "# Validation against the ROSAT 2RXS all-sky catalogue (0.1-2.4 keV)",
        "",
        f"Boller+ 2016 (VizieR {VIZIER}), ExiML >= {EXI_MIN:.0f} and screening flag 0: {res['n_2rxs']['all']} sources",
        f"(DE {res['n_2rxs'].get('DE', 0)}, RU {res['n_2rxs'].get('RU', 0)}), outside SNR masks and the image joins.",
        f'Matches within {RADIUS:.0f}" (set by the 2RXS position error, the dominant term, hence the same radius',
        f"for all three catalogues); chance rate measured with positions displaced by {SHIFT:.0f}' and divided out.",
        "Count rates converted to flux with the HR1-dependent ECF of Schmitt+ (1995).",
        "",
        "## Completeness for 2RXS sources (N in brackets)",
        "",
        "| subset | " + " | ".join(keys) + " |",
        "|---|" + "---|" * len(keys),
        "| all | " + " | ".join(fmt(res[k]["completeness_all"]) for k in keys) + " |",
        "| point-like | " + " | ".join(fmt(res[k]["completeness_point"]) for k in keys) + " |",
        "| abs(b) < 10 | " + " | ".join(fmt(res[k]["completeness_abs_b_lt10"]) for k in keys) + " |",
        "| abs(b) >= 10 | " + " | ".join(fmt(res[k]["completeness_abs_b_gt10"]) for k in keys) + " |",
    ]
    for b in res["image_DE"]["completeness_vs_flux"]:
        L.append(f"| F(0.1-2.4) {b} | " + " | ".join(fmt(res[k]["completeness_vs_flux"][b]) for k in keys) + " |")
    L += [
        "",
        f'Chance-match rate at {RADIUS:.0f}": '
        + ", ".join(f"{k} {res[k]['completeness_all']['chance']:.1%}" for k in keys)
        + ".",
        "",
        "## Positions (point-like matches)",
        "",
        "| catalogue | N | median | 68% | 90% | dl cos b | db |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        p = res[k]["positions"]
        L.append(
            f'| {k} | {p["n"]} | {p["median"]:.0f}" | {p["p68"]:.0f}" | {p["p90"]:.0f}" | '
            f'{p["offset_dl_cosb"]["median"]:+.0f} +- {p["offset_dl_cosb"]["err"]:.0f}" | '
            f'{p["offset_db"]["median"]:+.0f} +- {p["offset_db"]["err"]:.0f}" |'
        )
    L += [
        "",
        "## Flux ratio log10(F_cat(0.2-2.3) / F_2RXS(0.1-2.4))",
        "",
        "The absolute value contains the band conversion and ~30 years of variability; only the",
        "differences below, measured on the same 2RXS sources, are meaningful.",
        "",
        f"| catalogue | N | median +- err | scatter | F_2RXS > {FLUX_BRIGHT:.0e} |",
        "|---|---|---|---|---|",
    ]
    for k in keys:
        v, bt = res[k]["flux_ratio"], res[k]["flux_ratio_bright"]
        L.append(
            f"| {k} | {v['n']} | {v['median']:+.3f} +- {v['err']:.3f} | {v['scatter']:.3f} | "
            f"{bt['median']:+.3f} +- {bt['err']:.3f} ({bt['n']}) |"
        )
    d1, d2 = res["image_RU_minus_image_DE_bright"], res["image_RU_minus_image_DE"]
    p, t = res["image_minus_erass1_DE_paired"], res["image_minus_erass1_DE_paired_test"]
    rev = res["image_with_2rxs_counterpart"]
    L += [
        "",
        f"- image RU - image DE, F_2RXS > {FLUX_BRIGHT:.0e} (both halves ~85% complete, so unbiased): "
        f"{d1['dex']:+.3f} +- {d1['err']:.3f} dex ({abs(d1['dex']) / d1['err']:.1f} sigma residual RU "
        "flux-scale offset after the step-12 correction)",
        f"- same over all matches: {d2['dex']:+.3f} +- {d2['err']:.3f} dex - the halves match different "
        "fractions of 2RXS, so this one also contains a selection term",
        f"- image DE - eRASS1 DE (same 2RXS sources, paired): {p['median']:+.3f} +- {p['err']:.3f} dex "
        f"(scatter {p['scatter']:.3f}, N={p['n']}; in-sample for most sources)",
        f"- same, DE test split only (out-of-sample): {t['median']:+.3f} +- {t['err']:.3f} dex "
        f"(scatter {t['scatter']:.3f}, N={t['n']})",
        "",
        "## Reverse direction: image sources with a 2RXS counterpart",
        "",
        "| half | N image | matched (chance-corrected) | chance |",
        "|---|---|---|---|",
    ]
    for half, v in rev.items():
        L.append(f"| {half} | {v['n']} | {v['completeness']:.1%} | {v['chance']:.1%} |")
    L += ["", "Figure: `rosat_2rxs.pdf`.", ""]
    (OUTDIR / "rosat_2rxs_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
