"""Step 12 external validation: SRG/ART-XC all-sky surveys (4-12 keV).

Catalogues (CDS/VizieR, cached in data/):
  artss1   Pavlinsky+ 2022, J/A+A/661/A38 - first survey year (Dec 2019 - Dec 2020), i.e.
           contemporaneous with eRASS1
  artss4   Sazonov+ 2024, J/A+A/687/A183 (updated 2025) - four-year survey, with 98% positional errors

ART-XC observes the whole sky uniformly, so the same comparison on both halves tests whether the
RU part of the image catalogue behaves like the DE part. eRASS1 (DE only) is compared with the
same ART-XC sources as the reference for what a soft-band survey recovers.

Per catalogue, for image DE, image RU and eRASS1 DE:
  * completeness (chance-corrected with positions displaced by 30'), vs ART-XC flux, source
    class and |b|
  * positional offsets of matches (median / 68% / 90%), mean offset (dl cos b, db), and the
    offset normalised by the combined 1-sigma errors (artss4 only)
  * flux ratio log10(F_0.2-2.3 / F_4-12) per source class; the band ratio depends on the
    spectrum, so only differences between catalogues are meaningful:
      image DE - eRASS1 DE on the same ART-XC sources (image calibration at the bright end)
      image RU - image DE per class (RU flux-scale check, bootstrap errors)
Excluded: sources inside SNR masks, extended ART-XC sources/clusters for positions and fluxes,
sources within 0.5 deg of the image joins.

Outputs (out/12_external/): artxc.json, artxc_report.md, artxc.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from epngcat.config import DATA, OUT, SEED, is_german, save_json, step_dir
from epngcat.evaluate import in_snr, shifted
from epngcat.plotting import figure_size, save_figure
from epngcat.sky import crossmatch

OUTDIR = step_dir("12_external")
CATALOGS = {"artss1": "J/A+A/661/A38", "artss4": "J/A+A/687/A183"}
R_IMAGE = 240.0  # arcsec
R_ERASS = 60.0  # arcsec
SHIFT = 30.0  # arcmin
FLUX_EDGES = np.array([0, 5, 10, 30, 100, np.inf]) * 1e-12
CLASSES = {
    "AGN": ("SEYFERT", "AGN", "BLAZAR", "QSO"),
    "Galactic binaries": ("LMXB", "HMXB", "CV"),
    "stars": ("STAR",),
    "clusters": ("CLUSTER",),
    "unidentified": ("UNIDENT",),
}


def source_class(types):
    """Coarse class from the ART-XC type column (uncertain types, '?', count as their class)."""
    t = pd.Series(types).astype(str).str.upper().str.replace("?", "", regex=False).str.strip()
    out = np.full(len(t), "other", dtype=object)
    for name, keys in CLASSES.items():
        out[t.isin(keys).values] = name
    return out


def load(name):
    """ART-XC catalogue with Galactic positions, 4-12 keV flux [erg s$^{-1}$ cm$^{-2}$] and class."""
    cache = DATA / f"artxc_{name}.parquet"
    if not cache.exists():
        from astroquery.vizier import Vizier

        DATA.mkdir(exist_ok=True)
        t = Vizier(columns=["**"], row_limit=-1).get_catalogs(CATALOGS[name])[0]
        df = t.to_pandas()
        df[[c for c in df.columns if c != "recno"]].to_parquet(cache)
    df = pd.read_parquet(cache)
    g = SkyCoord(df.RAJ2000.values, df.DEJ2000.values, unit="deg").galactic
    out = pd.DataFrame(
        {
            "name": df.Name.astype(str).values,
            "l": g.l.deg,
            "b": g.b.deg,
            "flux": df.Flux.values * 1e-12,
            "cls": source_class(df.Type),
            "extended": (df["Ext"].fillna(0).values > 0) if "Ext" in df else np.zeros(len(df), bool),
            # 98% radius -> 1-sigma per axis for a 2D Gaussian (sqrt(-2 ln 0.02) = 2.797).
            "pos_sigma": df["ePos"].values / 2.797 if "ePos" in df else np.full(len(df), np.nan),
        }
    )
    out["extended"] |= out.cls == "clusters"
    keep = ~in_snr(out.l.values, out.b.values) & (np.abs(((out.l + 90) % 180) - 90) > 0.5)
    out = out[keep].reset_index(drop=True)
    out["half"] = np.where(is_german(out.l), "DE", "RU")
    return out


def match(art, lon, lat, radius):
    """Nearest match of each ART-XC source and the chance-match flags."""
    idx, sep = crossmatch(art.l.values, art.b.values, lon, lat, radius)
    ic, _ = crossmatch(*shifted(art.l.values, art.b.values, SHIFT), lon, lat, radius)
    return idx, sep, ic >= 0


def completeness(matched, chance, sel):
    """Chance-corrected completeness of a subset."""
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    f, pc = matched[sel].mean(), chance[sel].mean()
    return {"n": n, "completeness": float((f - pc) / (1 - pc)), "chance": float(pc)}


def boot_median(v, n=2000):
    """Median and bootstrap standard error."""
    v = np.asarray(v)
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


def analyse(art, cats):
    """All comparisons for one ART-XC catalogue."""
    res, extra = {}, {}
    for key, (cat, lcol, bcol, fcol, perr, radius) in cats.items():
        half = key.split("_")[1]
        a = art[art.half == half].reset_index(drop=True)
        idx, sep, chance = match(a, cat[lcol].values, cat[bcol].values, radius)
        ok = idx >= 0
        r = {
            "completeness_all": completeness(ok, chance, np.ones(len(a), bool)),
            "completeness_point": completeness(ok, chance, ~a.extended.values),
        }
        r["completeness_vs_flux"] = {
            f"{lo:.0e}-{hi:.0e}": completeness(ok, chance, (a.flux.values >= lo) & (a.flux.values < hi))
            for lo, hi in zip(FLUX_EDGES[:-1], FLUX_EDGES[1:])
        }
        r["completeness_vs_class"] = {c: completeness(ok, chance, a.cls.values == c) for c in [*CLASSES, "other"]}
        r["completeness_abs_b_lt10"] = completeness(ok, chance, np.abs(a.b.values) < 10)
        r["completeness_abs_b_gt10"] = completeness(ok, chance, np.abs(a.b.values) >= 10)

        pt = ok & ~a.extended.values
        m = cat.iloc[idx[pt]]
        s = sep[pt]
        dl = ((m[lcol].values - a.l.values[pt] + 180) % 360 - 180) * np.cos(np.radians(a.b.values[pt])) * 3600
        db = (m[bcol].values - a.b.values[pt]) * 3600
        pos = {
            "n": int(pt.sum()),
            "median": float(np.median(s)),
            "p68": float(np.percentile(s, 68)),
            "p90": float(np.percentile(s, 90)),
            "offset_dl_cosb": boot_median(dl),
            "offset_db": boot_median(db),
        }
        sig = np.hypot(m[perr].values if perr else 0.0, a.pos_sigma.values[pt])
        if np.isfinite(sig).any():
            z = s / sig
            pos["frac_within_1sigma"] = float(np.nanmean(z < 1))
            pos["frac_within_2sigma"] = float(np.nanmean(z < 2))
        r["positions"] = pos

        ratio = np.log10(m[fcol].values) - np.log10(a.flux.values[pt])
        cls = a.cls.values[pt]
        r["flux_ratio"] = {"all": boot_median(ratio), **{c: boot_median(ratio[cls == c]) for c in CLASSES}}
        res[key] = r
        extra[key] = {
            "flux": a.flux.values,
            "ok": ok,
            "chance": chance,
            "ratio": ratio,
            "art_flux": a.flux.values[pt],
            "cls": cls,
            "sep": s,
            "names": a.name.values[pt],
            "cat_idx": idx[pt],
        }

    # Same ART-XC sources: image DE vs eRASS1 DE (bright-end calibration of the image fluxes).
    # DE train/val image fluxes are in-sample for the flux model, so the test split is quoted separately.
    e, i = extra["erass1_DE"], extra["image_DE"]
    _, ie, ii = np.intersect1d(e["names"], i["names"], return_indices=True)
    diff = i["ratio"][ii] - e["ratio"][ie]
    split = cats["image_DE"][0].SPLIT.values[i["cat_idx"][ii]]
    res["image_minus_erass1_DE"] = boot_median(diff)
    res["image_minus_erass1_DE_test"] = boot_median(diff[split == "test"])
    # RU flux scale: image RU - image DE per class (difference of medians).
    res["image_RU_minus_DE"] = {}
    for c in ["all", *CLASSES]:
        a, b = res["image_RU"]["flux_ratio"].get(c, {}), res["image_DE"]["flux_ratio"].get(c, {})
        if "median" in a and "median" in b:
            res["image_RU_minus_DE"][c] = {"dex": a["median"] - b["median"], "err": float(np.hypot(a["err"], b["err"]))}
    return res, extra


def main():
    """Run the ART-XC part of step 12 external validation."""
    de = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    ru = pd.read_parquet(OUT / "11_catalog_ru" / "ru_catalog.parquet")
    img = pd.concat([de, ru], ignore_index=True)
    img = img[~img.IN_SNR_MASK].reset_index(drop=True)
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)].reset_index(drop=True)
    er["POS_SIG"] = er.POS_ERR / np.sqrt(2)  # eRASS1 POS_ERR is the 2D (radial) 1-sigma error
    cats = {
        "image_DE": (img, "LII", "BII", "FLUX_1", "POS_ERR", R_IMAGE),
        "image_RU": (img, "LII", "BII", "FLUX_1", "POS_ERR", R_IMAGE),
        "erass1_DE": (er, "LII", "BII", "ML_FLUX_1", "POS_SIG", R_ERASS),
    }
    results, extras = {}, {}
    for name in CATALOGS:
        art = load(name)
        print(name, len(art), "ePos sigma median [arcsec]:", np.nanmedian(art.pos_sigma))
        results[name], extras[name] = analyse(art, cats)
    save_json(results, OUTDIR / "artxc.json")
    plot(extras)
    write_report(results)
    for name, r in results.items():
        print(
            name,
            {k: r[k]["completeness_point"] for k in cats},
            r["image_minus_erass1_DE"],
            r["image_RU_minus_DE"].get("all"),
        )


def plot(extras):
    """Completeness vs flux, separation CDFs and flux ratios."""
    fig, axs = plt.subplots(len(extras), 3, figsize=figure_size(len(extras), 3), squeeze=False)
    colors = {"image_DE": "C0", "image_RU": "C3", "erass1_DE": "k"}
    for row, (name, ex) in zip(axs, extras.items()):
        for key, e in ex.items():
            comp = []
            for lo, hi in zip(FLUX_EDGES[:-1], FLUX_EDGES[1:]):
                sel = (e["flux"] >= lo) & (e["flux"] < hi)
                if sel.sum() > 5:
                    pc = e["chance"][sel].mean()
                    comp.append((e["ok"][sel].mean() - pc) / (1 - pc))
                else:
                    comp.append(np.nan)
            row[0].stairs(comp, np.clip(FLUX_EDGES, 2e-12, 3e-10), baseline=None, color=colors[key], label=key)
            s = np.sort(e["sep"])
            row[1].plot(s, np.arange(1, len(s) + 1) / len(s), c=colors[key], label=key)
            if key != "erass1_DE":
                row[2].scatter(e["art_flux"], e["ratio"], s=8, c=colors[key], alpha=0.6, label=key)
        row[0].set_xscale("log")
        row[0].set_ylim(0, 1.05)
        row[0].set_xlabel(f"{name} F(4-12 keV)")
        row[0].set_ylabel("completeness (chance-corrected)")
        row[1].set_xlim(0, 240)
        row[1].set_xlabel("separation [arcsec]")
        row[1].set_ylabel("cumulative fraction")
        row[2].set_xscale("log")
        row[2].set_xlabel(f"{name} F(4-12 keV)")
        row[2].set_ylabel(r"$\log_{10}(F_{\mathrm{image}} / F_{\mathrm{ART}})$")
        for ax in row:
            ax.legend()
            ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "artxc.pdf")


def fmt(c):
    """Completeness cell."""
    return f"{c['completeness']:.0%} ({c['n']})" if c.get("n") else "-"


def write_report(results):
    """Markdown report."""
    L = [
        "# Validation against SRG/ART-XC all-sky surveys (4-12 keV)",
        "",
        "ARTSS1: Pavlinsky+ 2022 (J/A+A/661/A38, first year, contemporaneous with eRASS1); ARTSS4: Sazonov+ 2024",
        f'(J/A+A/687/A183, four years). Matches within {R_IMAGE:.0f}" (image) / {R_ERASS:.0f}" (eRASS1), completeness',
        f"chance-corrected with positions displaced by {SHIFT:.0f}'. SNR masks and the image joins are excluded;",
        "positions and fluxes use point-like ART-XC sources only.",
        "",
    ]
    for name, r in results.items():
        keys = ["image_DE", "image_RU", "erass1_DE"]
        L += [
            f"## {name}",
            "",
            "### Completeness (N in brackets)",
            "",
            "| subset | " + " | ".join(keys) + " |",
            "|---|" + "---|" * len(keys),
        ]
        L.append("| all | " + " | ".join(fmt(r[k]["completeness_all"]) for k in keys) + " |")
        L.append("| point-like | " + " | ".join(fmt(r[k]["completeness_point"]) for k in keys) + " |")
        L.append("| abs(b) < 10 | " + " | ".join(fmt(r[k]["completeness_abs_b_lt10"]) for k in keys) + " |")
        L.append("| abs(b) >= 10 | " + " | ".join(fmt(r[k]["completeness_abs_b_gt10"]) for k in keys) + " |")
        for b in r["image_DE"]["completeness_vs_flux"]:
            L.append(f"| F(4-12) {b} | " + " | ".join(fmt(r[k]["completeness_vs_flux"][b]) for k in keys) + " |")
        for c in [*CLASSES, "other"]:
            L.append(f"| {c} | " + " | ".join(fmt(r[k]["completeness_vs_class"][c]) for k in keys) + " |")
        L += [
            "",
            "### Positions (point-like matches)",
            "",
            "| catalogue | N | median | 68% | 90% | dl cos b | db | within 1 / 2 sigma |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k in keys:
            p = r[k]["positions"]
            w = f"{p['frac_within_1sigma']:.0%} / {p['frac_within_2sigma']:.0%}" if "frac_within_1sigma" in p else "-"
            L.append(
                f'| {k} | {p["n"]} | {p["median"]:.0f}" | {p["p68"]:.0f}" | {p["p90"]:.0f}" | '
                f'{p["offset_dl_cosb"]["median"]:+.0f} +- {p["offset_dl_cosb"]["err"]:.0f}" | '
                f'{p["offset_db"]["median"]:+.0f} +- {p["offset_db"]["err"]:.0f}" | {w} |'
            )
        L += [
            "",
            "Expected for Gaussian errors: 39% / 86% within 1 / 2 sigma.",
            "",
            "### Flux ratio log10(F_0.2-2.3 / F_ART 4-12): median +- error (scatter), N",
            "",
            "| class | " + " | ".join(keys) + " |",
            "|---|" + "---|" * len(keys),
        ]
        for c in ["all", *CLASSES]:
            cells = []
            for k in keys:
                v = r[k]["flux_ratio"].get(c, {})
                cells.append(
                    f"{v['median']:+.2f} +- {v['err']:.2f} ({v['scatter']:.2f}), {v['n']}" if "median" in v else "-"
                )
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        d = r["image_minus_erass1_DE"]
        t = r["image_minus_erass1_DE_test"]
        L += [
            "",
            f"- image DE - eRASS1 DE on the same ART-XC sources: {d['median']:+.3f} +- {d['err']:.3f} dex "
            f"(scatter {d['scatter']:.2f}, N={d['n']}; in-sample for most sources)",
            f"- same, DE test split only (out-of-sample): {t['median']:+.3f} +- {t['err']:.3f} dex "
            f"(scatter {t['scatter']:.2f}, N={t['n']})",
        ]
        for c, v in r["image_RU_minus_DE"].items():
            L.append(f"- image RU - image DE, {c}: {v['dex']:+.3f} +- {v['err']:.3f} dex")
        L.append("")
    L += ["Figure: `artxc.pdf`.", ""]
    (OUTDIR / "artxc_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
