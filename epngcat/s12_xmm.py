"""Step 12 external validation: final-catalogue flux comparison against 5XMM-DR15.

5XMM-DR15 is the deepest external catalogue that overlaps the image, and its EPIC bands 1 + 2 + 3
sum to 0.2-2.0 keV, the same band as the catalogue's FLUX_P1 + FLUX_P2 + FLUX_P3. Neither side
needs a band conversion, so this is the cleanest flux check available - unlike the step-12
comparisons with XMMSL3 (0.2-2 keV, one fixed ECF per source) and the ROSAT BSC (0.1-2.4 keV, a
hardness-dependent conversion), both of which carry a band term of their own.

5XMM covers ~1400 deg2 in pointed observations, so the overlap is a small, effectively random
sample of the sky, independent of anything the catalogue was tuned on.

The two halves are compared separately: they come from different processings of the survey, and
the final catalogue already carries the residual RU-DE flux-scale correction measured in step 12,
so a leftover difference here would show that the correction is incomplete.

5XMM is filtered in step 01 to fluxes above XMM_FLUX_MIN, comfortably below the image's own
completeness limit, so the comparison is not limited by that cut - but it does mean the plot has
no counterpart to the faint cloud of the eROSITA DR2 figure, which reaches far below it.

Matching is nearest-neighbour within MATCH_ARCSEC, with the chance rate measured by repeating the
match on positions displaced by SHIFT_ARCMIN and subtracted from the counts.

Outputs (out/12_external/): xmm_flux.json, xmm_flux.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epngcat.config import OUT, is_german, save_json, step_dir
from epngcat.evaluate import in_snr, shifted
from epngcat.plotting import figure_size, save_figure
from epngcat.s01_prepare_catalogs import XMM_FLUX_MIN
from epngcat.sky import crossmatch

OUTDIR = step_dir("12_external")
MATCH_ARCSEC = 45.0  # comparable with the catalogue's own POS_ERR; the chance rate is measured
SHIFT_ARCMIN = 30.0
STRONG_SIGMA = 8.0  # "strong" sample: this many sigma in both catalogues
HALVES = {"DE": "tab:blue", "RU": "tab:red"}


def load():
    """Load the final catalogue and 5XMM-DR15, both with a 0.2-2 keV flux and outside SNR masks."""
    cat = pd.read_csv(
        OUT / "13_final" / "erass1_image_catalog.csv",
        keep_default_na=False,
        na_values=[""],
        dtype={"ERASS1_NAME": str},
        low_memory=False,
    )
    for c in ("FLUX_P1", "FLUX_P2", "FLUX_P3", "FLUX_P1_LO", "FLUX_P3_HI", "SNR", "LII", "BII"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["flux_02_2"] = cat.FLUX_P1 + cat.FLUX_P2 + cat.FLUX_P3
    cat = cat[np.isfinite(cat.flux_02_2) & (cat.flux_02_2 > 0)]
    cat = cat[~in_snr(cat.LII.values, cat.BII.values)].reset_index(drop=True)
    xmm = pd.read_parquet(OUT / "01_catalogs" / "xmm_dr15.parquet")
    xmm = xmm[~in_snr(xmm.lii.values, xmm.bii.values)].reset_index(drop=True)
    return cat, xmm


def match(cat, xmm):
    """Pair the two catalogues; returns the matches with both fluxes and the chance-match fraction."""
    idx, sep = crossmatch(cat.LII.values, cat.BII.values, xmm.lii.values, xmm.bii.values, MATCH_ARCSEC)
    idx_s, _ = crossmatch(
        *shifted(cat.LII.values, cat.BII.values, SHIFT_ARCMIN), xmm.lii.values, xmm.bii.values, MATCH_ARCSEC
    )
    ok = idx >= 0
    m = xmm.iloc[idx[ok]]
    pairs = pd.DataFrame(
        {
            "half": cat.HALF.values[ok],
            "sep": sep[ok],
            "f_img": cat.flux_02_2.values[ok],
            "snr_img": cat.SNR.values[ok],
            "f_xmm": m.flux_02_2.values,
            "e_xmm": m.flux_02_2_err.values,
        }
    )
    pairs["ratio_dex"] = np.log10(pairs.f_img) - np.log10(pairs.f_xmm)
    pairs["strong"] = (pairs.snr_img >= STRONG_SIGMA) & (pairs.f_xmm >= STRONG_SIGMA * pairs.e_xmm)
    return pairs, float((idx_s >= 0).mean())


def stats(r):
    """Median offset, robust scatter and bootstrap error of a set of log flux ratios."""
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {"n": int(len(r))}
    med = float(np.median(r))
    boot = np.median(np.random.default_rng(0).choice(r, (1000, len(r))), axis=1)
    return {
        "n": int(len(r)),
        "median_dex": med,
        "median_err_dex": float(boot.std()),
        "scatter_dex": float(1.4826 * np.median(np.abs(r - med))),
        "frac_within_2x": float(np.mean(np.abs(r) < np.log10(2))),
    }


def main():
    """Run the 5XMM part of step 12 external validation."""
    cat, xmm = load()
    pairs, chance = match(cat, xmm)
    print(f"{len(cat)} catalogue x {len(xmm)} 5XMM -> {len(pairs)} matches, chance fraction {chance:.3f}")

    out = {
        "match_radius_arcsec": MATCH_ARCSEC,
        "chance_fraction": chance,
        "band": "0.2-2.0 keV (image P1+P2+P3 vs 5XMM EP_1+EP_2+EP_3)",
        "n_5xmm": len(xmm),
        "sep_median_arcsec": float(np.median(pairs.sep)),
        "all": stats(pairs.ratio_dex.values),
        "strong": stats(pairs.ratio_dex.values[pairs.strong.values]),
    }
    for h in HALVES:
        s = pairs[pairs.half == h]
        out[h] = {
            "n_match": len(s),
            "n_cat": int((cat.HALF.values == h).sum()),
            "n_5xmm": int((is_german(xmm.lii.values) == (h == "DE")).sum()),
            "sep_median_arcsec": float(np.median(s.sep)) if len(s) else np.nan,
            "all": stats(s.ratio_dex.values),
            "strong": stats(s.ratio_dex.values[s.strong.values]),
        }
    # The halves cover comparable sky, so a match rate that differs by much more than the source
    # counts do means one half is genuinely matching 5XMM less often, not that it was sampled less.
    rate = {h: out[h]["n_match"] / max(out[h]["n_5xmm"], 1) for h in HALVES}
    out["match_rate_per_5xmm_source"] = rate
    out["de_over_ru_match_rate"] = float(rate["DE"] / rate["RU"]) if rate["RU"] else np.inf
    de, ru = out["DE"]["strong"], out["RU"]["strong"]
    if "median_dex" in de and "median_dex" in ru:
        d = ru["median_dex"] - de["median_dex"]
        out["ru_minus_de_dex"] = {
            "value": d,
            "err": float(np.hypot(de["median_err_dex"], ru["median_err_dex"])),
            "significance": float(abs(d) / np.hypot(de["median_err_dex"], ru["median_err_dex"])),
        }
    save_json(out, OUTDIR / "xmm_flux.json")
    for k in ("all", "strong"):
        v = out[k]
        print(
            f"  {k:7s} n={v['n']:5d}  {v['median_dex']:+.3f} +- {v['median_err_dex']:.3f} dex, "
            f"scatter {v['scatter_dex']:.3f}, within 2x {v['frac_within_2x']:.0%}"
        )

    plot_results(pairs, chance)


def plot_results(pairs, chance):
    """Render diagnostics separately from the expensive analysis."""
    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.7))
    lim = (0.5 * XMM_FLUX_MIN, 1e-9)
    grid = np.array(lim)
    ax.plot(grid, grid, "k-", lw=1.2, zorder=5)
    for f in (2.0, 0.5):
        ax.plot(grid, f * grid, "k--", lw=0.9, zorder=5)
    for h, colour in HALVES.items():
        s = pairs[pairs.half == h]
        ax.scatter(s.f_xmm, s.f_img, s=3, c=colour, alpha=0.35, lw=0, label=f"image {h}  N={len(s)}")
        t = s[s.strong]
        ax.scatter(
            t.f_xmm,
            t.f_img,
            s=5,
            marker=".",
            color=colour,
            alpha=0.65,
            lw=0,
            label=f"{h} >{STRONG_SIGMA:.0f}sigma  N={len(t)}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("5XMM flux (0.2–2.0 keV)\n[erg cm$^{-2}$ s$^{-1}$]")
    ax.set_ylabel("Image flux (0.2–2.0 keV)\n[erg cm$^{-2}$ s$^{-1}$]")
    ax.set_title(rf"5XMM comparison (${MATCH_ARCSEC:.0f}^{{\prime\prime}}$ match)")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.85)

    save_figure(fig, OUTDIR / "xmm_flux.pdf")

    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.2))
    ax.axhline(0, color="k", lw=1.2)
    for f in (np.log10(2), -np.log10(2)):
        ax.axhline(f, color="k", ls="--", lw=0.9)
    edges = np.logspace(np.log10(XMM_FLUX_MIN), -10, 12)
    for h, colour in HALVES.items():
        s = pairs[pairs.half == h]
        ax.scatter(s.f_xmm, s.ratio_dex, s=3, c=colour, alpha=0.3, lw=0, label=f"image {h}")
        k = np.digitize(s.f_xmm.values, edges)
        medians = [
            np.median(s.ratio_dex.values[k == i]) if (k == i).sum() >= 10 else np.nan for i in range(1, len(edges))
        ]
        ax.stairs(medians, edges, baseline=None, color=colour, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.5 * XMM_FLUX_MIN, 1e-9)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("5XMM flux (0.2–2.0 keV)\n[erg cm$^{-2}$ s$^{-1}$]")
    ax.set_ylabel(r"$\log_{10}(F_{\mathrm{image}} / F_{\mathrm{5XMM}})$")
    ax.set_title("Median residuals per flux bin")
    ax.legend(loc="upper right", markerscale=3)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "xmm_flux_ratio.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
