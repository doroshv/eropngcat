"""Step 13: final all-sky catalogue (FITS + CSV) and README.

* DE (step 10) and RU (step 11) image catalogues merged
* RU flux-scale correction from step 12 applied only if significant (> CORR_SIGMA and
  > CORR_MIN_DEX); the applied factor is stored per source
* RU POS_ERR gets an additive term (in quadrature) from the RU/DE difference of median offsets to
  bright external sources (step 12): RU positions get no residual sky correction (step 09 has no
  RU training catalogue), which adds a roughly constant systematic rather than scaling the error
* flags: image edge / join proximity, SNR (any size, incl. Antlia), extended, saturation,
  position fallback; DE sources get their eRASS1 counterpart for reference
* README.md with method, accuracies (from steps 03, 05, 07, 10, 11, 12) and caveats

Outputs (out/13_final/): erass1_image_catalog.fits, erass1_image_catalog.csv, README.md
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage

from epngcat.config import OUT, ROOT, load_json, save_json, step_dir
from epngcat.evaluate import F_FLOOR, MATCH_R, brightest_within
from epngcat.evaluate import catalog as reference_catalog
from epngcat.image import sky_mask
from epngcat.projection import load_alignment
from epngcat.s09_models import BANDS
from epngcat.sky import sep_arcsec
from epngcat.uncertainty import apply_uncertainty_systematics, calibrate_uncertainties, test_coverage

OUTDIR = step_dir("13_final")
CORR_SIGMA, CORR_MIN_DEX = 3.0, 0.03
POS_REF = ("xmmsl3", "rass_bsc")  # step-12 catalogues used for the RU positional-error term
EDGE_PX = 4.0  # native pixels from the drawn sky outline or from the join between the halves
CODE_URL = "https://github.com/doroshv/eropngcat"  # the public pipeline repository

UNITS = {
    "RA": "deg",
    "DEC": "deg",
    "LII": "deg",
    "BII": "deg",
    "POS_ERR": "arcsec",
    "SAT_AREA_ARCMIN2": "arcmin2",
    "ERASS1_SEP": "arcsec",
}
UNITS.update({c: "erg/s/cm2" for b in BANDS for c in (f"FLUX_{b}", f"FLUX_{b}_LO", f"FLUX_{b}_HI")})
DESCR = {
    "NAME": "source name (Galactic position)",
    "RA": "right ascension ICRS",
    "DEC": "declination ICRS",
    "LII": "Galactic longitude",
    "BII": "Galactic latitude",
    "POS_ERR": "1-sigma positional error per axis, including the global eRASS1-DE systematic and POS_ERR_ADD",
    "FLUX_1": "0.2-2.3 keV flux (image model)",
    "FLUX_P1": "0.2-0.5 keV flux (image model)",
    "FLUX_P2": "0.5-1.0 keV flux (image model)",
    "FLUX_P3": "1.0-2.0 keV flux (image model)",
    "SNR": "detection significance (matched filter); inf for saturated blobs",
    "SATURATED": "source contains saturated image pixels",
    "SAT_AREA_ARCMIN2": "area of the saturated blob",
    "PSF_WIDTH": "free-width PSF fit scale (1 = point source)",
    "FLAG_EXTENDED": "PSF_WIDTH > 1.5 or SAT_AREA_ARCMIN2 > 500: possibly extended",
    "FLAG_POS_FALLBACK": "PSF fit failed, detection position used",
    "FLAG_EDGE": "near the image outline or the join of the two halves",
    "SNR_NAME": "inside the mask of this SNRcat supernova remnant",
    "HALF": "DE (eRASS1-DE footprint) or RU",
    "TILE": "2x2 deg tile id",
    "SPLIT": "DE: train/val/test block of the hold-out scheme; RU: ru",
    "FLUX_SCALE_CORR": "multiplicative flux correction applied (RU)",
    "POS_ERR_ADD": "additive RU positional systematic, applied in quadrature (RU)",
    "ERASS1_NAME": "brightest eRASS1 source within 4 arcmin (F >= 1e-13), DE only",
    "ERASS1_SEP": "separation to ERASS1_NAME",
    "ERASS1_FLUX_1": "eRASS1 0.2-2.3 keV flux of ERASS1_NAME",
}


def edge_flag(l, b):
    """Flag positions near the drawn sky outline or the join between the two image halves."""
    proj = load_alignment()
    dist = ndimage.distance_transform_edt(sky_mask())
    px, py = proj.sky2pix(l, b)
    ix = np.clip(np.round(px).astype(int), 0, dist.shape[1] - 1)
    iy = np.clip(np.round(py).astype(int), 0, dist.shape[0] - 1)
    return (dist[iy, ix] < EDGE_PX) | (np.abs(px - proj.de.x0) < EDGE_PX)


def ru_pos_add(ext):
    """Per-axis RU positional systematic [arcsec]: sqrt(sigma_RU^2 - sigma_DE^2), match-weighted.

    Median radial separations are converted to per-axis sigma (Rayleigh median = sigma sqrt(2 ln 2)).
    The external catalogue's own error is common to both halves and cancels in the difference.
    """
    rayleigh = np.sqrt(2 * np.log(2))
    d2 = [
        (ext[c]["image_RU"]["sep_median"] ** 2 - ext[c]["image_DE"]["sep_median"] ** 2) / rayleigh**2 for c in POS_REF
    ]
    weight = [ext[c]["image_RU"]["n_match"] + ext[c]["image_DE"]["n_match"] for c in POS_REF]
    return float(np.sqrt(max(np.average(d2, weights=weight), 0.0)))


def names(l, b):
    """IAU-style names from Galactic coordinates."""
    return np.array([f"EIMG G{x:07.3f}{y:+07.3f}" for x, y in zip(l, b)])


def product_overview():
    """Describe the products of the full Make run, including stages after catalogue assembly."""
    return f"""## Generated products

Paths below are relative to the repository root. `make all` generates the complete set;
products from stages 14–18 appear after this README is first written in stage 13.

### Released files

The Zenodo archive (built by `tools/make_release.sh` after `make all`) contains data only: this README,
the input image `erass1_allsky.png`, `out/13_final/erass1_image_catalog.fits` and `.csv`,
`out/13_final/uncertainty_calibration.json`, `out/17_orphans/ru_candidates.csv` and `.parquet`,
the three `out/16_healpix/` maps, and `out/run_manifest.json`, at the paths listed below. All other
products, including the per-tile FITS, PDF and HiPS comparison files in `out/14_tiles/`, are not
included; `make all` regenerates them. The pipeline code is not included; it is available at
{CODE_URL}. The input catalogues `eRASS1_Main.v1.2.fits` (eRASS1-DE DR1
Main v1.2), `xmmsl3_v1.0.fits` (XMMSL3) and `5XMM_DR15cat_v1.0.fits` (5XMM-DR15) are public and must
be placed in the repository root; the remaining catalogues and the HiPS maps are downloaded
automatically. `out/run_manifest.json` records the checksums of all inputs.

| Product | Contents |
|---|---|
| `out/13_final/erass1_image_catalog.fits`, `.csv` | Final all-sky source catalogue with calibrated positions, four flux bands, uncertainty bounds, quality flags, and reference associations. |
| `out/13_final/README.md`, `CATALOG_README.md` | Identical catalogue documentation, validation summary, product inventory, and column definitions. |
| `out/13_final/uncertainty_calibration.json` | Selected global uncertainty terms, training/validation comparisons, and base versus corrected test coverage. |
| `out/14_tiles/index.csv` | Tile coordinates, footprint/split, source counts, residual diagnostics, and paths to FITS, PDF, and HiPS products. |
| `out/14_tiles/fits/tile_*.fits` | TAN-projected RGB tiles at 2 arcmin per pixel, in display units: `DATA`, source-plus-background `MODEL`, `BACKGROUND`, source-subtracted `SUB`, and `RESIDUAL` (model minus data), with `VALID`, per-channel `MODELLED`, and a `SOURCES` table. |
| `out/14_tiles/pdf/tile_*.pdf` | Tile previews comparing data, fitted model, background, subtraction and residuals, with catalogue overlays and available DE HiPS comparisons. |
| `out/14_tiles/hires/tile_*_hires.fits` | DE comparison pairs: three-band eRASS1 HiPS rate cubes (`HIRES`, 30 arcsec per pixel) and aligned image cutouts (`LOWRES`, 2 arcmin per pixel), plus `VALID` and WCS information. |
| `out/15_units/units.json`, `.pdf` | Affine conversion from linearised image values to approximate count-rate surface brightness, per-band calibration scatter and diagnostics. |
| `out/16_healpix/healpix_data_nside1024.fits` | Full image with sources retained, assembled from the tile `DATA` planes; no source, core, or footprint mask is applied. |
| `out/16_healpix/healpix_clean_nside1024.fits` | Source-subtracted image from the tile `SUB` planes. Fitted source components are removed; the absent rendered footprint is `healpy.UNSEEN`. This is the primary unfilled subtraction product. |
| `out/16_healpix/healpix_clean_filled_nside1024.fits` | Display companion: the source-subtracted footprint gap is interpolated, then the whole map is Gaussian-smoothed at the calibrated image PSF scale. |
| `out/17_orphans/orphans.parquet`, `orphans.json`, `orphans_report.md`, `orphans.pdf` | Sources lacking specified catalogue associations, match statistics, and diagnostics; absence of a match does not establish a new astrophysical source. |
| `out/17_orphans/ru_candidates.parquet`, `.csv`, `ru_calibration.pdf` | Complementary-footprint candidate list: RU sources with no 2RXS, 1RXS, XMMSL3, 5XMM or ART-XC source within 3 combined per-axis positional errors (`POS_ERR` and the external error calibrated against eRASS1), with approximate spurious probabilities `P_SPUR`, morphology classes, and calibration diagnostics. |
| `out/18_figures/*.pdf` | Publication figures: input image, tile layout and split, catalogue distribution/density, cumulative counts and ratios, full/source-subtracted HEALPix renderings, and bright-source residual examples. |
| `out/18_figures/number_counts.json`, `.csv` | Common count-selection mask definition, retained areas and sample sizes, and the plotted cumulative densities with counting errors. |
| `out/run_manifest.json` | Completed-run record: code revision and checksums, input/cache checksums, retained HiPS files, package versions, and principal result checksums. |

### HEALPix conventions

All three HEALPix files use Galactic coordinates, RING ordering, `NSIDE=1024` (about
3.4 arcmin sampling), and separate `R`, `G`, `B` columns. The default Make products are
in `ct/s/arcmin2`, empirically calibrated to the HiPS 0.2–0.5, 0.5–1.0, and 1.0–2.0 keV
bands, respectively. Header keywords record the stretch, conversion coefficients and their
scatter, masking, and (for the filled map) interpolation fraction and smoothing width.

The source-subtracted maps retain unresolved sources and imperfectly modelled extended emission.
Clipped cores in `SUB` contain fitted background values, not recovered measurements. The
tile-level `VALID` and `MODELLED` masks are not propagated to the HEALPix maps; use the tile
products when these distinctions matter. The gap-filled map is for visualisation.

### Intermediate products and validation

| Directory | Contents |
|---|---|
| `out/01_catalogs/` | Prepared reference catalogues (eRASS1, ROSAT BSC, XMMSL3, 5XMM, SNRcat, MAXI and RXTE). Downloaded/prepared input caches are retained in `data/`. |
| `out/02_align_coarse/`, `out/03_align/` | Image-outline and source-based registration parameters, matched reference-source tables, channel shifts, and alignment diagnostic PDFs. |
| `out/04_tiles/` | Tile geometry, coverage and spatial splits, eRASS1-to-tile assignments, summary statistics, and footprint/split figures. |
| `out/05_calibrate/` | Rendering/stretch and measurement-PSF calibration, calibration-source measurements, and diagnostic figures. |
| `out/06_detect_baseline/`, `out/07_detect_optimize/` | Baseline detections, completeness diagnostics, detection-search trials, adopted settings, and resumable trial cache. |
| `out/07b_psf_scene/` | Training-tile subtraction-PSF calibration and stacked-residual summaries. |
| `out/08_de_pipeline/`, `out/09_models/` | DE image measurements and reference matches, fitted position/flux models, validation-based model selection, and diagnostic tables/figures. |
| `out/10_validate_de/` | Base DE catalogue, spatial-test metrics and report, and detection, positional-error and flux diagnostic figures. |
| `out/11_catalog_ru/` | Complementary-footprint measurements/catalogue and cross-footprint consistency diagnostics before final assembly. |
| `out/12_external/` | External-match statistics, reports and figures for ROSAT, XMM and higher-energy surveys; `flux_comparison.pdf` combines 2RXS, XMMSL3 and DE eRASS1 comparisons, with plotted pairs in `flux_comparison_pairs.csv`. |

`make replot` redraws diagnostics from saved products without refitting the source models.
`out/.stamps/` contains Make completion records, not scientific data.
"""


def main():
    """Run step 13."""
    de = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    ru = pd.read_parquet(OUT / "11_catalog_ru" / "ru_catalog.parquet")
    metrics = load_json(OUT / "10_validate_de" / "metrics.json")
    calibration = calibrate_uncertainties(
        de[~de.IN_SNR_MASK].reset_index(drop=True), reference_catalog(), metrics["validation_detection"]["F50"]
    )
    calibration["test_base"] = test_coverage(de, reference_catalog("test"))
    de = apply_uncertainty_systematics(de, calibration)
    calibration["test_corrected"] = test_coverage(de, reference_catalog("test"))
    save_json(calibration, OUTDIR / "uncertainty_calibration.json")
    ru = apply_uncertainty_systematics(ru, calibration)
    ext = load_json(OUT / "12_external" / "external.json")
    off = ext["ru_minus_de_flux_offset_combined"]
    apply_corr = off["significance"] > CORR_SIGMA and abs(off["dex"]) > CORR_MIN_DEX
    factor = 10 ** (-off["dex"]) if apply_corr else 1.0
    ru["FLUX_SCALE_CORR"] = factor
    de["FLUX_SCALE_CORR"] = 1.0
    for b in BANDS:
        for c in (f"FLUX_{b}", f"FLUX_{b}_LO", f"FLUX_{b}_HI"):
            ru[c] *= factor
    pos_add = ru_pos_add(ext)
    ru["POS_ERR"] = np.hypot(ru.POS_ERR, pos_add)
    ru["POS_ERR_ADD"] = pos_add
    de["POS_ERR_ADD"] = 0.0

    cat = pd.concat([de, ru], ignore_index=True).sort_values(["LII", "BII"]).reset_index(drop=True)
    cat.insert(0, "NAME", names(cat.LII.values, cat.BII.values))
    cat["FLAG_EDGE"] = edge_flag(cat.LII.values, cat.BII.values)
    cat["SNR"] = cat.SNR.astype(float)

    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1)].reset_index(drop=True)
    is_de = (cat.HALF == "DE").values
    idx = brightest_within(cat.LII.values[is_de], cat.BII.values[is_de], er, MATCH_R, F_FLOOR)
    ok = idx >= 0
    cat["ERASS1_NAME"] = ""
    cat["ERASS1_SEP"] = np.nan
    cat["ERASS1_FLUX_1"] = np.nan
    rows = np.nonzero(is_de)[0][ok]
    cat.loc[rows, "ERASS1_NAME"] = er.IAUNAME.values[idx[ok]]
    cat.loc[rows, "ERASS1_SEP"] = sep_arcsec(
        cat.LII.values[rows], cat.BII.values[rows], er.LII.values[idx[ok]], er.BII.values[idx[ok]]
    )
    cat.loc[rows, "ERASS1_FLUX_1"] = er.ML_FLUX_1.values[idx[ok]]
    cat = cat.drop(columns=["IN_SNR_MASK"])

    t = Table.from_pandas(cat)
    for c in t.colnames:
        if c in UNITS:
            t[c].unit = UNITS[c]
        base = c.removesuffix("_LO").removesuffix("_HI")
        if base in DESCR:
            t[c].description = DESCR[base] + (
                " (lower bound of calibrated prediction interval including global systematic)"
                if c.endswith("_LO")
                else " (upper bound of calibrated prediction interval including global systematic)"
                if c.endswith("_HI")
                else ""
            )
    hdu = fits.table_to_hdu(t)
    hdu.name = "CATALOG"
    hdu.header["ORIGIN"] = "epngcat: sources extracted from the eRASS1 all-sky RGB press image"
    hdu.header["RUFLXCOR"] = (factor, "flux scale factor applied to the RU half")
    hdu.header["RUPOSADD"] = (pos_add, "RU positional systematic added in quadrature [arcsec]")
    hdu.header["POSADD"] = (calibration["position"]["add_arcsec"], "global DE positional floor [arcsec]")
    hdu.header["POSMULT"] = (calibration["position"]["fraction"], "global DE positional multiplicative term")
    for band in BANDS:
        hdu.header[f"F{band}ADD"] = (calibration["flux"][band]["add_flux"], "global DE additive flux systematic")
        hdu.header[f"F{band}MULT"] = (calibration["flux"][band]["fraction"], "global DE fractional flux systematic")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(OUTDIR / "erass1_image_catalog.fits", overwrite=True)
    cat.to_csv(OUTDIR / "erass1_image_catalog.csv", index=False)
    write_readme(cat, apply_corr, factor, calibration)
    print(len(cat), cat.HALF.value_counts().to_dict(), "RU correction:", factor)


def write_readme(cat, apply_corr, factor, uncertainty):
    """README with method summary, accuracies and caveats (numbers read from step outputs)."""
    al = load_json(OUT / "03_align" / "alignment.json")
    tiles = load_json(OUT / "04_tiles" / "summary.json")
    cal = load_json(OUT / "05_calibrate" / "calibration.json")
    detp = load_json(OUT / "07_detect_optimize" / "detection_params.json")
    sel = load_json(OUT / "09_models" / "selection.json")
    val = load_json(OUT / "10_validate_de" / "metrics.json")
    cons = load_json(OUT / "11_catalog_ru" / "consistency.json")
    ext = load_json(OUT / "12_external" / "external.json")
    art = load_json(OUT / "12_external" / "artxc.json")
    hard = load_json(OUT / "12_external" / "hard.json")
    rep = al["report"]
    sh = rep["channel_shifts"]
    d, p, f = val["detection"], val["positions"], val["fluxes"]
    ftab = pd.read_csv(OUT / "09_models" / "flux_models.csv")
    ptab = pd.read_csv(OUT / "09_models" / "position_models.csv")
    off = ext["ru_minus_de_flux_offset_combined"]
    n_flag = {k: int(cat[k].sum()) for k in ("SATURATED", "FLAG_EXTENDED", "FLAG_EDGE", "FLAG_POS_FALLBACK")}
    n_snr = int((cat.SNR_NAME != "").sum())

    def fr(x):
        return f"{x:.2e}"

    L = [
        "# eRASS1 image catalogue (EIMG)",
        "",
        "X-ray point-source catalogue extracted from the SRG/eROSITA eRASS1 all-sky RGB press image",
        "(`erass1_allsky.png`, 0.3-2.3 keV) alone. Positions and fluxes are calibrated and validated on the",
        "eRASS1-DE Main catalogue v1.2 (Galactic longitude 180-360 deg) and applied to the whole sky.",
        "",
        "Input image: eRASS1 all-sky press image, MPE/IKI press kit of 19 June 2020,"
        " <https://www.mpe.mpg.de/7461950/erass1-presskit>. Credit: Jeremy Sanders, Hermann Brunner and the"
        " eSASS team (MPE); Eugene Churazov, Marat Gilfanov (on behalf of IKI). It is redistributed here"
        " unchanged, for reproducibility only; it is not covered by the licences below, and its use is subject"
        " to the press-kit terms (free use with the credit above).",
        "",
        "Licence: the catalogue, candidate list and HEALPix maps are dedicated to the public domain under CC0 1.0"
        " (<https://creativecommons.org/publicdomain/zero/1.0/>). The pipeline code is hosted separately at"
        f" <{CODE_URL}> under the MIT licence. Citation of the accompanying paper is appreciated.",
        "",
        f"- sources: **{len(cat)}** (DE: {int((cat.HALF == 'DE').sum())}, RU: {int((cat.HALF == 'RU').sum())})",
        "- files: `erass1_image_catalog.fits` (HDU `CATALOG`, units and column descriptions), `erass1_image_catalog.csv`",
        "- reproduce: see *Reproducing the analysis* below (steps `epngcat/sNN_*.py`, outputs in `out/NN_*`)",
        "- run record: `out/run_manifest.json` records code, input checksums and the locked environment;"
        " the input catalogues are public and not redistributed, and the checksums verify that a re-run uses"
        " the same ones.",
        "",
        "## Reproducing the analysis",
        "",
        f"1. Clone the pipeline from <{CODE_URL}> and place `erass1_allsky.png` from this archive in the"
        " repository root.",
        "2. Install uv: <https://docs.astral.sh/uv/getting-started/installation/>. No other setup is needed;"
        " every Make stage runs through `uv run --frozen`, which creates the Python environment from"
        " `uv.lock` on first use (`uv sync --frozen` does the same explicitly).",
        "3. Place the three public input catalogues in the repository root: `eRASS1_Main.v1.2.fits`"
        " (<https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/>), `xmmsl3_v1.0.fits`"
        " (<https://www.cosmos.esa.int/web/esdc/esasky-catalogues>) and `5XMM_DR15cat_v1.0.fits`"
        " (<https://xmmssc.irap.omp.eu/Catalogue/5XMM-DR15/5XMM_DR15.html>). The other reference catalogues"
        " and the eRASS1 HiPS maps are downloaded automatically.",
        "4. Run `make all` (`make test` runs the unit tests). A full run takes several hours and downloads"
        " about 25 000 HiPS tiles per band; `out/run_manifest.json` from the release lists the expected input"
        " checksums.",
        "",
        product_overview(),
        "",
        "## Accuracy (eRASS1-DE test split, held out from source-model fitting)",
        "",
        "These tests are conditional on shared global image registration. The halo safeguards are fixed"
        " empirical rules derived from inspected fields and fits across both halves; these checks do not"
        " independently demonstrate improved neighbour recovery or diffuse-flux preservation.",
        "",
        f"- reliability (chance-corrected): **{d['reliability']:.1%}**",
        f"- detection limit, 0.2-2.3 keV point sources: 50% complete at **{fr(d['F50'])}** erg/s/cm2; "
        + (f"90% at **{fr(d['F90'])}**" if np.isfinite(d["F90"]) else "90% is not reached")
        + f" (completeness plateaus at {d['completeness_plateau']:.0%}: confusion, extended and flagged sources)",
        "  (50% limit vs |b|: " + ", ".join(f"{k} deg {fr(v)}" for k, v in d["F50_vs_absb"].items()) + ")",
        f'- positions: median / 68% / 90% offset **{p["median_arcsec"]:.0f}" / {p["p68_arcsec"]:.0f}" / {p["p90_arcsec"]:.0f}"**;',
        "  `POS_ERR` model sigma interpolated in log S/N between binned validation values ("
        + ", ".join(
            f'{s:.0f}: {e:.0f}"'
            for s, e in zip(sel["position"]["error_model"]["bins_snr"], sel["position"]["error_model"]["bins_sigma"])
        )
        + "),"
        f' saturated {sel["position"]["error_model"]["sat"]:.0f}"; {p["frac_within_1sigma"]:.0%} / {p["frac_within_2sigma"]:.0%} of offsets within 1 / 2 sigma'
        " (39% / 86% expected)",
        "- fluxes (log10 image / eRASS1):",
        "",
        "| band | bias [dex] | robust scatter [dex] | >0.3 dex outliers | 16-84% interval coverage |",
        "|---|---|---|---|---|",
    ]
    L += [
        f"| {b} | {v['bias_dex']:+.3f} | {v['scatter_dex']:.3f} | {v['frac_outlier_0.3dex']:.1%} | {v['coverage_16_84']:.0%} |"
        for b, v in f.items()
    ]
    L += [
        "",
        "Full numbers: `out/10_validate_de/validation_report.md`.",
        "",
        "### Global uncertainty systematics",
        "",
        f"Systematic terms were calibrated from {uncertainty['n_unique_position_matches']} one-to-one DE associations"
        f" to point-like eRASS1 sources brighter than the validation-set image sensitivity limit ({fr(uncertainty['flux_limit'])})."
        " Candidate base, additive, multiplicative and combined models were fitted on training blocks, selected"
        " by BIC on validation blocks, checked on downstream test blocks, and applied without an all-data refit.",
        f"The positional term is `{uncertainty['position']['selected']}`: final per-axis uncertainty is"
        f" sqrt(POS_ERR_base^2 + ({uncertainty['position']['fraction']:.4f} POS_ERR_base)^2 +"
        f" {uncertainty['position']['add_arcsec']:.2f}^2) arcsec.",
        "Flux interval sides are enlarged in quadrature by the selected global term:",
        "",
        "| band | selected term | additive [erg s$^{-1}$ cm$^{-2}$] | multiplicative fraction |",
        "|---|---|---|---|",
        *[
            f"| {band} | {uncertainty['flux'][band]['selected']} | {uncertainty['flux'][band]['add_flux']:.3e} "
            f"| {uncertainty['flux'][band]['fraction']:.4f} |"
            for band in BANDS
        ],
        "",
        "Full model comparisons and downstream test diagnostics: `out/13_final/uncertainty_calibration.json`.",
        "",
        "### Final uncertainty coverage on the step-10 test sample",
        "",
        "These samples use the same reference/matching cuts as the base-model validation above; no uncertainty-dependent selection is applied.",
        f"Positions: {uncertainty['test_corrected']['positions']['n']} matched point sources; "
        f"{uncertainty['test_corrected']['positions']['frac_within_1sigma']:.1%} / "
        f"{uncertainty['test_corrected']['positions']['frac_within_2sigma']:.1%} within 1 / 2 combined per-axis sigmas "
        "(39.3% / 86.5% expected for a circular Gaussian).",
        "",
        "| band | N | base 16-84% coverage | corrected 16-84% coverage |",
        "|---|---|---|---|",
        *[
            f"| {b} | {uncertainty['test_corrected']['fluxes'][b]['n']} | "
            f"{uncertainty['test_base']['fluxes'][b]['coverage_16_84']:.1%} | "
            f"{uncertainty['test_corrected']['fluxes'][b]['coverage_16_84']:.1%} |"
            for b in BANDS
        ],
        "",
        "## External validation",
        "",
        "| catalogue | vs XMMSL3: detected / median sep / flux scatter | vs RASS BSC: detected / median sep / flux scatter |",
        "|---|---|---|",
    ]
    for key in ("erass1_DE", "image_DE", "image_RU"):
        a, b = ext["xmmsl3"][key], ext["rass_bsc"][key]
        L.append(
            f'| {key} | {a["detected_frac"]:.0%} / {a["sep_median"]:.0f}" / {a["flux_ratio_scatter_dex"]:.2f} dex '
            f'| {b["detected_frac"]:.0%} / {b["sep_median"]:.0f}" / {b["flux_ratio_scatter_dex"]:.2f} dex |'
        )
    L += [
        "",
        "eRASS1 itself scatters by the amount in its row against these catalogues (variability over 5-30 years,",
        "band conversion); the image catalogue adds its own scatter in quadrature.",
        f"RU - DE flux-scale offset: {off['dex']:+.3f} +- {off['err']:.3f} dex ({off['significance']:.1f} sigma); "
        + (
            f"**applied** to RU fluxes (factor {factor:.3f}, column `FLUX_SCALE_CORR`)."
            if apply_corr
            else "not significant, no correction applied."
        ),
        f'RU `POS_ERR` includes a {cat.POS_ERR_ADD[cat.HALF == "RU"].iloc[0]:.1f}" systematic in quadrature, from the RU/DE'
        " difference of median offsets to bright XMMSL3 and ROSAT BSC sources (column `POS_ERR_ADD`); the external"
        " catalogues' own errors cancel in that difference. The excess RU scatter is source-to-source rather than"
        ' a correctable registration error: offsets of neighbouring RU sources share only ~8" per axis.',
        f"Image-only number counts give an RU flux offset of {cons['ru_flux_offset_from_counts_dex']:+.3f} dex (consistency check,"
        " not applied). Full numbers: `out/12_external/external_report.md`, `out/11_catalog_ru/consistency.json`.",
        "",
        "### High-energy all-sky surveys (completeness; bands do not overlap, fluxes compared only as differences)",
        "",
        "| survey | image DE | image RU | eRASS1 DE |",
        "|---|---|---|---|",
        *[
            f"| {n} | {hard[n]['image_DE']['all']['completeness']:.0%} | {hard[n]['image_RU']['all']['completeness']:.0%} "
            f"| {hard[n]['erass1_DE']['all']['completeness']:.0%} |"
            for n in ("maxi_gsc", "rxte_xss", "maxi_ssc")
        ],
        *[
            f"| SRG/ART-XC {n} (point-like) | {art[n]['image_DE']['completeness_point']['completeness']:.0%} "
            f"| {art[n]['image_RU']['completeness_point']['completeness']:.0%} "
            f"| {art[n]['erass1_DE']['completeness_point']['completeness']:.0%} |"
            for n in ("artss1", "artss4")
        ],
        "",
        "SRG/ART-XC (contemporaneous, uniform over both halves): median offsets image DE / RU "
        f'{art["artss4"]["image_DE"]["positions"]["median"]:.0f}" / {art["artss4"]["image_RU"]["positions"]["median"]:.0f}"; '
        "RU - DE flux scale from the 0.2-2.3 / 4-12 keV ratios "
        f"{art['artss4']['image_RU_minus_DE']['all']['dex']:+.3f} +- {art['artss4']['image_RU_minus_DE']['all']['err']:.3f} dex "
        f"(ARTSS4) and {art['artss1']['image_RU_minus_DE']['all']['dex']:+.3f} +- "
        f"{art['artss1']['image_RU_minus_DE']['all']['err']:.3f} dex (ARTSS1); image - eRASS1 flux on the same ART-XC sources "
        f"(DE test split) {art['artss4']['image_minus_erass1_DE_test']['median']:+.3f} +- "
        f"{art['artss4']['image_minus_erass1_DE_test']['err']:.3f} dex. Full numbers: `out/12_external/artxc_report.md`.",
        "",
        "## Method",
        "",
        "1. **Alignment** (`s02`, `s03`). The image is a Hammer-Aitoff projection in Galactic coordinates"
        f' (Mollweide rejected: {rep["projection_comparison_de_erass1_holdout"]["mollweide_poly0"]["median"]:.0f}" vs'
        f' {rep["projection_comparison_de_erass1_holdout"]["hammer_poly0"]["median"]:.0f}" median offsets).'
        " **The two halves were projected separately**: a single projection leaves"
        f' {rep["single_projection"]["bsc_holdout"]["ru"]["median"]:.0f}" median offsets on the RU side, a separate RU fit'
        f' {rep["split_projection"]["bsc_holdout"]["ru"]["median"]:.0f}" (RASS BSC hold-out). The DE half is fitted on bright'
        " eRASS1 sources, the RU half on the RASS BSC; polynomial distortions do not improve hold-out residuals."
        " Inverse projection is exact and tested at the poles and the l=180 seam; image patches are local gnomonic"
        " (TAN) reprojections, so nothing degenerates at the poles.",
        "2. **Channel registration.** In the DE half the blue channel is shifted by"
        f" ({sh['de']['shift_px'][2][0]:+.2f}, {sh['de']['shift_px'][2][1]:+.2f}) native pixels relative to red/green"
        f" (RU: ({sh['ru']['shift_px'][2][0]:+.2f}, {sh['ru']['shift_px'][2][1]:+.2f})); every channel is sampled at its own position.",
        f"3. **Tiling and hold-out** (`s04`): {tiles['n_tiles']} tiles of ~2x2 deg; DE tiles grouped into 10 deg blocks,"
        " randomly split 60/20/20 into train/validation/test.",
        "4. **Supernova remnants** (SNRcat): sources inside remnants up to 5 deg across (mask radius 1.2 x radius + 6')"
        " are excluded from all calibration and validation; the 24-deg Antlia remnant is flagged only.",
        f"5. **Display stretch** (`s05`): inverse asinh, I = sinh((v - v0)/A) with A = {', '.join(f'{a:.0f}' for a in cal['stretch']['A'])}"
        f" (R, G, B), chosen over linear, gamma and log stretches on held-out sources.",
        f"6. **PSF** (`s05`): Gaussian with covariance t2*I + s2*J^-1 J^-T in the TAN patch (sky sigma"
        f" {cal['psf']['sky_sigma_arcmin']:.1f}', plus a native-pixel term that elongates sources where the Hammer grid is sheared).",
        "7. **Detection** (`s06`, `s07`): background-subtracted, noise-weighted linearised channels, matched filter, local"
        " maxima; saturated blobs are one detection each. Parameters tuned on train"
        f" (search config: {detp['search_config']}) with S/N threshold {detp['params']['thresh']}, selected on validation.",
        "8. **Measurement and models** (`s08`, `s09`): positions from"
        f" `{sel['position']['estimator']}` (per-S/N-bin choice among {ptab.estimator.nunique() - 1} estimators, each optionally"
        " with a per-half sky-frame astrometric correction (+sky; +skybin adds a per-bin offset), selected on"
        " validation; bins: "
        + (
            ", ".join(
                f"{k}: {v[0]}{ {'sky': '+sky', 'skybin': '+skybin'}.get(v[1], '') }"
                for k, v in sel["position"]["corrected"].items()
            )
            if sel["position"]["estimator"] == "blend"
            else str(sel["position"]["corrected"])
        )
        + f"); fluxes from `{sel['flux']['model']}` (best of"
        f" {', '.join(ftab[ftab.band == '1'].model)}) using PSF amplitudes, apertures, saturated-blob area, background and"
        " geometry features; quantile boosting supplies the base interval and the global DE systematic calibration"
        " above is added in quadrature to give `FLUX_*_LO/HI`.",
        "",
        "## Caveats and identified issues",
        "",
        "- The image is a rendering: 8-bit, stretched, smoothed (PSF sigma ~5'), and saturated for bright sources"
        f" ({n_flag['SATURATED']} saturated sources). Saturated fluxes rely on blob size and wings; see the saturated"
        " scatter in the validation report.",
        "- Confusion: at ~5' resolution, image sources are often blends of several eRASS1 sources. The flux of a source"
        " is calibrated against its brightest counterpart, so blends are biased high; positions follow the light centroid.",
        "- Eddington bias: near the detection limit image fluxes of detected sources are biased high.",
        f"- Coverage: {tiles['sky_area_not_in_image_deg2']:.0f} deg2 ({tiles['sky_area_not_in_image_frac']:.2%}) along the image outline"
        " (around l = 180 deg and the poles) is cropped from the image and has no sources;"
        f" {n_flag['FLAG_EDGE']} sources near the outline or the join between the halves are flagged `FLAG_EDGE`.",
        "- The RU half is calibrated with DE models only. Its projection, channel registration and stretch were checked"
        " (RASS BSC, XMMSL3, number counts), but RU-specific rendering differences smaller than these checks cannot be excluded.",
        f"- Extended emission: {n_flag['FLAG_EXTENDED']} sources have broad profiles (`FLAG_EXTENDED`); {n_snr} lie inside"
        " SNRcat remnants (`SNR_NAME`). Clusters, SNR knots, the Magellanic Clouds and diffuse clumps can appear as sources.",
        "- The eRASS1-DE / RU footprint boundary is 179.944 / 359.944 deg, while the image halves join at l = 0 / 180 deg;"
        " sources in the 0.056-deg sliver are labelled by the footprint (`HALF`).",
        "- DE entries in the train/validation splits are in-sample for the models (`SPLIT`); quote test-split accuracies.",
        "- Fluxes assume eRASS1 spectral mix; sources with unusual spectra (very soft/hard) have larger band-flux errors.",
        "- Band-flux validation covers only sources eRASS1 detected in that band: its `ML_FLUX_P1` is exactly zero"
        " for 28% of the catalogue (P2 6%, P3 9%), and those rows drop out of the P1/P2/P3 rows above. The"
        " quoted scatter and coverage therefore describe detected-in-band sources, not the band as a whole.",
        "",
        "## Columns",
        "",
        "| column | unit | description |",
        "|---|---|---|",
    ]
    for c in cat.columns:
        base = c.removesuffix("_LO").removesuffix("_HI")
        desc = DESCR.get(base, "")
        if c.endswith("_LO"):
            desc += " (lower calibrated prediction bound including global systematic)"
        elif c.endswith("_HI"):
            desc += " (upper calibrated prediction bound including global systematic)"
        L.append(f"| {c} | {UNITS.get(c, '')} | {desc} |")
    L.append("")
    text = "\n".join(L)
    (OUTDIR / "README.md").write_text(text)
    (ROOT / "CATALOG_README.md").write_text(text)


if __name__ == "__main__":
    main()
