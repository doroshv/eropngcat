# eRASS1 image catalogue (EIMG)

X-ray point-source catalogue extracted from the SRG/eROSITA eRASS1 all-sky RGB press image
(`erass1_allsky.png`, 0.3-2.3 keV) alone. Positions and fluxes are calibrated and validated on the
eRASS1-DE Main catalogue v1.2 (Galactic longitude 180-360 deg) and applied to the whole sky.

Input image: eRASS1 all-sky press image, MPE/IKI press kit of 19 June 2020, <https://www.mpe.mpg.de/7461950/erass1-presskit>. Credit: Jeremy Sanders, Hermann Brunner and the eSASS team (MPE); Eugene Churazov, Marat Gilfanov (on behalf of IKI). It is redistributed here unchanged, for reproducibility only; it is not covered by the licences below, and its use is subject to the press-kit terms (free use with the credit above).

Licence: the catalogue, candidate list and HEALPix maps are dedicated to the public domain under CC0 1.0 (<https://creativecommons.org/publicdomain/zero/1.0/>). The pipeline code is hosted separately at <https://github.com/doroshv/eropngcat> under the MIT licence. Citation of the accompanying paper is appreciated.

- sources: **63260** (DE: 32379, RU: 30881)
- files: `erass1_image_catalog.fits` (HDU `CATALOG`, units and column descriptions), `erass1_image_catalog.csv`
- reproduce: see *Reproducing the analysis* below (steps `epngcat/sNN_*.py`, outputs in `out/NN_*`)
- run record: `out/run_manifest.json` records code, input checksums and the locked environment; the input catalogues are public and not redistributed, and the checksums verify that a re-run uses the same ones.

## Reproducing the analysis

1. Clone the pipeline from <https://github.com/doroshv/eropngcat> and place `erass1_allsky.png` from this archive in the repository root.
2. Install uv: <https://docs.astral.sh/uv/getting-started/installation/>. No other setup is needed; every Make stage runs through `uv run --frozen`, which creates the Python environment from `uv.lock` on first use (`uv sync --frozen` does the same explicitly).
3. Place the three public input catalogues in the repository root: `eRASS1_Main.v1.2.fits` (<https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/>), `xmmsl3_v1.0.fits` (<https://www.cosmos.esa.int/web/esdc/esasky-catalogues>) and `5XMM_DR15cat_v1.0.fits` (<https://xmmssc.irap.omp.eu/Catalogue/5XMM-DR15/5XMM_DR15.html>). The other reference catalogues and the eRASS1 HiPS maps are downloaded automatically.
4. Run `make all` (`make test` runs the unit tests). A full run takes several hours and downloads about 25 000 HiPS tiles per band; `out/run_manifest.json` from the release lists the expected input checksums.

## Generated products

Paths below are relative to the repository root. `make all` generates the complete set;
products from stages 14–18 appear after this README is first written in stage 13.

### Released files

The Zenodo archive (built by `tools/make_release.sh` after `make all`) contains data only: this README,
the input image `erass1_allsky.png`, `out/13_final/erass1_image_catalog.fits` and `.csv`,
`out/13_final/uncertainty_calibration.json`, `out/17_orphans/ru_candidates.csv` and `.parquet`,
the three `out/16_healpix/` maps, and `out/run_manifest.json`, at the paths listed below. All other
products, including the per-tile FITS, PDF and HiPS comparison files in `out/14_tiles/`, are not
included; `make all` regenerates them. The pipeline code is not included; it is available at
https://github.com/doroshv/eropngcat. The input catalogues `eRASS1_Main.v1.2.fits` (eRASS1-DE DR1
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


## Accuracy (eRASS1-DE test split, held out from source-model fitting)

These tests are conditional on shared global image registration. The halo safeguards are fixed empirical rules derived from inspected fields and fits across both halves; these checks do not independently demonstrate improved neighbour recovery or diffuse-flux preservation.

- reliability (chance-corrected): **87.5%**
- detection limit, 0.2-2.3 keV point sources: 50% complete at **3.12e-13** erg/s/cm2; 90% is not reached (completeness plateaus at 87%: confusion, extended and flagged sources)
  (50% limit vs |b|: 0-10 deg 2.51e-13, 10-30 deg 3.17e-13, 30-60 deg 3.14e-13, 60-90 deg 3.79e-13)
- positions: median / 68% / 90% offset **72" / 105" / 176"**;
  `POS_ERR` model sigma interpolated in log S/N between binned validation values (4: 84", 4: 82", 4: 75", 5: 72", 6: 70", 6: 64", 8: 62", 9: 58", 13: 38", 22: 25"), saturated 20"; 40% / 79% of offsets within 1 / 2 sigma (39% / 86% expected)
- fluxes (log10 image / eRASS1):

| band | bias [dex] | robust scatter [dex] | >0.3 dex outliers | 16-84% interval coverage |
|---|---|---|---|---|
| 1 | -0.002 | 0.102 | 4.0% | 67% |
| P1 | +0.008 | 0.212 | 20.7% | 68% |
| P2 | +0.000 | 0.120 | 8.4% | 69% |
| P3 | -0.006 | 0.141 | 10.2% | 68% |

Full numbers: `out/10_validate_de/validation_report.md`.

### Global uncertainty systematics

Systematic terms were calibrated from 12731 one-to-one DE associations to point-like eRASS1 sources brighter than the validation-set image sensitivity limit (3.08e-13). Candidate base, additive, multiplicative and combined models were fitted on training blocks, selected by BIC on validation blocks, checked on downstream test blocks, and applied without an all-data refit.
The positional term is `additive`: final per-axis uncertainty is sqrt(POS_ERR_base^2 + (0.0000 POS_ERR_base)^2 + 22.63^2) arcsec.
Flux interval sides are enlarged in quadrature by the selected global term:

| band | selected term | additive [erg s$^{-1}$ cm$^{-2}$] | multiplicative fraction |
|---|---|---|---|
| 1 | multiplicative | 0.000e+00 | 0.0761 |
| P1 | multiplicative | 0.000e+00 | 0.0470 |
| P2 | base | 0.000e+00 | 0.0000 |
| P3 | multiplicative | 0.000e+00 | 0.0620 |

Full model comparisons and downstream test diagnostics: `out/13_final/uncertainty_calibration.json`.

### Final uncertainty coverage on the step-10 test sample

These samples use the same reference/matching cuts as the base-model validation above; no uncertainty-dependent selection is applied.
Positions: 4380 matched point sources; 45.1% / 81.8% within 1 / 2 combined per-axis sigmas (39.3% / 86.5% expected for a circular Gaussian).

| band | N | base 16-84% coverage | corrected 16-84% coverage |
|---|---|---|---|
| 1 | 3697 | 66.8% | 71.3% |
| P1 | 3471 | 67.5% | 68.1% |
| P2 | 3693 | 69.4% | 69.4% |
| P3 | 3680 | 67.8% | 69.9% |

## External validation

| catalogue | vs XMMSL3: detected / median sep / flux scatter | vs RASS BSC: detected / median sep / flux scatter |
|---|---|---|
| erass1_DE | 92% / 6" / 0.33 dex | 99% / 9" / 0.24 dex |
| image_DE | 79% / 27" / 0.27 dex | 90% / 28" / 0.22 dex |
| image_RU | 78% / 46" / 0.27 dex | 86% / 47" / 0.23 dex |

eRASS1 itself scatters by the amount in its row against these catalogues (variability over 5-30 years,
band conversion); the image catalogue adds its own scatter in quadrature.
RU - DE flux-scale offset: -0.052 +- 0.013 dex (4.1 sigma); **applied** to RU fluxes (factor 1.127, column `FLUX_SCALE_CORR`).
RU `POS_ERR` includes a 31.7" systematic in quadrature, from the RU/DE difference of median offsets to bright XMMSL3 and ROSAT BSC sources (column `POS_ERR_ADD`); the external catalogues' own errors cancel in that difference. The excess RU scatter is source-to-source rather than a correctable registration error: offsets of neighbouring RU sources share only ~8" per axis.
Image-only number counts give an RU flux offset of +0.044 dex (consistency check, not applied). Full numbers: `out/12_external/external_report.md`, `out/11_catalog_ru/consistency.json`.

### High-energy all-sky surveys (completeness; bands do not overlap, fluxes compared only as differences)

| survey | image DE | image RU | eRASS1 DE |
|---|---|---|---|
| maxi_gsc | 73% | 71% | 91% |
| rxte_xss | 81% | 76% | 90% |
| maxi_ssc | 74% | 80% | 92% |
| SRG/ART-XC artss1 (point-like) | 66% | 71% | 85% |
| SRG/ART-XC artss4 (point-like) | 69% | 74% | 90% |

SRG/ART-XC (contemporaneous, uniform over both halves): median offsets image DE / RU 30" / 47"; RU - DE flux scale from the 0.2-2.3 / 4-12 keV ratios +0.068 +- 0.043 dex (ARTSS4) and +0.080 +- 0.091 dex (ARTSS1); image - eRASS1 flux on the same ART-XC sources (DE test split) -0.012 +- 0.009 dex. Full numbers: `out/12_external/artxc_report.md`.

## Method

1. **Alignment** (`s02`, `s03`). The image is a Hammer-Aitoff projection in Galactic coordinates (Mollweide rejected: 802" vs 51" median offsets). **The two halves were projected separately**: a single projection leaves 599" median offsets on the RU side, a separate RU fit 94" (RASS BSC hold-out). The DE half is fitted on bright eRASS1 sources, the RU half on the RASS BSC; polynomial distortions do not improve hold-out residuals. Inverse projection is exact and tested at the poles and the l=180 seam; image patches are local gnomonic (TAN) reprojections, so nothing degenerates at the poles.
2. **Channel registration.** In the DE half the blue channel is shifted by (+1.04, +0.05) native pixels relative to red/green (RU: (+0.03, +0.01)); every channel is sampled at its own position.
3. **Tiling and hold-out** (`s04`): 10568 tiles of ~2x2 deg; DE tiles grouped into 10 deg blocks, randomly split 60/20/20 into train/validation/test.
4. **Supernova remnants** (SNRcat): sources inside remnants up to 5 deg across (mask radius 1.2 x radius + 6') are excluded from all calibration and validation; the 24-deg Antlia remnant is flagged only.
5. **Display stretch** (`s05`): inverse asinh, I = sinh((v - v0)/A) with A = 100, 138, 163 (R, G, B), chosen over linear, gamma and log stretches on held-out sources.
6. **PSF** (`s05`): Gaussian with covariance t2*I + s2*J^-1 J^-T in the TAN patch (sky sigma 4.6', plus a native-pixel term that elongates sources where the Hammer grid is sheared).
7. **Detection** (`s06`, `s07`): background-subtracted, noise-weighted linearised channels, matched filter, local maxima; saturated blobs are one detection each. Parameters tuned on train (search config: {'bkg_size': 11, 'kernel': ['iso', 0.75], 'min_sep': 3, 'stretch': {'A': [100.02981157217404, 138.29145208326997, 162.60289736146927], 'v0': -6.0}, 'weights': [0.5, 1, 1]}) with S/N threshold 3.5, selected on validation.
8. **Measurement and models** (`s08`, `s09`): positions from `blend` (per-S/N-bin choice among 8 estimators, each optionally with a per-half sky-frame astrometric correction (+sky; +skybin adds a per-bin offset), selected on validation; bins: 0: cen+skybin, 1: cen+sky, 2: cen+sky, 3: cen+sky, 4: psfnw+sky, 5: psfnw+sky, 6: psfnw+skybin, 7: psfn+skybin); fluxes from `hgb` (best of hgb, xgb, param_psf_native, ridge, param_aperture, param_psf_scene) using PSF amplitudes, apertures, saturated-blob area, background and geometry features; quantile boosting supplies the base interval and the global DE systematic calibration above is added in quadrature to give `FLUX_*_LO/HI`.

## Caveats and identified issues

- The image is a rendering: 8-bit, stretched, smoothed (PSF sigma ~5'), and saturated for bright sources (2759 saturated sources). Saturated fluxes rely on blob size and wings; see the saturated scatter in the validation report.
- Confusion: at ~5' resolution, image sources are often blends of several eRASS1 sources. The flux of a source is calibrated against its brightest counterpart, so blends are biased high; positions follow the light centroid.
- Eddington bias: near the detection limit image fluxes of detected sources are biased high.
- Coverage: 188 deg2 (0.46%) along the image outline (around l = 180 deg and the poles) is cropped from the image and has no sources; 455 sources near the outline or the join between the halves are flagged `FLAG_EDGE`.
- The RU half is calibrated with DE models only. Its projection, channel registration and stretch were checked (RASS BSC, XMMSL3, number counts), but RU-specific rendering differences smaller than these checks cannot be excluded.
- Extended emission: 9849 sources have broad profiles (`FLAG_EXTENDED`); 1665 lie inside SNRcat remnants (`SNR_NAME`). Clusters, SNR knots, the Magellanic Clouds and diffuse clumps can appear as sources.
- The eRASS1-DE / RU footprint boundary is 179.944 / 359.944 deg, while the image halves join at l = 0 / 180 deg; sources in the 0.056-deg sliver are labelled by the footprint (`HALF`).
- DE entries in the train/validation splits are in-sample for the models (`SPLIT`); quote test-split accuracies.
- Fluxes assume eRASS1 spectral mix; sources with unusual spectra (very soft/hard) have larger band-flux errors.
- Band-flux validation covers only sources eRASS1 detected in that band: its `ML_FLUX_P1` is exactly zero for 28% of the catalogue (P2 6%, P3 9%), and those rows drop out of the P1/P2/P3 rows above. The quoted scatter and coverage therefore describe detected-in-band sources, not the band as a whole.

## Columns

| column | unit | description |
|---|---|---|
| NAME |  | source name (Galactic position) |
| RA | deg | right ascension ICRS |
| DEC | deg | declination ICRS |
| LII | deg | Galactic longitude |
| BII | deg | Galactic latitude |
| POS_ERR | arcsec | 1-sigma positional error per axis, including the global eRASS1-DE systematic and POS_ERR_ADD |
| FLUX_1 | erg/s/cm2 | 0.2-2.3 keV flux (image model) |
| FLUX_1_LO | erg/s/cm2 | 0.2-2.3 keV flux (image model) (lower calibrated prediction bound including global systematic) |
| FLUX_1_HI | erg/s/cm2 | 0.2-2.3 keV flux (image model) (upper calibrated prediction bound including global systematic) |
| FLUX_P1 | erg/s/cm2 | 0.2-0.5 keV flux (image model) |
| FLUX_P1_LO | erg/s/cm2 | 0.2-0.5 keV flux (image model) (lower calibrated prediction bound including global systematic) |
| FLUX_P1_HI | erg/s/cm2 | 0.2-0.5 keV flux (image model) (upper calibrated prediction bound including global systematic) |
| FLUX_P2 | erg/s/cm2 | 0.5-1.0 keV flux (image model) |
| FLUX_P2_LO | erg/s/cm2 | 0.5-1.0 keV flux (image model) (lower calibrated prediction bound including global systematic) |
| FLUX_P2_HI | erg/s/cm2 | 0.5-1.0 keV flux (image model) (upper calibrated prediction bound including global systematic) |
| FLUX_P3 | erg/s/cm2 | 1.0-2.0 keV flux (image model) |
| FLUX_P3_LO | erg/s/cm2 | 1.0-2.0 keV flux (image model) (lower calibrated prediction bound including global systematic) |
| FLUX_P3_HI | erg/s/cm2 | 1.0-2.0 keV flux (image model) (upper calibrated prediction bound including global systematic) |
| SNR |  | detection significance (matched filter); inf for saturated blobs |
| SATURATED |  | source contains saturated image pixels |
| SAT_AREA_ARCMIN2 | arcmin2 | area of the saturated blob |
| PSF_WIDTH |  | free-width PSF fit scale (1 = point source) |
| FLAG_EXTENDED |  | PSF_WIDTH > 1.5 or SAT_AREA_ARCMIN2 > 500: possibly extended |
| FLAG_POS_FALLBACK |  | PSF fit failed, detection position used |
| SNR_NAME |  | inside the mask of this SNRcat supernova remnant |
| HALF |  | DE (eRASS1-DE footprint) or RU |
| TILE |  | 2x2 deg tile id |
| SPLIT |  | DE: train/val/test block of the hold-out scheme; RU: ru |
| FLUX_SCALE_CORR |  | multiplicative flux correction applied (RU) |
| POS_ERR_ADD |  | additive RU positional systematic, applied in quadrature (RU) |
| FLAG_EDGE |  | near the image outline or the join of the two halves |
| ERASS1_NAME |  | brightest eRASS1 source within 4 arcmin (F >= 1e-13), DE only |
| ERASS1_SEP | arcsec | separation to ERASS1_NAME |
| ERASS1_FLUX_1 |  | eRASS1 0.2-2.3 keV flux of ERASS1_NAME |
