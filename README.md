# epngcat

Pipeline that extracts an X-ray point-source catalogue (ePNGCAT) from the SRG/eROSITA eRASS1 all-sky RGB
press-kit image <https://www.mpe.mpg.de/7461950/erass1-presskit>. Positions and fluxes are calibrated and validated against the eRASS1-DE Main
catalogue v 1.2 (western Galactic half, l = 180-360°) and then applied to the whole sky, including the eastern (RU) half for which no catalogue is published as of now. For western part of the sky the proper eRASS1 (or eRASS:3) catalogue is obviously preferred.

The catalogue, HEALPix maps and the input image are released separately on Zenodo:
<https://zenodo.org/records/XXXXXXX>. This repository holds only the code, input data must be downloaded separately.

## Quick start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Every Make stage runs through
   `uv run --frozen`, which creates the Python environment from `uv.lock` on first use.
2. Place the inputs in the repository root:
   - `erass1_allsky.png`: the eRASS1 press image, from the Zenodo record or the
     [MPE/IKI press kit](https://www.mpe.mpg.de/7461950/erass1-presskit) (19 June 2020; credit Jeremy
     Sanders, Hermann Brunner and the eSASS team (MPE); Eugene Churazov, Marat Gilfanov (IKI)).
   - `eRASS1_Main.v1.2.fits` ([eRASS1 DR1](https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/)),
     `xmmsl3_v1.0.fits` ([ESASky](https://www.cosmos.esa.int/web/esdc/esasky-catalogues)) and
     `5XMM_DR15cat_v1.0.fits` ([5XMM-DR15](https://xmmssc.irap.omp.eu/Catalogue/5XMM-DR15/5XMM_DR15.html)).

   The remaining reference catalogues and the eRASS1 HiPS maps are downloaded automatically.
3. `make all`. A full run takes several hours and downloads about 25 000 HiPS tiles per band.
   Set `EPNGCAT_WORKERS` to limit the number of worker processes (each needs ~0.5 GB of RAM; default 6).

Other targets: `make test` (unit tests), `make lint` (ruff), `make replot` (redraw diagnostics from
saved products without refitting), `make clean` (remove `out/`).

`make` reruns a stage only when its inputs or the code it imports change (see `tools/stage_deps.py`).
`out/run_manifest.json` records code and input checksums and the locked environment of a completed run.

## Pipeline

Each step is a module `epngcat/sNN_*.py` that writes to `out/NN_*/`. The DE (western) half is divided into
~2×2° tiles grouped in ~10° blocks, which are randomly split 60/20/20 into training, validation and test
regions. Models are tuned on training blocks, selected on validation blocks, and evaluated on test blocks.

| Step | Purpose |
|---|---|
| 01, 01b | Prepare input catalogues; fetch ROSAT and high-energy reference catalogues |
| 02, 03 | Coarse and refined image↔sky registration (one Hammer–Aitoff projection per half) |
| 04 | Sky tiles, image coverage and the block-wise train/val/test split |
| 05 | Calibrate the display stretch and the PSF on training sources |
| 06, 07, 07b | Baseline detection; detection tuning (train → val); scene-model PSF width |
| 08 | Detection and measurement on all DE tiles |
| 09 | Select position and flux models (train → val) |
| 10 | Validation on the held-out test blocks |
| 11 | Catalogue of the RU half and DE/RU consistency checks |
| 12 | External validation: ROSAT (BSC, 2RXS), XMMSL3, 5XMM-DR15, SRG/ART-XC |
| 13 | Final all-sky catalogue (FITS, CSV) and its README |
| 14 | Per-tile products: data, model, background, residual FITS and previews |
| 15 | Surface-brightness calibration against eRASS1 HiPS rate maps |
| 16 | All-sky HEALPix maps from the tile products |
| 17 | Sources without counterparts in other X-ray catalogues |
| 18 | Publication figures |

The generated catalogue documentation (`CATALOG_README.md`, written by step 13) describes every
product, the validation results and all catalogue columns. You don't need to download zenodo record if you intend to reproduce results, those are provided for convinience. The run time of the full pipeline is a few hours on MBP M1 pro, shall be comparable on pc. 

## Data release

`tools/make_release.sh` packs the released data products (catalogue, uncertainty calibration, RU
candidate list, HEALPix maps, run manifest, input image and `CATALOG_README.md`) into a tarball for
Zenodo after a completed `make all`. It contains no code.

## Licence

Code: MIT (`LICENSE`). The released catalogue and maps are CC0 1.0. The press image is not part of
this repository and is subject to the press-kit terms. If you use the code or catalogue, please cite
the accompanying paper.
