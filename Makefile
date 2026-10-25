# Reproducible stage DAG; stamps are written only after successful completion.
PY := uv run --frozen python -m epngcat
BUILD_PYTHON ?= python3
# Each stage depends on its own module and the epngcat modules it imports (transitively, including
# imports inside functions; see tools/stage_deps.py), plus the environment and build files.
CODE := pyproject.toml uv.lock Makefile tools/check_tile_products.py
out/.deps.mk: $(wildcard epngcat/*.py) tools/stage_deps.py
	@mkdir -p out
	$(BUILD_PYTHON) tools/stage_deps.py > $@
include out/.deps.mk
# The run manifest hashes all code.
DEPS_provenance := $(wildcard epngcat/*.py tools/*.py)
# Stages already use multiprocessing; avoid multiplying their memory footprint.
.NOTPARALLEL:
.DELETE_ON_ERROR:
.PHONY: all external-validation figures replot test lint clean
all: external-validation out/17_orphans/orphans.json out/18_figures/fig01_input_image.pdf out/run_manifest.json
external-validation: out/12_external/external.json out/12_external/artxc.json out/12_external/xmm_flux.json out/12_external/rosat_2rxs.json out/12_external/flux_comparison.pdf
figures: out/18_figures/fig01_input_image.pdf out/12_external/flux_comparison.pdf

# $(1): module; $(2): all products; $(3): input products; $(4): CLI options.
# Missing secondary products invalidate their producer, even when its primary exists.
ifneq ($(shell $(BUILD_PYTHON) tools/check_tile_products.py),)
.PHONY: out/.stamps/s14_tile_products
endif

define stage
$(if $(filter-out $(wildcard $(2)),$(2)),.PHONY: out/.stamps/$(1))
out/.stamps/$(1): $(CODE) $(DEPS_$(1)) $(3)
	@mkdir -p out/.stamps
	@rm -f $$@
	$(PY).$(1) $(4)
	@$(foreach product,$(2),test -f $(product) && )touch $$@
$(2): out/.stamps/$(1)
	@test -f $$@
endef

$(eval $(call stage,s01_prepare_catalogs,\
  out/01_catalogs/erass1.parquet \
  out/01_catalogs/rass_bsc.parquet \
  out/01_catalogs/snrcat.parquet \
  out/01_catalogs/xmm_dr15.parquet \
  out/01_catalogs/xmmsl3.parquet,\
  eRASS1_Main.v1.2.fits xmmsl3_v1.0.fits $(wildcard data/xmm_dr15_bright.parquet data/rass_bsc_IX10A_1rxs.fits data/snrcat.csv),))

$(eval $(call stage,s02_align_coarse,\
  out/02_align_coarse/alignment_coarse.json \
  out/02_align_coarse/outline.pdf,\
  erass1_allsky.png,))

$(eval $(call stage,s03_align_refine,\
  out/03_align/align_sources_bsc.parquet \
  out/03_align/align_sources_de.parquet \
  out/03_align/alignment.json \
  out/03_align/residuals.pdf,\
  out/01_catalogs/erass1.parquet out/02_align_coarse/alignment_coarse.json,))

$(eval $(call stage,s04_tiles,\
  out/04_tiles/coverage.pdf \
  out/04_tiles/erass1_tiles.parquet \
  out/04_tiles/split.pdf \
  out/04_tiles/summary.json \
  out/04_tiles/tiles.parquet,\
  out/03_align/alignment.json,))

$(eval $(call stage,s05_calibrate,\
  out/05_calibrate/calibration.json \
  out/05_calibrate/psf.pdf \
  out/05_calibrate/sources.parquet \
  out/05_calibrate/stretch.pdf,\
  out/04_tiles/tiles.parquet,))

$(eval $(call stage,s06_detect_baseline,\
  out/06_detect_baseline/completeness_isotropic.pdf \
  out/06_detect_baseline/completeness_tan_jacobian.pdf \
  out/06_detect_baseline/detections_isotropic.parquet \
  out/06_detect_baseline/detections_tan_jacobian.parquet \
  out/06_detect_baseline/summary.json,\
  out/05_calibrate/calibration.json,))

$(eval $(call stage,s07_detect_optimize,\
  out/07_detect_optimize/completeness_val.pdf \
  out/07_detect_optimize/detection_params.json \
  out/07_detect_optimize/trials.csv,\
  out/06_detect_baseline/summary.json,))

$(eval $(call stage,s07b_psf_scene,\
  out/07b_psf_scene/psf_scene.json,\
  out/07_detect_optimize/detection_params.json,))

$(eval $(call stage,s08_de_pipeline,\
  out/08_de_pipeline/de_pipeline.parquet,\
  out/07b_psf_scene/psf_scene.json,))

$(eval $(call stage,s09_models,\
  out/09_models/de_matched.parquet \
  out/09_models/flux_models.csv \
  out/09_models/fluxes.pdf \
  out/09_models/models.pkl \
  out/09_models/position_models.csv \
  out/09_models/positions.pdf \
  out/09_models/selection.json,\
  out/08_de_pipeline/de_pipeline.parquet,))

$(eval $(call stage,s10_validate_de,\
  out/10_validate_de/de_catalog.parquet \
  out/10_validate_de/detection_positions.pdf \
  out/10_validate_de/fluxes.pdf \
  out/10_validate_de/metrics.json \
  out/10_validate_de/validation_report.md \
  out/10_validate_de/reliability.pdf \
  out/10_validate_de/position_calibration.pdf,\
  out/09_models/selection.json,))

$(eval $(call stage,s11_catalog_ru,\
  out/11_catalog_ru/consistency.json \
  out/11_catalog_ru/consistency.pdf \
  out/11_catalog_ru/ru_catalog.parquet \
  out/11_catalog_ru/ru_pipeline.parquet,\
  out/10_validate_de/metrics.json,))

$(eval $(call stage,s01b_hard_catalogs,\
  out/01_catalogs/maxi_gsc.parquet \
  out/01_catalogs/maxi_ssc.parquet \
  out/01_catalogs/rxte_xss.parquet,\
  $(wildcard data/heasarc_*.parquet),))

$(eval $(call stage,s12_external,\
  out/12_external/external.json \
  out/12_external/external.pdf \
  out/12_external/external_report.md \
  out/12_external/hard.json \
  out/12_external/hard.pdf,\
  out/11_catalog_ru/consistency.json out/01_catalogs/maxi_gsc.parquet,))

$(eval $(call stage,s12_artxc,\
  out/12_external/artxc.json \
  out/12_external/artxc.pdf \
  out/12_external/artxc_report.md,\
  out/11_catalog_ru/consistency.json $(wildcard data/artxc_*.parquet),))

$(eval $(call stage,s13_final_catalog,\
  out/13_final/README.md \
  out/13_final/erass1_image_catalog.csv \
  out/13_final/erass1_image_catalog.fits \
  out/13_final/uncertainty_calibration.json,\
  out/12_external/external.json out/12_external/artxc.json,))

$(eval $(call stage,s12_xmm,\
  out/12_external/xmm_flux.json \
  out/12_external/xmm_flux.pdf \
  out/12_external/xmm_flux_ratio.pdf,\
  out/13_final/erass1_image_catalog.fits,))

$(eval $(call stage,s12_rosat,\
  out/12_external/rosat_2rxs.json \
  out/12_external/rosat_2rxs.pdf \
  out/12_external/rosat_2rxs_flux.pdf \
  out/12_external/rosat_2rxs_ratio.pdf \
  out/12_external/rosat_2rxs_report.md,\
  out/13_final/erass1_image_catalog.fits $(wildcard data/rass_2rxs.parquet),))

$(eval $(call stage,s12_flux_comparison,\
  out/12_external/flux_comparison.pdf \
  out/12_external/flux_comparison_pairs.csv,\
  out/13_final/erass1_image_catalog.csv out/10_validate_de/de_catalog.parquet out/04_tiles/erass1_tiles.parquet out/01_catalogs/erass1.parquet out/01_catalogs/xmmsl3.parquet $(wildcard data/rass_2rxs.parquet),))

$(eval $(call stage,s14_tile_products,\
  out/14_tiles/index.csv,\
  out/13_final/erass1_image_catalog.fits,--hires))

$(eval $(call stage,s15_units,\
  out/15_units/units.json \
  out/15_units/units.pdf,\
  out/14_tiles/index.csv,))

$(eval $(call stage,s16_healpix,\
  out/16_healpix/healpix_clean_filled_nside1024.fits \
  out/16_healpix/healpix_clean_nside1024.fits \
  out/16_healpix/healpix_data_nside1024.fits,\
  out/15_units/units.json,))

$(eval $(call stage,s17_orphans,\
  out/17_orphans/orphans.json \
  out/17_orphans/orphans.parquet \
  out/17_orphans/orphans.pdf \
  out/17_orphans/orphans_report.md \
  out/17_orphans/ru_calibration.pdf \
  out/17_orphans/ru_candidates.csv \
  out/17_orphans/ru_candidates.parquet,\
  out/12_external/rosat_2rxs.json out/12_external/artxc.json,))

$(eval $(call stage,s18_figures,\
  out/18_figures/fig01_input_image.pdf \
  out/18_figures/fig09_sky_distribution.pdf \
  out/18_figures/fig11_lognlogs.pdf \
  out/18_figures/figA7_healpix_data.pdf \
  out/18_figures/figA7_healpix_clean.pdf \
  out/18_figures/figA8_bright_residuals.pdf \
  out/18_figures/fig01_tile_grid.pdf \
  out/18_figures/fig01_sky_split.pdf \
  out/18_figures/fig09_surface_density.pdf \
  out/18_figures/fig11_count_ratios.pdf \
  out/18_figures/number_counts.json \
  out/18_figures/number_counts.csv,\
  out/16_healpix/healpix_clean_nside1024.fits out/12_external/rosat_2rxs.json out/16_healpix/healpix_clean_filled_nside1024.fits out/16_healpix/healpix_data_nside1024.fits,))

$(eval $(call stage,provenance,\
  out/run_manifest.json,\
  out/18_figures/fig01_input_image.pdf out/17_orphans/orphans.json out/12_external/flux_comparison.pdf out/12_external/xmm_flux.json $(wildcard data/*),))

replot:
	$(PY).replot

test:
	uv run --frozen python -m pytest -q

lint:
	uv run --frozen python -m ruff check epngcat tests tools --no-fix
	uv run --frozen python -m ruff format --check epngcat tests tools

clean:
	rm -rf out
