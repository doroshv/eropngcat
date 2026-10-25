"""Re-render saved diagnostics without refitting detection or calibration models.

Run with ``python -m epngcat.replot``. Uses the current saved analysis products;
this is a styling preview, not a substitute for ``make all`` after analysis edits.
"""

import pickle

import numpy as np
import pandas as pd

from epngcat.config import load_json


def main():
    """Render the diagnostics recoverable from saved intermediate products."""
    from epngcat import s03_align_refine as align
    from epngcat import s05_calibrate as calibrate
    from epngcat import s06_detect_baseline as baseline
    from epngcat import s09_models as models
    from epngcat import s10_validate_de as validation
    from epngcat import s11_catalog_ru as ru
    from epngcat import s12_external as external
    from epngcat import s12_flux_comparison as flux_comparison
    from epngcat import s12_rosat as rosat
    from epngcat import s12_xmm as xmm
    from epngcat import s17_orphans as orphans
    from epngcat import s18_figures as paper
    from epngcat.image import load_rgb
    from epngcat.projection import SplitProjection, load_alignment

    rgb = load_rgb()
    bsc = pd.read_parquet(align.OUTDIR / "align_sources_bsc.parquet")
    projection = load_alignment()
    align.plot_results(
        rgb[..., 0] + rgb[..., 1], bsc[~bsc.train], SplitProjection(projection.de, projection.de), projection
    )
    del rgb
    print("rendered alignment", flush=True)

    src = pd.read_parquet(calibrate.OUTDIR / "sources.parquet")
    cal = load_json(calibrate.OUTDIR / "calibration.json")
    calibrate.plot_results(
        src, cal["stretch_family"], cal["stretch_families"], src[src.cxx.notna()], cal["psf"]["s2"], cal["psf"]["t2"]
    )
    summary = load_json(baseline.OUTDIR / "summary.json")
    for name, values in summary.items():
        baseline.plot_completeness(values["splits"], baseline.OUTDIR / f"completeness_{name}.pdf", name)
    print("rendered calibration and baseline", flush=True)

    m = pd.read_parquet(models.OUTDIR / "de_matched.parquet")
    va = m[(m.cat_EXT == 0) & (m.sep_fin < models.TRAIN_SEP) & (m.split == "val")]
    selection = load_json(models.OUTDIR / "selection.json")
    with (models.OUTDIR / "models.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    best = selection["flux"]["model"]
    table = pd.read_csv(models.OUTDIR / "flux_models.csv", dtype={"band": str})
    models.plot_results(
        m,
        va,
        selection["position"]["error_model"],
        {b: {best: fit} for b, fit in saved["flux"].items()},
        best,
        {b: table[table.band == b] for b in models.BANDS},
    )
    ru.plot_results(load_json(ru.OUTDIR / "consistency.json"))
    external.replot()
    rosat.replot()
    flux_comparison.main()
    cat, reference = xmm.load()
    pairs, chance = xmm.match(cat, reference)
    xmm.plot_results(pairs, chance)
    saved_cat = pd.read_parquet(validation.OUTDIR / "de_catalog.parquet")
    test = saved_cat[(saved_cat.SPLIT == "test") & ~saved_cat.IN_SNR_MASK].reset_index(drop=True)
    matched = validation.match_detections(test.rename(columns={"LII": "l", "BII": "b"}), validation.catalog("test"))
    pt = matched[(matched.cat_EXT == 0) & matched.cat_ML_FLUX_1.notna()].copy()
    pt["sep"] = validation.sep_arcsec(pt.l, pt.b, pt.cat_LII, pt.cat_BII)
    ratio = pt.sep / np.sqrt(pt.POS_ERR**2 + pt.cat_POS_ERR.fillna(0) ** 2 / 2)
    metrics = load_json(validation.OUTDIR / "metrics.json")
    validation.plot_results(metrics, ratio, pt, pt.sep < 150, metrics["fluxes"])
    cal = load_json(orphans.OUTDIR / "orphans.json")["calibration"]
    candidates = pd.read_parquet(orphans.OUTDIR / "ru_candidates.parquet")
    orphans.plot_calibration(None, cal, cal["ru_prevalence"], cal["validation"], candidates, cal["snr_bin_edges"])
    print("rendered models, consistency and XMM", flush=True)

    for render in (
        paper.fig_input,
        paper.fig_skymap,
        paper.fig_lognlogs,
        paper.fig_healpix,
        paper.fig_bright_residuals,
    ):
        render()
        print("rendered", render.__name__, flush=True)


if __name__ == "__main__":
    main()
