"""Step 06: baseline source detection over ~2x2 deg tangent-plane patches of the DE half.

Raw luminance (R+G+B), median-filter background, Gaussian matched filter, local maxima above
5 sigma. Two kernels are compared: isotropic, and the calibrated TAN-patch PSF including the
native-pixel (Jacobian) term from step 05. Evaluated per split against eRASS1-DE
(completeness vs flux, reliability); known SNR regions are excluded from the evaluation.

Outputs (out/06_detect_baseline/): detections_<variant>.parquet, summary.json, completeness_<variant>.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epngcat.config import OUT, load_json, save_json, step_dir
from epngcat.detect import DEFAULTS, detect_tiles
from epngcat.evaluate import catalog, clean, completeness, flux_at_completeness, match_detections, reliability
from epngcat.plotting import figure_size, save_figure

OUTDIR = step_dir("06_detect_baseline")
FLUX_BINS = np.geomspace(1e-14, 1e-10, 25)


def evaluate_split(det, split):
    """Summary numbers of a detection list for one split."""
    cat = catalog(split)
    d = clean(det[det.split == split])
    rel, fmatch, pc = reliability(d, cat)
    comp, n = completeness(d, cat, FLUX_BINS)
    m = match_detections(d, cat)
    return {
        "n_det": len(d),
        "reliability": rel,
        "match_frac": fmatch,
        "chance_frac": pc,
        "F50": flux_at_completeness(0.5, comp, FLUX_BINS, n),
        "F90": flux_at_completeness(0.9, comp, FLUX_BINS, n),
        "median_sep_arcsec": float(np.nanmedian(m.sep)),
        "completeness": comp.tolist(),
        "n_cat": n.tolist(),
    }


def plot_completeness(summary, path, title):
    """Completeness curves per split."""
    fig, ax = plt.subplots(figsize=figure_size())
    for split, s in summary.items():
        n = np.asarray(s["n_cat"])
        ax.stairs(np.where(n > 0, np.asarray(s["completeness"]), np.nan), FLUX_BINS, baseline=None, label=split)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{\mathrm{eRASS1}}$ [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel("completeness (point sources)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_figure(fig, path)


def main():
    """Run step 06."""
    tiles = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    tiles = tiles[tiles.de]
    psf = load_json(OUT / "05_calibrate" / "calibration.json")["psf"]
    variants = {"isotropic": {}, "tan_jacobian": {"psf": {"t2": psf["t2"], "s2": psf["s2"]}}}
    out = {}
    for name, params in variants.items():
        det = detect_tiles(tiles, params)
        det["split"] = tiles.set_index("tile").split.reindex(det.tile).values
        det.to_parquet(OUTDIR / f"detections_{name}.parquet")
        summary = {s: evaluate_split(det, s) for s in ("train", "val", "test")}
        out[name] = {"params": {**DEFAULTS, **params}, "splits": summary}
        for s, v in summary.items():
            print(
                name, s, {k: v[k] for k in ("n_det", "reliability", "chance_frac", "F50", "F90", "median_sep_arcsec")}
            )
        plot_completeness(summary, OUTDIR / f"completeness_{name}.pdf", f"baseline detection, raw luminance, {name}")
    save_json(out, OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
