"""Step 07: optimise the detection on the training split, select on the validation split.

Each configuration is run once with a low threshold; because local maxima do not depend on the
threshold, the S/N cut is chosen afterwards. Reliability plateaus at ~0.93-0.97 even for bright
detections (diffuse clumps, offset centroids), so a fixed reliability target is ill-posed.
Objective instead: utility U = N_real - LAMBDA * N_spurious, maximised over the cut
(equivalently: keep detections whose marginal reliability is >= LAMBDA / (1 + LAMBDA) = 0.8).
N_real and N_spurious use the chance-corrected reliability.

Search: coordinate descent over the parameters below (two passes) on training tiles; the best
TOP_K configurations are then re-scored on validation tiles and the winner is stored.

Outputs (out/07_detect_optimize/): detection_params.json, trials.csv, completeness_val.pdf,
trials_cache.jsonl (resume cache keyed by configuration and calibration; delete to recompute)
"""

import json

import numpy as np
import pandas as pd

from epngcat.config import IMAGE, OUT, ROOT, load_json, save_json, step_dir
from epngcat.detect import DEFAULTS, detect_tiles
from epngcat.evaluate import (
    F_FLOOR,
    MATCH_R,
    brightest_within,
    catalog,
    clean,
    completeness,
    flux_at_completeness,
    shifted,
)
from epngcat.provenance import fingerprint
from epngcat.s06_detect_baseline import FLUX_BINS, plot_completeness

OUTDIR = step_dir("07_detect_optimize")
LAMBDA = 4.0
LOW_THRESH = 3.5
TOP_K = 5
# Adopted S/N cut, overriding the one LAMBDA selects (4.25). The validation utility is flat over
# 3.5-4.25 (3432 vs 3631, 5%) while the depth is not: 3.5 gives +1442 detections, +1114 of them
# real, F50 3.76e-13 -> 3.24e-13 (14% deeper), for reliability 0.942 -> 0.904. The shell it adds is
# 74% real. Set to None to follow LAMBDA (equivalent to LAMBDA = 2.75 on the validation split).
THRESH = 3.5


def search_space(calib):
    """Candidate values per parameter."""
    psf = calib["psf"]
    return {
        "stretch": [None, calib["stretch"]],
        "weights": [[1, 1, 1], [0.5, 1, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 0.5]],
        "kernel": [
            ("iso", 0.75),
            ("iso", 1.0),
            ("iso", 1.5),
            ("iso", 2.0),
            ("iso", 2.5),
            ("psf", 0.25),
            ("psf", 0.5),
            ("psf", 0.75),
            ("psf", 1.0),
        ],
        "bkg_size": [11, 15, 21, 31, 45],
        "min_sep": [1, 2, 3],
        "_psf": psf,
    }


def to_params(cfg, psf):
    """Search configuration -> detect_patch parameters."""
    kind, v = cfg["kernel"]
    out = {k: cfg[k] for k in ("stretch", "weights", "bkg_size", "min_sep")}
    out["thresh"] = LOW_THRESH
    if kind == "iso":
        out["kernel_sigma"], out["psf"] = v, None
    else:
        out["psf"] = {"t2": psf["t2"] * v, "s2": psf["s2"] * v}
    return out


def score(det, cat):
    """S/N cut maximising the utility and the resulting numbers (SNR regions excluded)."""
    det = clean(det)
    empty = {"thresh": np.nan, "n": 0, "n_real": 0.0, "reliability": np.nan, "utility": 0.0}
    if len(det) == 0:
        return empty
    real = brightest_within(det.l.values, det.b.values, cat, MATCH_R, F_FLOOR) >= 0
    fake = brightest_within(*shifted(det.l.values, det.b.values), cat, MATCH_R, F_FLOOR) >= 0
    snr = np.where(det.sat_area.values > 0, np.inf, det.snr.values)
    best = empty
    for thr in np.arange(LOW_THRESH, 15.01, 0.25):
        m = snr >= thr
        if m.sum() < 50:
            break
        f, pc = real[m].mean(), fake[m].mean()
        rel = (f - pc) / (1 - pc)
        n_real = rel * m.sum()
        util = n_real - LAMBDA * (m.sum() - n_real)
        if util > best["utility"]:
            best = {
                "thresh": float(thr),
                "n": int(m.sum()),
                "n_real": float(n_real),
                "reliability": float(rel),
                "utility": float(util),
            }
    return best


def evaluate(cfg, psf, tiles, cat):
    """Run detection for a configuration and score it."""
    det = detect_tiles(tiles, to_params(cfg, psf))
    return score(det, cat), det


def key(cfg):
    """Hashable identity of a configuration."""
    return json.dumps(cfg, sort_keys=True)


def main():
    """Run step 07."""
    calib = load_json(OUT / "05_calibrate" / "calibration.json")
    space = search_space(calib)
    psf = space.pop("_psf")
    tiles = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    tr_tiles, va_tiles = tiles[tiles.split == "train"], tiles[tiles.split == "val"]
    cat_tr, cat_va = catalog("train"), catalog("val")

    cfg = {
        "stretch": None,
        "weights": [1, 1, 1],
        "kernel": ("iso", DEFAULTS["kernel_sigma"]),
        "bkg_size": DEFAULTS["bkg_size"],
        "min_sep": DEFAULTS["min_sep"],
    }
    cache_path = OUTDIR / "trials_cache.jsonl"
    # Resume only when the code, environment, sky split, alignment and data agree.
    inputs = [
        *sorted((ROOT / "epngcat").glob("*.py")),
        ROOT / "uv.lock",
        IMAGE,
        OUT / "03_align" / "alignment.json",
        OUT / "04_tiles" / "tiles.parquet",
        OUT / "04_tiles" / "erass1_tiles.parquet",
        OUT / "01_catalogs" / "erass1.parquet",
        OUT / "01_catalogs" / "snrcat.parquet",
    ]
    stamp = key(calib) + fingerprint(inputs)
    trials = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.pop("stamp") == stamp:
                rec["cfg"]["kernel"] = tuple(rec["cfg"]["kernel"])
                trials[key(rec["cfg"])] = rec

    def run(c):
        k = key(c)
        if k not in trials:
            trials[k] = {"cfg": c, **evaluate(c, psf, tr_tiles, cat_tr)[0]}
            with cache_path.open("a") as fh:
                fh.write(json.dumps({"stamp": stamp, **trials[k]}) + "\n")
            t = trials[k]
            print(
                round(t["utility"]),
                round(t["n_real"]),
                round(t["reliability"], 3),
                t["thresh"],
                {p: c[p] for p in space},
                flush=True,
            )
        return trials[k]["utility"]

    run(cfg)
    for _ in range(2):
        for name, values in space.items():
            scores = {i: run({**cfg, name: v}) for i, v in enumerate(values)}
            cfg = {**cfg, name: values[max(scores, key=scores.get)]}

    table = pd.DataFrame(
        [
            {**{k: json.dumps(v) for k, v in t["cfg"].items()}, **{k: v for k, v in t.items() if k != "cfg"}}
            for t in trials.values()
        ]
    ).sort_values("utility", ascending=False)
    top = sorted(trials.values(), key=lambda t: -t["utility"])[:TOP_K]
    val = []
    for t in top:
        det = detect_tiles(va_tiles, to_params(t["cfg"], psf))
        s = score(det, cat_va)
        val.append((s, t, det))
        print("val", round(s["utility"]), round(s["n_real"]), s["thresh"], t["cfg"])
    s_best, t_best, det_best = max(val, key=lambda v: v[0]["utility"])
    table["val_utility"] = table.apply(
        lambda r: next(
            (v[0]["utility"] for v in val if key(v[1]["cfg"]) == key({k: json.loads(r[k]) for k in cfg})), np.nan
        ),
        axis=1,
    )
    table.to_csv(OUTDIR / "trials.csv", index=False)

    params = {**to_params(t_best["cfg"], psf), "thresh": THRESH if THRESH is not None else s_best["thresh"]}
    det_best = clean(det_best[(det_best.snr >= params["thresh"]) | (det_best.sat_area > 0)])
    comp, n = completeness(det_best, cat_va, FLUX_BINS)
    base = load_json(OUT / "06_detect_baseline" / "summary.json")["isotropic"]["splits"]["val"]
    base_det = pd.read_parquet(OUT / "06_detect_baseline" / "detections_isotropic.parquet")
    base_score = score(base_det[base_det.split == "val"], cat_va)
    summary = {
        "params": params,
        "thresh_lambda_optimal": s_best["thresh"],
        "search_config": t_best["cfg"],
        "train": {k: t_best[k] for k in ("thresh", "n", "n_real", "reliability", "utility")},
        "val": s_best,
        "val_F50": flux_at_completeness(0.5, comp, FLUX_BINS, n),
        "val_F90": flux_at_completeness(0.9, comp, FLUX_BINS, n),
        "baseline_val_same_objective": base_score,
        "baseline_val_F50_at_5sigma": base["F50"],
        "n_trials": len(trials),
    }
    save_json(summary, OUTDIR / "detection_params.json")
    print(summary)
    plot_completeness(
        {
            "optimised (val)": {
                "n_det": len(det_best),
                "reliability": s_best["reliability"],
                "F50": summary["val_F50"],
                "completeness": comp.tolist(),
                "n_cat": n.tolist(),
            },
            "baseline (val)": base,
        },
        OUTDIR / "completeness_val.pdf",
        "detection: optimised vs baseline (validation split)",
    )


if __name__ == "__main__":
    main()
