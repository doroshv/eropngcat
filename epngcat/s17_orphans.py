"""Step 17: image sources with no counterpart in any other X-ray catalogue, and what they are.

An "orphan" is a final-catalogue source with no external-catalogue source within K_MATCH combined
positional errors (image POS_ERR and the empirically calibrated external error, see external_sigma).
Two flags are computed, because the two halves have different references available:

  matched_ext  all-sky catalogues only (2RXS, 1RXS, XMMSL3, 5XMM, ARTSS1/4) - available for both
               halves, so DE and RU orphan fractions can be compared directly;
  matched_any  the same plus the eRASS1-DE counterpart already carried in the catalogue
               (ERASS1_NAME, brightest eRASS1 source within 4'), which exists only in the DE half.

That asymmetry is the point of the step. In the DE half eRASS1 is complete far below the image
limit, so a DE source that is an orphan under matched_any is almost certainly spurious: the DE
orphan fraction vs SNR *is* the false-positive rate. The RU half has no such reference, so its
spurious fraction cannot be measured directly - but the ext-only orphan fraction can be measured
in both halves, and the RU-minus-DE excess at the same SNR is the extra unmatched population of
the RU half, which is the number this step exists to produce.

Orphans are not automatically spurious. The external all-sky catalogues are shallower than the
image catalogue over most of the sky (2RXS reaches ~1e-13, XMMSL3 ~6e-13), so a faint real source
can legitimately have no counterpart; the flux and SNR distributions separate the two cases, since
a spurious detection population piles up at low SNR while a real one does not.

Excluded: sources inside SNR masks (counted separately - diffuse remnant emission is unmatched by
construction and would dominate) and sources within 0.5 deg of the image joins, where the external
catalogues are cut themselves. Chance matches are measured by repeating the whole search on
positions displaced by SHIFT and used to correct the orphan fractions upward.

The second half of the step turns that into a usable selection. P(spurious) is calibrated on the DE
half, where eRASS1 decides which ext-orphans are real, in SNR bins x morphology class; it is then
transported to the RU half through q(F) = P(no external counterpart | real source of flux F), the
one quantity that is a property of the external catalogues rather than of the image half, and
prior-shifted in odds space to the RU prevalence that q implies. Refitting on half the DE sample
and testing on the other half checks that the resulting probabilities mean what they say. The
product is ru_candidates: every RU source without an X-ray counterpart, with a calibrated P_SPUR,
so a sample of new sources can be cut at a chosen confidence.

Outputs (out/17_orphans/): orphans.parquet (all orphans), ru_candidates.parquet / .csv (the RU
orphans with P_SPUR), orphans.json, orphans_report.md, orphans.pdf, ru_calibration.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

from epngcat.config import OUT, SEED, is_german, load_json, save_json, step_dir
from epngcat.evaluate import shifted
from epngcat.plotting import figure_size, save_figure
from epngcat.s12_artxc import load as load_artxc
from epngcat.s12_rosat import load as load_2rxs
from epngcat.sky import crossmatch, radec_unit

OUTDIR = step_dir("17_orphans")
# Counterpart criterion: offset r < K_MATCH * sqrt(POS_ERR^2 + sigma_ext^2), both per-axis 1-sigma. A fixed
# radius would treat the two halves differently, since RU positions carry an extra error (POS_ERR_ADD)
# and a real RU source is then pushed outside the radius more often and counted as spurious. With the
# errors normalised out, K_MATCH = 3 keeps ~90-97% of true eRASS1 counterparts in every S/N bin and
# within ~1% the same fraction in both halves (checked against 2RXS).
K_MATCH = 3.0
R_MAX = 900.0  # arcsec, search cap (3 x the largest combined error)
N_NEIGHBOURS = 8  # candidates per source tested against the normalised criterion
# sigma_ext is calibrated against point-like eRASS1 sources in the DE half (quoted errors are
# unavailable for 2RXS and ARTSS1 and underestimated by 1.2-3x for the others). Pairs are taken within
# CAL_RADIUS [arcsec] of eRASS1, well above each catalogue's own scatter; 2RXS errors depend strongly
# on the detection likelihood, so it gets one value per likelihood quartile.
CAL_RADIUS = {"2rxs": 60.0, "1rxs": 60.0, "xmmsl3": 45.0, "5xmm": 30.0, "artss1": 120.0, "artss4": 120.0}
RAYLEIGH = np.sqrt(2 * np.log(2))  # median radial offset / per-axis sigma
SHIFT = 30.0  # arcmin, for the chance-match rate
EXI_ALL = 6.5  # 2RXS catalogue limit: a marginal detection still counts as a counterpart
SNR_EDGES = np.array([0, 4, 5, 6, 8, 12, 20, np.inf])
FLUX_EDGES = np.array([0, 3e-13, 6e-13, 1e-12, 3e-12, np.inf])
N_CAL_BINS = 10  # quantile SNR bins of the calibration
N_FLUX_BINS = 8  # quantile flux bins of the DE -> RU prevalence transport
MIN_CELL = 50  # below this a calibration cell falls back to a coarser one
PI_MAX = 0.99  # cap on an inferred RU spurious prevalence, so the odds stay finite
THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05)


def externals():
    """Every external catalogue of the counterpart search: (l, b, grouping), grouping = None or 2RXS ExiML."""
    xmmsl = pd.read_parquet(OUT / "01_catalogs" / "xmmsl3.parquet")
    xmm = pd.read_parquet(OUT / "01_catalogs" / "xmm_dr15.parquet")
    bsc = pd.read_parquet(OUT / "01_catalogs" / "rass_bsc.parquet")
    g = SkyCoord(bsc.RA.values, bsc.DEC.values, unit="deg").galactic
    rx = load_2rxs(EXI_ALL)
    art = {n: load_artxc(n) for n in ("artss1", "artss4")}
    return {
        "2rxs": (rx.l.values, rx.b.values, rx.exi.values),
        "1rxs": (g.l.deg, g.b.deg, None),
        "xmmsl3": (xmmsl.LII.values, xmmsl.BII.values, None),
        "5xmm": (xmm.lii.values, xmm.bii.values, None),
        **{n: (a.l.values, a.b.values, None) for n, a in art.items()},
    }


def external_sigma(ext):
    """Per-axis positional error of each external catalogue [arcsec], from offsets to eRASS1 (DE).

    sigma = sqrt((median offset / RAYLEIGH)^2 - sigma_eRASS1^2), per 2RXS likelihood quartile.
    Returns (calibration record, {name: per-source sigma array}).
    """
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet", columns=["LII", "BII", "POS_ERR", "EXT", "ML_FLUX_1"])
    er = er[(er.EXT == 0) & (er.ML_FLUX_1 > 1e-13)]
    er_sig = er.POS_ERR.values / np.sqrt(2)  # eRASS1 POS_ERR is the radial 1-sigma error
    rec, sig = {}, {}
    for n, (l, b, grp) in ext.items():
        de = is_german(l)
        idx, sep = crossmatch(l[de], b[de], er.LII.values, er.BII.values, CAL_RADIUS[n])
        ok = idx >= 0
        q = np.quantile(grp[de][ok], [0.25, 0.5, 0.75]) if grp is not None else np.array([])
        k_cal = np.digitize(grp[de][ok], q) if grp is not None else np.zeros(ok.sum(), int)
        s = []
        for i in range(len(q) + 1):
            m = k_cal == i
            var = (np.median(sep[ok][m]) / RAYLEIGH) ** 2 - np.median(er_sig[idx[ok]][m]) ** 2
            s.append(float(np.sqrt(max(var, 1.0))))
        sig[n] = np.array(s)[np.digitize(grp, q)] if grp is not None else np.full(len(l), s[0])
        rec[n] = {"n_pairs": int(ok.sum()), "sigma_arcsec": s, **({"exi_quartile_edges": q.tolist()} if len(q) else {})}
    return rec, sig


def match_all(lon, lat, sig_img, ext, sig_ext):
    """Per-catalogue match flags: any source within K_MATCH combined per-axis errors of (lon, lat)."""
    u = radec_unit(lon, lat)
    chord = 2 * np.sin(np.radians(R_MAX / 3600) / 2)
    out = {}
    for n, (el, eb, _) in ext.items():
        d, i = cKDTree(radec_unit(el, eb)).query(u, k=N_NEIGHBOURS, distance_upper_bound=chord)
        ok = np.isfinite(d)
        r = np.where(ok, np.degrees(2 * np.arcsin(np.minimum(np.where(ok, d, 0), 2) / 2)) * 3600, np.inf)
        s = np.hypot(sig_img[:, None], sig_ext[n][np.where(ok, i, 0)])
        out[n] = np.any(r < K_MATCH * s, axis=1)
    return out


def load_catalog():
    """Read the final catalogue and flag SNR-mask membership and the eRASS1 counterpart."""
    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    cat["IN_SNR"] = cat.SNR_NAME.notna() & (cat.SNR_NAME != "")
    cat["matched_erass1"] = cat.ERASS1_NAME.notna() & (cat.ERASS1_NAME != "")
    return cat


def corrected(orphan, chance):
    """Orphan fraction corrected for chance matches: a true orphan is matched by chance with p_c."""
    f, pc = float(orphan.mean()), float(chance.mean())
    return {
        "n": int(len(orphan)),
        "n_orphan": int(orphan.sum()),
        "fraction_raw": f,
        "chance_match_rate": pc,
        "fraction": float(min(f / (1 - pc), 1.0)) if pc < 1 else np.nan,
    }


def by_bin(cat, orphan, chance, col, edges):
    """Orphan fraction of each half in bins of a catalogue column."""
    out = {}
    for half in ("DE", "RU"):
        h = (half == cat.HALF).values
        out[half] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = h & (cat[col].values >= lo) & (cat[col].values < hi)
            v = corrected(orphan[sel], chance[sel]) if sel.sum() else {"n": 0}
            out[half][f"{lo:g}-{hi:g}"] = {**v, "lo": float(lo), "hi": float(hi)}
    return out


def describe(cat, orphan):
    """Compare orphans with matched sources: medians, flag rates and KS tests."""
    out = {}
    for label, sel in (("orphan", orphan), ("matched", ~orphan)):
        s = cat[sel]
        out[label] = {
            "n": int(len(s)),
            "snr_median": float(s.SNR.median()),
            "flux_median": float(s.FLUX_1.median()),
            "abs_b_median": float(s.BII.abs().median()),
            "psf_width_median": float(s.PSF_WIDTH.median()),
            "frac_saturated": float(s.SATURATED.mean()),
            "frac_extended": float(s.FLAG_EXTENDED.mean()),
            "frac_edge": float(s.FLAG_EDGE.mean()),
            "frac_pos_fallback": float(s.FLAG_POS_FALLBACK.mean()),
        }
    for col, key in (("SNR", "snr"), ("FLUX_1", "flux"), ("BII", "abs_b")):
        a = cat[col].values[orphan]
        b = cat[col].values[~orphan]
        if key == "abs_b":
            a, b = np.abs(a), np.abs(b)
        k = ks_2samp(a, b)
        out[f"ks_{key}"] = {"statistic": float(k.statistic), "pvalue": float(k.pvalue)}
    return out


def source_class(cat):
    """Coarse morphology class: a blob (extended or saturated) is far more often spurious."""
    return np.where(cat.FLAG_EXTENDED.values | cat.SATURATED.values, "blob", "point")


def cal_edges(snr, n=N_CAL_BINS):
    """Quantile SNR bin edges, shared by both halves so DE and RU cells are comparable."""
    e = np.unique(np.quantile(snr[np.isfinite(snr)], np.linspace(0, 1, n + 1)))
    e[0], e[-1] = 0.0, np.inf
    return e


def calibrate(cat, train, spurious, chance, edges):
    """P(spurious | SNR bin, class) from the DE training sample, corrected for chance eRASS1 matches.

    A cell with fewer than MIN_CELL sources takes the value of the nearest populated cell of the
    *same class*, not of the merged bin: blobs are spurious several times more often than points at
    the same SNR, so borrowing across classes would badly underestimate a thin blob cell (it did,
    and the held-out check caught it). Only a class with no populated cell at all falls back to the
    sample as a whole.
    """
    cls = source_class(cat)
    k = np.digitize(cat.SNR.values, edges) - 1
    overall = corrected(spurious[train], chance[train])
    out = {"bins": [float(x) for x in edges], "overall": overall, "cells": {}}
    for c in ("point", "blob", "both"):
        out["cells"][c] = []
        for i in range(len(edges) - 1):
            m = train & (k == i) & ((cls == c) if c != "both" else True)
            v = corrected(spurious[m], chance[m]) if m.sum() else {"n": 0}
            out["cells"][c].append({**v, "snr_lo": float(edges[i]), "snr_hi": float(edges[i + 1])})
    for c in ("point", "blob"):
        cells = out["cells"][c]
        good = [i for i, v in enumerate(cells) if v.get("n", 0) >= MIN_CELL and np.isfinite(v.get("fraction", np.nan))]
        for i, v in enumerate(cells):
            if i in good:
                continue
            j = min(good, key=lambda g: abs(g - i)) if good else None
            v["fraction"], v["fallback"] = (cells[j]["fraction"] if j is not None else overall["fraction"]), True
    return out


def predict(cat, cal, sel=None):
    """P(spurious) of every source from a calibration (NaN outside the selection)."""
    cls = source_class(cat)
    k = np.clip(np.digitize(cat.SNR.values, np.array(cal["bins"])) - 1, 0, len(cal["bins"]) - 2)
    p = np.full(len(cat), np.nan)
    for c in ("point", "blob"):
        f = np.array([v["fraction"] for v in cal["cells"][c]])
        m = cls == c
        p[m] = f[k[m]]
    return p if sel is None else np.where(sel, p, np.nan)


def flux_edges(flux, n=N_FLUX_BINS):
    """Quantile flux bin edges, open at both ends."""
    e = np.unique(np.quantile(flux[np.isfinite(flux) & (flux > 0)], np.linspace(0, 1, n + 1)))
    e[0], e[-1] = 0.0, np.inf
    return e


def ru_prevalence(cat, ext_any, ext_chance, erass, spurious, erass_chance, edges):
    """Spurious fraction of the RU orphans per *flux* bin, without RU ground truth.

    The quantity transported from DE to RU is q(F) = P(no external counterpart | real source of
    flux F). That is a property of the external all-sky catalogues and of the sky, not of which
    half of the image the source sits in, and flux is comparable between the halves (the corrected
    RU flux scale agrees with DE's within the step-12 2RXS uncertainty). Binning by SNR instead would not
    do: SNR is a property of the image, and the two halves reach a given SNR at different fluxes.

    With N sources in an RU flux bin, s of them spurious, and O of them ext-orphans,
        O = s (1 - p_chance) + q (N - s)   ->   s = (O - q N) / (1 - p_chance - q),
    where p_chance is the measured rate at which a position matches an external catalogue by
    accident. q is taken from the DE sources that have an eRASS1 counterpart, i.e. the real ones.
    """
    k = np.digitize(cat.FLUX_1.values, edges) - 1
    de, ru = (cat.HALF.values == "DE"), (cat.HALF.values == "RU")
    out = []
    for i in range(len(edges) - 1):
        b = k == i
        de_orph, ru_b = de & b & ~ext_any, ru & b
        real_de = de & b & erass
        v = {
            "flux_lo": float(edges[i]),
            "flux_hi": float(edges[i + 1]),
            "n_de_orphan": int(de_orph.sum()),
            "n_ru": int(ru_b.sum()),
            "n_ru_orphan": int((ru_b & ~ext_any).sum()),
            "n_de_real": int(real_de.sum()),
        }
        pi_de = corrected(spurious[de_orph], erass_chance[de_orph]).get("fraction", np.nan) if de_orph.sum() else np.nan
        v["pi_de"] = float(pi_de) if np.isfinite(pi_de) else np.nan
        v["q"] = float((~ext_any[real_de]).mean()) if real_de.sum() >= MIN_CELL else np.nan
        v["p_chance"] = float(ext_chance[ru_b].mean()) if ru_b.sum() else np.nan
        denom = 1 - v["p_chance"] - v["q"]
        if v["n_ru_orphan"] and np.isfinite(denom) and denom > 0.05:
            s = (v["n_ru_orphan"] - v["q"] * v["n_ru"]) / denom
            v["n_ru_spurious"] = float(max(s, 0.0))
            v["pi_ru"] = float(np.clip(s / v["n_ru_orphan"], 0.0, PI_MAX))
        else:
            v["n_ru_spurious"], v["pi_ru"] = np.nan, np.nan
        out.append(v)
    return out


def _odds(p):
    """Odds of a probability, clipped away from 0 and 1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


def shift_to_ru(p_de, cat, prev, edges):
    """Prior-shift the DE probabilities to the RU spurious prevalence, bin by bin.

    The DE calibration ranks sources within a flux bin; what differs between the halves is the
    prevalence of spurious sources, not the ordering. Rescaling the odds by the ratio of the two
    prevalences keeps the DE ranking and fixes only the level - the standard prior correction.
    """
    k = np.clip(np.digitize(cat.FLUX_1.values, edges) - 1, 0, len(edges) - 2)
    ratio = np.array([_odds(v["pi_ru"]) / _odds(v["pi_de"]) if np.isfinite(v["pi_ru"]) else 1.0 for v in prev])
    o = ratio[k] * _odds(p_de)
    return o / (1 + o)


def validate_calibration(cat, train, spurious, chance, edges, n_bins=5):
    """Fit on half the DE sample, compare predicted and observed spurious rates on the other half."""
    rng = np.random.default_rng(SEED)
    a = train & (rng.random(len(cat)) < 0.5)
    b = train & ~a
    p = predict(cat, calibrate(cat, a, spurious, chance, edges))
    q = np.quantile(p[b], np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    rows = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = b & (p >= lo) & (p < hi)
        if m.sum() < MIN_CELL:
            continue
        obs = corrected(spurious[m], chance[m])
        rows.append(
            {
                "n": int(m.sum()),
                "predicted": float(np.mean(p[m])),
                "observed": obs["fraction"],
                "observed_err": float(
                    np.sqrt(obs["fraction_raw"] * (1 - obs["fraction_raw"]) / m.sum()) / (1 - obs["chance_match_rate"])
                ),
            }
        )
    pred_n = float(np.sum(p[b]))
    obs_n = float(corrected(spurious[b], chance[b])["fraction"] * b.sum())
    return {
        "n_train": int(a.sum()),
        "n_test": int(b.sum()),
        "predicted_spurious": pred_n,
        "observed_spurious": obs_n,
        "ratio": pred_n / obs_n if obs_n else np.nan,
        "bins": rows,
    }


def select(p, thresholds=THRESHOLDS):
    """Count the RU candidates and their expected yield at a set of P(spurious) thresholds."""
    out = {}
    for t in thresholds:
        s = p <= t
        out[f"{t:g}"] = {
            "n": int(s.sum()),
            "expected_real": float((1 - p[s]).sum()),
            "expected_spurious": float(p[s].sum()),
            "purity": float(1 - p[s].mean()) if s.sum() else np.nan,
        }
    return out


def main():
    """Run step 17."""
    cat = load_catalog()
    n_in_snr = int(cat.IN_SNR.sum())
    # The external loaders drop their own sources within 0.5 deg of the image joins (step 12),
    # so image sources there have nothing to match against and would be counted as orphans.
    near_join = (np.abs(((cat.LII + 90) % 180) - 90) <= 0.5).values
    n_join = int((near_join & ~cat.IN_SNR.values).sum())
    cat = cat[~cat.IN_SNR.values & ~near_join].reset_index(drop=True)
    ext = externals()
    print("external catalogues:", {n: len(v[0]) for n, v in ext.items()})
    sig_cal, sig_ext = external_sigma(ext)
    print("external per-axis errors:", {n: np.round(v["sigma_arcsec"], 1).tolist() for n, v in sig_cal.items()})

    sig_img = cat.POS_ERR.values
    flags = match_all(cat.LII.values, cat.BII.values, sig_img, ext, sig_ext)
    ls, bs = shifted(cat.LII.values, cat.BII.values, SHIFT, SEED)
    flags_shift = match_all(ls, bs, sig_img, ext, sig_ext)
    ext_any = np.logical_or.reduce(list(flags.values()))
    ext_any_shift = np.logical_or.reduce(list(flags_shift.values()))
    # The shifted eRASS1 rate is the chance rate of the DE-only reference; outside the DE half it
    # is irrelevant, and matched_any equals matched_ext there by construction.
    erass = cat.matched_erass1.values
    _, sep_shift = crossmatch(ls, bs, *_erass1_positions(), 4 * 60.0)
    erass_shift = np.isfinite(sep_shift) & (cat.HALF.values == "DE")
    cat["MATCHED_EXT"] = ext_any
    cat["MATCHED_ANY"] = ext_any | erass
    for n, f in flags.items():
        cat["MATCH_" + n.upper()] = f
    orphan = ~cat.MATCHED_ANY.values
    orphan_ext = ~ext_any
    chance_any = ext_any_shift | erass_shift
    chance_ext = ext_any_shift

    res = {
        "n_catalog": int(len(cat)),
        "n_in_snr_mask_excluded": n_in_snr,
        "n_near_join_excluded": n_join,
        "match_k_sigma": K_MATCH,
        "external_sigma": sig_cal,
        "n_external": {n: int(len(v[0])) for n, v in ext.items()},
        "matched_per_catalog": {n: int(f.sum()) for n, f in flags.items()},
        "matched_per_catalog_chance": {n: int(f.sum()) for n, f in flags_shift.items()},
        "orphans_any": {h: corrected(orphan[h == cat.HALF], chance_any[h == cat.HALF]) for h in ("DE", "RU")},
        "orphans_ext": {h: corrected(orphan_ext[h == cat.HALF], chance_ext[h == cat.HALF]) for h in ("DE", "RU")},
        "orphans_any_vs_snr": by_bin(cat, orphan, chance_any, "SNR", SNR_EDGES),
        "orphans_ext_vs_snr": by_bin(cat, orphan_ext, chance_ext, "SNR", SNR_EDGES),
        "orphans_any_vs_flux": by_bin(cat, orphan, chance_any, "FLUX_1", FLUX_EDGES),
        "distributions": describe(cat, orphan),
        "distributions_ext": describe(cat, orphan_ext),
    }
    # Cross-check: step 10 measured DE reliability against eRASS1 alone, on the detection list.
    res["de_unmatched_step10"] = 1 - load_json(OUT / "10_validate_de" / "metrics.json")["detection"]["reliability"]
    res["orphans_any"]["all"] = corrected(orphan, chance_any)
    res["orphans_ext"]["all"] = corrected(orphan_ext, chance_ext)
    res["abs_b"] = {
        h: {
            f"{lo:g}-{hi:g}": corrected(orphan[s], chance_any[s])
            for lo, hi in zip([0, 10, 30, 60], [10, 30, 60, 90])
            if (s := ((h == cat.HALF) & (cat.BII.abs() >= lo) & (cat.BII.abs() < hi)).values).sum()
        }
        for h in ("DE", "RU")
    }
    # RU - DE excess at the same SNR, on the ext-only flag (the only one both halves share).
    res["ru_excess_ext"] = {}
    n_excess = 0.0
    for b, de in res["orphans_ext_vs_snr"]["DE"].items():
        ru = res["orphans_ext_vs_snr"]["RU"][b]
        if de.get("n", 0) and ru.get("n", 0):
            d = ru["fraction"] - de["fraction"]
            # Binomial errors on the raw fractions, propagated through the chance correction.
            err = float(
                np.hypot(
                    np.sqrt(ru["fraction_raw"] * (1 - ru["fraction_raw"]) / ru["n"]) / (1 - ru["chance_match_rate"]),
                    np.sqrt(de["fraction_raw"] * (1 - de["fraction_raw"]) / de["n"]) / (1 - de["chance_match_rate"]),
                )
            )
            res["ru_excess_ext"][b] = {"excess": float(d), "err": err, "n_ru": ru["n"], "n_sources": d * ru["n"]}
            n_excess += d * ru["n"]
    res["ru_excess_ext_total_sources"] = float(n_excess)

    keep = [
        "NAME", "RA", "DEC", "LII", "BII", "POS_ERR", "FLUX_1", "FLUX_1_LO", "FLUX_1_HI", "SNR",
        "SATURATED", "PSF_WIDTH", "FLAG_EXTENDED", "FLAG_EDGE", "FLAG_POS_FALLBACK", "HALF", "TILE", "SPLIT",
        "MATCHED_EXT", "MATCHED_ANY",
    ]  # fmt: skip
    orph = cat.loc[orphan, keep].sort_values("FLUX_1", ascending=False).reset_index(drop=True)
    orph.to_parquet(OUTDIR / "orphans.parquet")
    # A genuinely new bright X-ray source would be remarkable; what the bright end of the orphan
    # list actually contains is the question, so quantify its flags instead of just listing it.
    b = orph[orph.FLUX_1 > 1e-11]
    res["bright_orphans"] = {
        "flux_min": 1e-11,
        "n": int(len(b)),
        "frac_extended": float(b.FLAG_EXTENDED.mean()),
        "frac_saturated": float(b.SATURATED.mean()),
        "frac_either": float((b.FLAG_EXTENDED | b.SATURATED).mean()),
    }
    res["brightest"] = [
        {
            "name": r.NAME,
            "l": r.LII,
            "b": r.BII,
            "flux": r.FLUX_1,
            "snr": r.SNR,
            "half": r.HALF,
            "extended": bool(r.FLAG_EXTENDED),
            "saturated": bool(r.SATURATED),
        }
        for r in orph.head(20).itertuples()
    ]
    # Calibration: DE ext-orphans are the population RU orphans belong to, and eRASS1 says which of
    # them are real, so P(spurious) is measured there and transported to the RU half.
    de, ru = (cat.HALF.values == "DE"), (cat.HALF.values == "RU")
    spurious = ~erass  # "no eRASS1 counterpart"; corrected for chance matches inside corrected()
    edges = cal_edges(cat.SNR.values[orphan_ext])
    fedges = flux_edges(cat.FLUX_1.values)
    cal = calibrate(cat, de & orphan_ext, spurious, erass_shift, edges)
    prev = ru_prevalence(cat, ext_any, chance_ext, erass, spurious, erass_shift, fedges)
    p_de = predict(cat, cal)
    p_ru = shift_to_ru(p_de, cat, prev, fedges)
    val = validate_calibration(cat, de & orphan_ext, spurious, erass_shift, edges)

    sel = ru & orphan_ext
    cand = cat.loc[sel, keep].copy()
    cand["CLASS"] = source_class(cat)[sel]
    cand["P_SPUR_DE"] = p_de[sel]  # DE calibration as measured, no prevalence correction
    cand["P_SPUR"] = p_ru[sel]  # the number to use: DE calibration shifted to the RU prevalence
    cand = cand.sort_values(["P_SPUR", "FLUX_1"], ascending=[True, False]).reset_index(drop=True)
    cand.to_parquet(OUTDIR / "ru_candidates.parquet")
    cand.to_csv(OUTDIR / "ru_candidates.csv", index=False)
    res["calibration"] = {
        "snr_bin_edges": cal["bins"],
        "flux_bin_edges": [float(x) for x in fedges],
        "de_overall": cal["overall"],
        "cells": cal["cells"],
        "ru_prevalence": prev,
        "validation": val,
        "n_ru_candidates": int(len(cand)),
        "p_spur_median_de_only": float(np.median(cand.P_SPUR_DE)),
        "p_spur_median": float(np.median(cand.P_SPUR)),
        "expected_real_total": float((1 - cand.P_SPUR).sum()),
        "selection": select(cand.P_SPUR.values),
        "selection_de_only": select(cand.P_SPUR_DE.values),
    }
    save_json(res, OUTDIR / "orphans.json")
    print(
        f"RU candidates: {len(cand)}, expected real {res['calibration']['expected_real_total']:.0f}; "
        f"validation predicted/observed {val['ratio']:.3f}"
    )
    print("  selection:", {k: v["n"] for k, v in res["calibration"]["selection"].items()})
    plot_calibration(cat, cal, prev, val, cand, edges)
    print("orphans (any):", {h: res["orphans_any"][h]["fraction"] for h in ("DE", "RU", "all")})
    print("orphans (ext):", {h: res["orphans_ext"][h]["fraction"] for h in ("DE", "RU", "all")})
    print("RU excess over DE (ext, sources):", res["ru_excess_ext_total_sources"])
    plot(cat, orphan, orphan_ext, res)
    write_report(res)


def _erass1_positions():
    """(l, b) of eRASS1 sources above the counterpart flux floor, for the chance-rate estimate."""
    from epngcat.evaluate import F_FLOOR

    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[er.ML_FLUX_1 >= F_FLOOR]
    return er.LII.values, er.BII.values


def _curve(d):
    """Bin centres and orphan fractions of one by_bin() series."""
    x, y = [], []
    for v in d.values():
        if v.get("n", 0) > 20:
            # Open bins get a centre a factor 3 from their finite edge, so they stay on the plot.
            lo = v["lo"] or v["hi"] / 3
            hi = v["hi"] if np.isfinite(v["hi"]) else v["lo"] * 3
            x.append(np.sqrt(lo * hi))
            y.append(v["fraction"])
    return x, y


def plot(cat, orphan, orphan_ext, res):
    """Orphan fraction vs SNR and flux, the SNR distributions, and the sky distribution."""
    fig = plt.figure(figsize=figure_size(2, 2))
    ax = fig.add_subplot(2, 2, 1)
    for half, c in (("DE", "C0"), ("RU", "C3")):
        ax.plot(
            *_curve(res["orphans_ext_vs_snr"][half]),
            "-",
            drawstyle="steps-mid",
            c=c,
            label=f"{half}, all-sky catalogues only",
        )
    # Only the DE half has eRASS1, so this second curve exists for DE alone.
    ax.plot(
        *_curve(res["orphans_any_vs_snr"]["DE"]), "--", drawstyle="steps-mid", c="C0", alpha=0.6, label="DE, + eRASS1"
    )
    ax.set_xscale("log")
    ax.set_xlabel("image SNR")
    ax.set_ylabel("orphan fraction (chance-corrected)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    for half, c in (("DE", "C0"), ("RU", "C3")):
        x, y = _curve(res["orphans_any_vs_flux"][half])
        ax.plot(x, y, "-", drawstyle="steps-mid", c=c, label=f"{half}, + eRASS1")
    ax.set_xscale("log")
    ax.set_xlabel("image F(0.2-2.3 keV) [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel("orphan fraction (chance-corrected)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 3)
    bins = np.geomspace(3, 100, 30)
    ax.hist(np.clip(cat.SNR[~orphan], 3, 100), bins=bins, histtype="step", density=True, color="k", label="matched")
    ax.hist(
        np.clip(cat.SNR[orphan], 3, 100), bins=bins, histtype="step", density=True, color="C1", label="orphan (any)"
    )
    ax.hist(
        np.clip(cat.SNR[orphan_ext], 3, 100),
        bins=bins,
        histtype="step",
        density=True,
        color="C2",
        ls="--",
        label="orphan (ext only)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("image SNR")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 4, projection="aitoff")
    for half, c in (("DE", "C0"), ("RU", "C3")):
        s = orphan & (cat.HALF.values == half)
        l = np.radians(((cat.LII.values[s] + 180) % 360) - 180)
        ax.scatter(-l, np.radians(cat.BII.values[s]), s=0.6, c=c, alpha=0.35, lw=0, label=f"{half} N={int(s.sum())}")
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])
    ax.legend(fontsize=8, markerscale=8, loc="lower right")
    ax.set_title("orphans on the sky (Galactic, l increasing to the left)", fontsize=9, pad=24)
    fig.suptitle("Step 17: final-catalogue sources without an external X-ray counterpart")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "orphans.pdf")
    plt.close(fig)


def plot_calibration(cat, cal, prev, val, cand, edges):
    """Calibration curve, inferred RU prevalence, the held-out check and the candidate yield."""

    def centres(v, lo, hi):
        """Geometric bin centre, with open bins placed a factor 2 beyond their finite edge."""
        a = v[lo] or v[hi] / 2
        b = v[hi] if np.isfinite(v[hi]) else v[lo] * 2
        return np.sqrt(a * b)

    fig, axs = plt.subplots(2, 2, figsize=figure_size(2, 2))

    ax = axs[0, 0]
    cen = [centres(v, "snr_lo", "snr_hi") for v in cal["cells"]["both"]]
    for c, col in (("point", "C0"), ("blob", "C1"), ("both", "k")):
        y = [v["fraction"] for v in cal["cells"][c]]
        n = sum(v.get("n", 0) for v in cal["cells"][c])
        ax.plot(cen, y, "-" if c != "both" else "--", drawstyle="steps-mid", c=col, label=f"DE {c} (N={n})")
    ax.set_xscale("log")
    ax.set_xlabel("image SNR")
    ax.set_ylabel("P(spurious) among ext-orphans")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("calibration curve on eRASS1-DE truth", fontsize=10)

    ax = axs[1, 0]
    fc = [centres(v, "flux_lo", "flux_hi") for v in prev]
    ax.plot(fc, [v["pi_de"] for v in prev], "-", drawstyle="steps-mid", c="C0", label="DE, measured against eRASS1")
    ax.plot(fc, [v["pi_ru"] for v in prev], "-", drawstyle="steps-mid", c="C3", label="RU, inferred from q(F)")
    ax.plot(
        fc, [v["q"] for v in prev], "--", drawstyle="steps-mid", c="C2", label="q(F) = P(no ext. counterpart | real)"
    )
    ax.set_xscale("log")
    ax.set_xlabel("image F(0.2-2.3 keV) [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel("spurious fraction among ext-orphans")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("prevalence transported per flux bin", fontsize=10)

    ax = axs[0, 1]
    ax.plot([0, 1], [0, 1], "k-", lw=1)
    ax.errorbar(
        [r["predicted"] for r in val["bins"]],
        [r["observed"] for r in val["bins"]],
        yerr=[r["observed_err"] for r in val["bins"]],
        fmt=".",
        c="C2",
    )
    ax.set_xlabel("predicted P(spurious), fitted on half the DE sample")
    ax.set_ylabel("observed spurious fraction, other half")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"held-out check: predicted/observed = {val['ratio']:.3f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axs[1, 1]
    p = np.sort(cand.P_SPUR.values)
    ax.plot(p, np.arange(1, len(p) + 1), c="C3", label="candidates selected")
    ax.plot(p, np.cumsum(1 - p), c="C2", label="expected real")
    ax.plot(p, np.cumsum(p), c="k", ls="--", label="expected spurious")
    ax.plot(
        np.sort(cand.P_SPUR_DE.values), np.arange(1, len(p) + 1), c="C0", alpha=0.5, label="selected, DE calib only"
    )
    ax.set_xlabel("P(spurious) threshold")
    ax.set_ylabel("cumulative number")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("RU candidate yield", fontsize=10)

    fig.suptitle("Step 17: spuriousness calibrated on the DE half and transported to the RU half")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "ru_calibration.pdf")
    plt.close(fig)


def write_report(res):
    """Markdown summary."""
    o, e = res["orphans_any"], res["orphans_ext"]
    L = [
        "# Sources without an external X-ray counterpart",
        "",
        f"{res['n_catalog']} final-catalogue sources. Excluded: {res['n_in_snr_mask_excluded']} inside SNR masks",
        "(remnant emission has no point-source counterpart by construction) and "
        f"{res['n_near_join_excluded']} within 0.5 deg of the image joins, where the external catalogues are",
        "themselves cut and every source would look like an orphan.",
        f"Counterpart criterion: offset < {res['match_k_sigma']:g} x sqrt(POS_ERR^2 + sigma_ext^2), per-axis errors;"
        " sigma_ext calibrated against eRASS1 in the DE half: "
        + ", ".join(f'{k} {"/".join(f"{x:.0f}" for x in v["sigma_arcsec"])}"' for k, v in res["external_sigma"].items())
        + " (2RXS per ExiML quartile). eRASS1: 4' (DE only, the catalogue's own ERASS1_NAME column).",
        f"Fractions are corrected for chance matches, measured by repeating the search {SHIFT:.0f}' away.",
        "",
        "## Counterparts per catalogue",
        "",
        "| catalogue | N in catalogue | image sources matched | of which chance |",
        "|---|---|---|---|",
    ]
    for n, v in res["matched_per_catalog"].items():
        L.append(f"| {n} | {res['n_external'][n]} | {v} | {res['matched_per_catalog_chance'][n]} |")
    L += [
        "",
        "## Orphan fraction",
        "",
        "| sample | half | N | orphans | raw | chance-corrected |",
        "|---|---|---|---|---|---|",
    ]
    for label, d in (("all-sky catalogues only", e), ("+ eRASS1 (DE)", o)):
        for half in ("DE", "RU", "all"):
            v = d[half]
            L.append(
                f"| {label} | {half} | {v['n']} | {v['n_orphan']} | {v['fraction_raw']:.1%} | {v['fraction']:.1%} |"
            )
    L += [
        "",
        "In the DE half eRASS1 reaches far below the image limit, so the `+ eRASS1` orphan fraction there is",
        "essentially the false-positive rate of the image catalogue. The all-sky-only rows are the ones that",
        "can be compared between the halves, since both are measured against the same catalogues.",
        "",
        f"Consistency: step 10 measured {res['de_unmatched_step10']:.1%} of DE detections unmatched in eRASS1 alone,",
        f"against {o['DE']['fraction']:.1%} here with the external catalogues added - the same number, from a",
        "different stage of the pipeline.",
        "",
        "## Orphan fraction vs SNR (all-sky catalogues only / + eRASS1)",
        "",
        "| SNR | DE ext | DE + eRASS1 | RU ext | RU - DE excess |",
        "|---|---|---|---|---|",
    ]
    for b in res["orphans_ext_vs_snr"]["DE"]:
        de_e, de_a = res["orphans_ext_vs_snr"]["DE"][b], res["orphans_any_vs_snr"]["DE"][b]
        ru_e = res["orphans_ext_vs_snr"]["RU"][b]
        x = res["ru_excess_ext"].get(b)
        L.append(
            f"| {b} | {de_e.get('fraction', np.nan):.1%} ({de_e.get('n', 0)}) | {de_a.get('fraction', np.nan):.1%} | "
            f"{ru_e.get('fraction', np.nan):.1%} ({ru_e.get('n', 0)}) | "
            + (f"{x['excess']:+.1%} +- {x['err']:.1%} ({x['n_sources']:+.0f} sources) |" if x else "- |")
        )
    L += [
        "",
        f"Summed over the SNR bins, the RU half has **{res['ru_excess_ext_total_sources']:+.0f}** more unmatched",
        f"sources than the DE half would have at the same SNR "
        f"({res['ru_excess_ext_total_sources'] / max(res['orphans_ext']['RU']['n'], 1):+.1%} of the RU catalogue).",
        "",
        "## Orphans vs matched sources",
        "",
        "| property | orphan (any) | matched | orphan (ext only) |",
        "|---|---|---|---|",
    ]
    d, dx = res["distributions"], res["distributions_ext"]
    rows = [
        ("N", "n", "{:d}"),
        ("median SNR", "snr_median", "{:.1f}"),
        ("median F(0.2-2.3)", "flux_median", "{:.2e}"),
        ("median abs(b)", "abs_b_median", "{:.1f}"),
        ("median PSF width", "psf_width_median", "{:.2f}"),
        ("saturated", "frac_saturated", "{:.1%}"),
        ("extended", "frac_extended", "{:.1%}"),
        ("near edge", "frac_edge", "{:.1%}"),
        ("position fallback", "frac_pos_fallback", "{:.1%}"),
    ]
    for label, key, f in rows:
        L.append(
            f"| {label} | {f.format(d['orphan'][key])} | {f.format(d['matched'][key])} | "
            f"{f.format(dx['orphan'][key])} |"
        )
    L += [
        "",
        "Two-sample KS tests, orphan (any) vs matched: "
        + ", ".join(
            f"{k.split('_', 1)[1]} D={v['statistic']:.3f} (p={v['pvalue']:.1e})"
            for k, v in d.items()
            if k.startswith("ks_")
        ),
        "",
        "## Orphan fraction vs Galactic latitude (+ eRASS1)",
        "",
        "| abs(b) | DE | RU |",
        "|---|---|---|",
    ]
    for b in res["abs_b"]["DE"]:
        de, ru = res["abs_b"]["DE"][b], res["abs_b"]["RU"].get(b, {})
        L.append(f"| {b} | {de['fraction']:.1%} ({de['n']}) | {ru.get('fraction', np.nan):.1%} ({ru.get('n', 0)}) |")
    L += [
        "",
        "## Brightest orphans",
        "",
        "| name | l | b | F(0.2-2.3) | SNR | half | extended | saturated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in res["brightest"]:
        L.append(
            f"| {s['name']} | {s['l']:.3f} | {s['b']:+.3f} | {s['flux']:.2e} | {s['snr']:.1f} | {s['half']} | "
            f"{'yes' if s['extended'] else ''} | {'yes' if s['saturated'] else ''} |"
        )
    c = res["calibration"]
    L += [
        "",
        "## Calibrated RU candidates",
        "",
        "`ru_candidates.parquet` / `.csv`: every RU source with no counterpart in the all-sky catalogues,",
        f"with `P_SPUR`, an approximate spurious probability. {c['n_ru_candidates']} sources, of which",
        f"the model estimates **{c['expected_real_total']:.0f} real sources**. Select candidates by cutting on",
        "`P_SPUR`; the columns `P_SPUR_DE` (DE calibration before the prevalence correction) and `CLASS` are",
        "kept for reference.",
        "Expected yields and purities are model estimates, not independently measured catalogue purity."
        " They assume comparable external-counterpart selection at fixed flux in both footprints.",
        "",
        "How it is derived, in three steps:",
        "",
        "1. **Calibration curve.** In the DE half eRASS1 is the reference: among DE sources that are",
        "   themselves ext-orphans the chance-corrected unmatched fraction is used to approximate P(spurious).",
        f"   It is measured in {len(c['snr_bin_edges']) - 1} quantile SNR bins x 2 morphology classes (point,",
        "   blob = extended or saturated); a thin cell is carried from the nearest populated cell of the same",
        "   class, never from the other class, which is several times cleaner at the same SNR.",
        "2. **Prevalence transport**, in flux bins rather than SNR bins: SNR is a property of the image and",
        "   the halves reach a given SNR at different fluxes. Step-12 comparisons check the flux scales",
        "   (see rosat_2rxs.json), but do not establish identical counterpart selection. What transports is",
        "   q(F) = P(no external counterpart | *real* source of flux F), a property of the external catalogues",
        "   and of the sky, measured on the DE sources that do have an eRASS1 counterpart. With N RU sources",
        "   in a flux bin, O of them ext-orphans and a chance-match rate p_c, the spurious count s follows",
        "   from O = s(1 - p_c) + q(N - s). The DE probabilities are then prior-shifted to that prevalence in",
        "   odds space, which keeps the DE ranking and corrects only the level.",
        "3. **Validation.** The curve is refitted on a random half of the DE sample and tested on the other.",
        "",
        "### Calibration curve (DE, per SNR bin)",
        "",
        "| SNR bin | N | P(spur) point | P(spur) blob | merged |",
        "|---|---|---|---|---|",
    ]
    for i, m in enumerate(c["cells"]["both"]):
        pt, bl = c["cells"]["point"][i], c["cells"]["blob"][i]
        hi = "inf" if not np.isfinite(m["snr_hi"]) else f"{m['snr_hi']:.1f}"
        L.append(
            f"| {m['snr_lo']:.1f}-{hi} | {m.get('n', 0)} | "
            f"{pt['fraction']:.3f}{'*' if pt.get('fallback') else ''} | "
            f"{bl['fraction']:.3f}{'*' if bl.get('fallback') else ''} | {m.get('fraction', np.nan):.3f} |"
        )
    L += [
        "",
        f"`*` = cell below {MIN_CELL} sources, carried from the nearest populated cell of the same class.",
        "",
        "### Prevalence transport (per flux bin)",
        "",
        "| F(0.2-2.3) | q(F) | N RU | RU orphans | chance | DE prevalence | RU prevalence | RU spurious |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in c["ru_prevalence"]:
        hi = "inf" if not np.isfinite(v["flux_hi"]) else f"{v['flux_hi']:.1e}"
        L.append(
            f"| {v['flux_lo']:.1e}-{hi} | {v['q']:.3f} | {v['n_ru']} | {v['n_ru_orphan']} | "
            f"{v['p_chance']:.1%} | {v['pi_de']:.3f} | {v['pi_ru']:.3f} | {v['n_ru_spurious']:.0f} |"
        )
    val = c["validation"]
    L += [
        "",
        f"Held-out check ({val['n_train']} train / {val['n_test']} test DE sources): predicted",
        f"{val['predicted_spurious']:.0f} spurious, observed {val['observed_spurious']:.0f}, ratio "
        f"**{val['ratio']:.3f}**.",
        "",
        "| predicted P(spur) | N | observed (chance-corrected) |",
        "|---|---|---|",
    ]
    for r in val["bins"]:
        L.append(f"| {r['predicted']:.3f} | {r['n']} | {r['observed']:.3f} +- {r['observed_err']:.3f} |")
    L += [
        "",
        "### Yield of the RU candidate list",
        "",
        "| P_SPUR <= | N selected | expected real | expected spurious | purity |",
        "|---|---|---|---|---|",
    ]
    for t, v in c["selection"].items():
        L.append(f"| {t} | {v['n']} | {v['expected_real']:.0f} | {v['expected_spurious']:.0f} | {v['purity']:.1%} |")
    L += [
        "",
        "Caveats: the probabilities are calibrated against *eRASS1*, so 'real' means 'a real X-ray source',",
        "not 'not in any catalogue at any wavelength' - no optical or SIMBAD search is done here. The",
        "transport assumes the two halves see the same sky population at a given SNR, which is what the",
        "step-12 external comparisons support but do not prove; and it assumes the RU image only adds spurious",
        "sources rather than changing which real ones it finds. Figure: `ru_calibration.pdf`.",
    ]

    b = res["bright_orphans"]
    L += [
        "",
        f"Of the {b['n']} orphans above {b['flux_min']:.0e} erg/s/cm2, {b['frac_extended']:.0%} are flagged extended",
        f"and {b['frac_saturated']:.0%} saturated ({b['frac_either']:.0%} either): the bright end of the orphan",
        "list is diffuse structure and saturated blobs measured as single sources, not undiscovered point sources.",
        "",
        "Full list: `orphans.parquet`. Figure: `orphans.pdf`.",
        "",
    ]
    (OUTDIR / "orphans_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
