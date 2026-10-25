"""Step 11: image-based catalogue of the RU half and DE/RU consistency checks.

The halves were rendered separately (different projection scale, different channel
registration), so the DE calibration is checked on image-only statistics that should agree
between statistically equivalent skies:
  * source surface density vs |b|
  * cumulative number counts N(>S) of image fluxes at |b| > 20 deg
  * raw background level and noise per channel at |b| > 20 deg
  * PSF width (free-width native fit) of unsaturated sources

Outputs (out/11_catalog_ru/): ru_pipeline.parquet, ru_catalog.parquet, consistency.json, consistency.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epngcat.apply_models import apply
from epngcat.config import OUT, save_json, step_dir
from epngcat.evaluate import in_snr
from epngcat.pipeline import process_tiles
from epngcat.plotting import figure_size, save_figure

OUTDIR = step_dir("11_catalog_ru")
B_HIGH = 20.0
S_GRID = np.geomspace(3e-13, 3e-11, 11)


def tile_area(tiles):
    """Area [deg2] of each tile times its image coverage."""
    dl = np.radians(tiles.l_width)
    area = np.degrees(1) ** 2 * dl * (np.sin(np.radians(tiles.b_hi)) - np.sin(np.radians(tiles.b_lo)))
    return area * tiles.coverage


def half_stats(cat, det, tiles):
    """Image-only statistics of one half."""
    area_tile = tile_area(tiles)
    clean = ~cat.IN_SNR_MASK.values
    out = {"n": int(clean.sum())}
    dens = {}
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 90)]:
        t = tiles[(tiles.b_c.abs() >= lo) & (tiles.b_c.abs() < hi)]
        sel = clean & cat.TILE.isin(t.tile).values
        dens[f"{lo}-{hi}"] = float(sel.sum() / area_tile[t.index].sum())
    out["density_per_deg2"] = dens
    hi_t = tiles[tiles.b_c.abs() >= B_HIGH]
    sel = clean & cat.TILE.isin(hi_t.tile).values
    area = float(area_tile[hi_t.index].sum())
    out["logN_logS_hiB"] = {f"{s:.2e}": float((cat.FLUX_1.values[sel] > s).sum() / area) for s in S_GRID}
    hb = sel & (np.abs(cat.BII.values) >= B_HIGH)
    for c in "RGB":
        out[f"rawbkg_{c}_median"] = float(np.median(det[f"rawbkg_{c}"].values[hb]))
        out[f"noise_{c}_median"] = float(np.median(det[f"noise_{c}"].values[hb]))
    uns = hb & ~cat.SATURATED.values
    out["psf_width_median"] = float(np.nanmedian(cat.PSF_WIDTH.values[uns]))
    return out


def main():
    """Run step 11."""
    tiles = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    ru_tiles = tiles[~tiles.de]
    det = process_tiles(ru_tiles)
    det["split"] = "ru"
    det["in_snr"] = in_snr(det.l.values, det.b.values)
    det.to_parquet(OUTDIR / "ru_pipeline.parquet")
    cat = apply(det)
    cat["SPLIT"] = "ru"
    cat["IN_SNR_MASK"] = det.in_snr.values
    cat.to_parquet(OUTDIR / "ru_catalog.parquet")

    de_det = pd.read_parquet(OUT / "08_de_pipeline" / "de_pipeline.parquet")
    de_cat = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    stats = {"de": half_stats(de_cat, de_det, tiles[tiles.de]), "ru": half_stats(cat, det, ru_tiles)}
    ratio = {
        s: stats["ru"]["logN_logS_hiB"][s] / max(stats["de"]["logN_logS_hiB"][s], 1e-9)
        for s in stats["de"]["logN_logS_hiB"]
    }
    # Flux offset that would align RU counts to DE counts (median over the flux grid; slope ~ -1.3).
    s_de = np.array(list(stats["de"]["logN_logS_hiB"].values()))
    s_ru = np.array(list(stats["ru"]["logN_logS_hiB"].values()))
    ok = (s_de > 0) & (s_ru > 0)
    logs = np.log10(S_GRID[ok])
    shift = np.interp(np.log10(s_ru[ok]), np.log10(s_de[ok])[::-1], logs[::-1]) - logs
    stats["ru_de_count_ratio"] = ratio
    stats["ru_flux_offset_from_counts_dex"] = float(np.median(shift))
    save_json(stats, OUTDIR / "consistency.json")
    print(stats)

    plot_results(stats)


def plot_results(stats):
    """Render diagnostics separately from the expensive analysis."""
    fig, axs = plt.subplots(1, 3, figsize=figure_size(1, 3))
    for h in ("de", "ru"):
        axs[0].stairs(list(stats[h]["density_per_deg2"].values()), [0, 10, 20, 40, 60, 90], baseline=None, label=h)
        axs[1].loglog(S_GRID, [stats[h]["logN_logS_hiB"][f"{s:.2e}"] for s in S_GRID], "-", label=h)
    axs[0].set_xlabel(r"$|b|$ range [deg]")
    axs[0].set_ylabel("sources / deg$^2$")
    axs[1].set_xlabel("image F(0.2-2.3 keV)")
    axs[1].set_ylabel(rf"$N(>S)$ [deg$^{{-2}}$], $|b| > {B_HIGH:.0f}^\circ$")
    for ax in axs[:2]:
        ax.legend()
    for k, c in enumerate("RGB"):
        for j, h in enumerate(("de", "ru")):
            axs[2].bar(k * 3 + j, stats[h][f"rawbkg_{c}_median"], color=c.lower(), alpha=0.5 + 0.4 * j)
    axs[2].set_xticks([0.5, 3.5, 6.5], ["R (DE, RU)", "G", "B"])
    axs[2].set_ylabel("median raw background")
    fig.tight_layout()
    save_figure(fig, OUTDIR / "consistency.pdf")


if __name__ == "__main__":
    main()
