"""Step 04: sky tiles, image coverage and hold-out split.

* ~2x2 deg tiles covering the sky (see epngcat.patches)
* per-tile coverage: fraction of the tile core that falls inside the drawn sky ellipse (the image
  is cropped by ~2-3 px along its outline, which removes up to ~1 deg around l = 180 and the
  poles)
* DE tiles are grouped in ~10x10 deg blocks and blocks are randomly assigned to
  train (60%) / val (20%) / test (20%), so neighbouring tiles never straddle two splits;
  RU tiles get split = "ru"
* eRASS1 sources get their tile id and split

Outputs (out/04_tiles/): tiles.parquet, erass1_tiles.parquet, coverage.pdf, split.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from epngcat.config import L_SPLIT, OUT, SEED, is_german, save_json, step_dir
from epngcat.image import sky_mask
from epngcat.patches import tile_of, tile_table
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import load_alignment

OUTDIR = step_dir("04_tiles")
BLOCK = 10.0  # deg
SPLIT_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}
N_COVERAGE = 4_000_000  # random sky points for coverage estimation


def coverage(tiles):
    """Fraction of each tile's area that is inside the drawn sky ellipse."""
    rng = np.random.default_rng(SEED)
    l = rng.uniform(0, 360, N_COVERAGE)
    b = np.degrees(np.arcsin(rng.uniform(-1, 1, N_COVERAGE)))
    px, py = load_alignment().sky2pix(l, b)
    mask = ndimage.binary_erosion(sky_mask())  # a full interpolation footprint on the sky
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
    ok = np.zeros(N_COVERAGE, bool)
    ok[inside] = mask[iy[inside], ix[inside]]
    ids = tile_of(l, b)
    n = np.bincount(ids, minlength=len(tiles))
    return np.bincount(ids, weights=ok, minlength=len(tiles)) / np.maximum(n, 1), (l, b, ok)


def assign_split(tiles):
    """Random block-wise train/val/test split of DE tiles."""
    brow = ((tiles.b_c + 90) // BLOCK).astype(int)
    bcol = (((tiles.l_c - L_SPLIT) % 360) * np.cos(np.radians(tiles.b_c)) // BLOCK).astype(int)
    block = brow * 1000 + bcol
    ublocks = np.unique(block[tiles.de])
    r = np.random.default_rng(SEED).random(len(ublocks))
    edges = np.cumsum(list(SPLIT_FRAC.values()))
    names = np.array(list(SPLIT_FRAC))[np.searchsorted(edges, r)]
    split = pd.Series(block).map(dict(zip(ublocks, names))).where(tiles.de, "ru")
    return block, split.values


def main():
    """Run step 04."""
    tiles = pd.DataFrame(tile_table())
    tiles["de"] = is_german(tiles.l_c)
    tiles["coverage"], (sl, sb, sok) = coverage(tiles)
    tiles["block"], tiles["split"] = assign_split(tiles)
    tiles.to_parquet(OUTDIR / "tiles.parquet")

    cat = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet", columns=["IAUNAME", "LII", "BII"])
    cat["tile"] = tile_of(cat.LII.values, cat.BII.values)
    cat["split"] = tiles.split.values[cat.tile]
    cat[["IAUNAME", "tile", "split"]].to_parquet(OUTDIR / "erass1_tiles.parquet")

    area = 4 * np.pi * (180 / np.pi) ** 2
    lost = 1 - sok.mean()
    summary = {
        "n_tiles": len(tiles),
        "n_tiles_by_split": tiles.split.value_counts().to_dict(),
        "erass1_sources_by_split": cat.split.value_counts().to_dict(),
        "sky_area_not_in_image_deg2": lost * area,
        "sky_area_not_in_image_frac": lost,
        "tiles_coverage_lt_50pct": int((tiles.coverage < 0.5).sum()),
    }
    save_json(summary, OUTDIR / "summary.json")
    print(summary)

    fig, ax = plt.subplots(figsize=figure_size(wide=True))
    miss = ~sok
    ax.scatter(((sl[miss] + 180) % 360) - 180, sb[miss], s=0.2, c="k")
    ax.set_xlim(180, -180)
    ax.set_xlabel(r"$l$ [deg] (wrapped)")
    ax.set_ylabel(r"$b$ [deg]")
    ax.set_title(f"sky not contained in the image: {lost * area:.0f} deg$^2$ ({100 * lost:.2f}%)")
    save_figure(fig, OUTDIR / "coverage.pdf")

    fig, ax = plt.subplots(figsize=figure_size(wide=True))
    colors = {"train": "C0", "val": "C1", "test": "C2", "ru": "0.8"}
    for name, c in colors.items():
        t = tiles[tiles.split == name]
        ax.scatter(((t.l_c + 180) % 360) - 180, t.b_c, s=2, c=c, label=f"{name} ({len(t)})")
    ax.set_xlim(180, -180)
    ax.legend(markerscale=5)
    ax.set_title("tile split")
    save_figure(fig, OUTDIR / "split.pdf")


if __name__ == "__main__":
    main()
