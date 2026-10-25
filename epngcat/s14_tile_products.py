"""Step 14: per-tile products for every tile of the sky.

For each ~2x2 deg tile (local TAN patch, 2'/px, core + 0.6 deg padding), the tile is reprocessed
exactly as in the pipeline (deterministic), and written as

  fits/tile_NNNNN.fits  compressed images in display units (0-255): DATA (R,G,B), MODEL
                        (background + sources), BACKGROUND (B-spline), SUB (DATA with fitted point
                        sources removed, background/noise left as-is), RESIDUAL (model - data),
                        VALID mask; a GLON-TAN/GLAT-TAN WCS; SOURCES table (catalogue sources whose
                        position lies in the tile core, with patch pixel coordinates)
  pdf/tile_NNNNN.pdf    data with catalogue detections and reference sources (eRASS1 for the DE
                        half, XMMSL3, ROSAT BSC, SNRcat outlines), model, background, cleaned
                        (point sources subtracted), residual; DE tiles also show the eRASS1-DE
                        DR1 HiPS rate maps (0.2-0.5 / 0.5-1 / 1-2 keV as RGB) at 30" and smoothed
                        to the press-image PSF
  hires/tile_NNNNN_hires.fits  (DE only) training pair for image-to-image models: HIRES
                        (3, 4n-3, 4n-3) HiPS rate cube [counts/s/arcmin2] at 30"/px (lossless) and
                        LOWRES (3, n, n) press-image cutout at 2'/px, both on the same TAN projection
                        (pixel centres of LOWRES coincide with every 4th HIRES pixel); see epngcat.hires

Model images are converted from linear units back to display units with the calibrated stretch
(v = A asinh(I) + v0), so the residual is directly comparable to the PDF pixel values.

`make` passes --hires, so a clean run fetches the HiPS maps: ~25k order-6 tiles per band from the
MPE server, for the 5284 DE tiles. Existing hires files are always reused (no download), so a
re-render costs nothing. Without --hires, DE tiles that have no hires file get the 2x2 layout only.

Outputs (out/14_tiles/): fits/, pdf/, hires/, index.csv
"""

from concurrent.futures import ProcessPoolExecutor
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse, Polygon

from epngcat.config import OUT, WORKERS, step_dir
from epngcat.hires import BAND_LABEL, BANDS, HIRES_FACTOR, hires_cube, hires_grid
from epngcat.patches import ROW_H
from epngcat.pipeline import load_params, tile_state
from epngcat.plotting import figure_size, save_figure

OUTDIR = step_dir("14_tiles")
RESID_RANGE = 40.0  # display units, colour range of the residual panel
QUANTIZE = 16  # FITS lossy compression level (noise / quantisation step)
# Fixed asinh display scales of the HiPS panels (per band, counts/s/arcmin2), same for all tiles.
HIRES_SMOOTH = 2.0  # hires px (1')
HIRES_SOFT = np.array([1.5e-4, 2e-4, 1.8e-4])
HIRES_TOP = 8.0
PSF_SOFT = np.array([4e-5, 5.5e-5, 5e-5])
PSF_TOP = 10.0


def tile_wcs(patch, factor=1):
    """Galactic TAN WCS of a patch (x axis along increasing l), optionally `factor` times finer."""
    c = patch.c * factor
    w = WCS(naxis=2)
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    w.wcs.crval = [patch.lc, patch.bc]
    w.wcs.crpix = [c + 1, c + 1]
    w.wcs.cdelt = [patch.scale / factor, patch.scale / factor]
    w.wcs.cunit = ["deg", "deg"]
    return w


def to_display(lin, stretch):
    """Linear units -> display units (inverse of detect.linearize)."""
    return np.clip(np.arcsinh(lin) * np.asarray(stretch["A"]) + stretch["v0"], 0, 255)


@cache
def references():
    """Load reference catalogues (Galactic positions, fluxes) and SNRs."""
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet", columns=["LII", "BII", "ML_FLUX_1", "EXT"])
    er = er[er.ML_FLUX_1 > 1e-13]
    xm = pd.read_parquet(OUT / "01_catalogs" / "xmmsl3.parquet", columns=["LII", "BII", "FLUX_B6"])
    bsc = pd.read_parquet(OUT / "01_catalogs" / "rass_bsc.parquet", columns=["RA", "DEC", "FLUX"])
    g = SkyCoord(bsc.RA.values, bsc.DEC.values, unit="deg").galactic
    bsc = pd.DataFrame({"LII": g.l.deg, "BII": g.b.deg, "FLUX": bsc.FLUX.values})
    snr = pd.read_parquet(OUT / "01_catalogs" / "snrcat.parquet")
    cat = pd.read_csv(
        OUT / "13_final" / "erass1_image_catalog.csv",
        keep_default_na=False,
        na_values=[""],
        dtype={"ERASS1_NAME": str},
    )
    return {"erass1": er, "xmmsl3": xm, "bsc": bsc, "snr": snr, "cat": cat}


def near(df, patch, lcol="LII", bcol="BII"):
    """Rows of df inside the patch, with patch pixel coordinates."""
    d = df[np.abs(df[bcol].values - patch.bc) < patch.npix * patch.scale]
    if len(d) == 0:
        return d.assign(x=[], y=[])
    x, y = patch.sky2pix(d[lcol].values, d[bcol].values)
    # Points on the far hemisphere of the tangent plane are not in the patch.
    lr, br, lc, bc = map(np.radians, (d[lcol].values, d[bcol].values, patch.lc, patch.bc))
    cosc = np.sin(bc) * np.sin(br) + np.cos(bc) * np.cos(br) * np.cos(lr - lc)
    inside = (cosc > 0) & (x > -0.5) & (y > -0.5) & (x < patch.npix - 0.5) & (y < patch.npix - 0.5)
    return d[inside].assign(x=x[inside], y=y[inside])


def asinh_rgb(cube, soft, top, sigma):
    """Display RGB (n, n, 3) of a rate cube with fixed per-band asinh scaling."""
    from scipy.ndimage import gaussian_filter

    sm = np.stack([gaussian_filter(np.nan_to_num(c), sigma) for c in cube], -1)
    return np.clip(np.arcsinh(sm / soft) / np.arcsinh(top), 0, 1)


def read_hires(path):
    """Cube of an existing hires file, or None if it is missing or was left truncated."""
    if not path.exists():
        return None
    try:
        return fits.getdata(path, "HIRES").astype(np.float32)
    except (KeyError, OSError, ValueError):
        path.unlink()
        return None


def write_hires(tid, patch, cube, data, tile):
    """Training pair file: HiPS rate cube (lossless) + press-image cutout on the same TAN grid."""
    n, _, scale = hires_grid(patch)
    hdr = fits.Header()
    hdr["TILE"] = tid
    hdr["L_CEN"], hdr["B_CEN"] = patch.lc, patch.bc
    hdr["SPLIT"] = tile["split"]
    hdr["FACTOR"] = (HIRES_FACTOR, "LOWRES pixel = FACTOR x HIRES pixel")
    hdr["SOURCE"] = "eRASS1-DE DR1 HiPS eRASS1_0NN_Rate_c010 (Merloni+ 2024), order 6"
    for k, band in enumerate(BANDS):
        hdr[f"BAND{k + 1}"] = (band, BAND_LABEL[band])
    hi = tile_wcs(patch, HIRES_FACTOR).to_header()
    hi["BUNIT"] = "ct/s/arcmin2"
    lo = tile_wcs(patch).to_header()
    lo["BUNIT"] = "display"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.CompImageHDU(cube, header=hi, name="HIRES", compression_type="GZIP_2", quantize_level=0.0),
            fits.CompImageHDU(np.moveaxis(data, -1, 0).astype(np.uint8), header=lo, name="LOWRES"),
            fits.CompImageHDU(patch.valid.astype(np.uint8), header=tile_wcs(patch).to_header(), name="VALID"),
        ]
    ).writeto(OUTDIR / "hires" / f"tile_{tid:05d}_hires.fits", overwrite=True)


def core_outline(tile, patch, n=40):
    """Tile core boundary in patch pixels."""
    s = np.linspace(0, 1, n)
    l = np.concatenate(
        [
            tile["l_lo"] + s * tile["l_width"],
            np.full(n, tile["l_lo"] + tile["l_width"]),
            tile["l_lo"] + s[::-1] * tile["l_width"],
            np.full(n, tile["l_lo"]),
        ]
    )
    b = np.concatenate(
        [np.full(n, tile["b_lo"]), tile["b_lo"] + s * ROW_H, np.full(n, tile["b_hi"]), tile["b_hi"] - s * ROW_H]
    )
    return np.stack(patch.sky2pix(l % 360, np.clip(b, -90, 90)), -1)


def product(args):
    """Make the FITS + PDF products of one tile."""
    tile, det_params, calib, with_hires = args
    tid = int(tile["tile"])
    fpath, ppath = OUTDIR / "fits" / f"tile_{tid:05d}.fits", OUTDIR / "pdf" / f"tile_{tid:05d}.pdf"
    st = tile_state(tile, det_params, calib)
    patch = st["patch"]
    stretch = det_params["stretch"]
    data = patch.rgb.astype(np.float32)
    bkg_lin, src_lin, clean_lin = st["background"], st["source"], st["clean"]
    model = to_display(bkg_lin + src_lin, stretch)
    bkg = to_display(bkg_lin, stretch)
    sub = to_display(clean_lin, stretch)
    # Clipped values do not enter the residual, but they remain valid in SUB: tile_state supplies
    # the counterfactual fitted background there and MODELLED records that provenance per channel.
    resid = np.where(patch.valid[..., None] & ~st["modelled"], model - data, np.nan)
    ref = references()
    cat = near(ref["cat"][ref["cat"].TILE.values == tid], patch)
    hpath = OUTDIR / "hires" / f"tile_{tid:05d}_hires.fits"
    cube = read_hires(hpath)  # re-rendering: reuse the downloaded cube if the file is intact
    de = bool(tile["de"]) and (with_hires or cube is not None)
    if de and cube is None:
        cube = hires_cube(patch)
        write_hires(tid, patch, cube, data, tile)

    # FITS
    w = tile_wcs(patch).to_header()
    hdr = fits.Header()
    hdr["TILE"] = tid
    hdr["L_CEN"], hdr["B_CEN"] = patch.lc, patch.bc
    hdr["L_LO"], hdr["L_WIDTH"], hdr["B_LO"], hdr["B_HI"] = tile["l_lo"], tile["l_width"], tile["b_lo"], tile["b_hi"]
    hdr["HALF"] = "DE" if tile["de"] else "RU"
    hdr["SPLIT"] = tile["split"]
    hdr["COVERAGE"] = tile["coverage"]
    hdr["N_SRC"] = len(cat)
    hdr["HALOREJ"] = (bool(st["halo_rejected"]), "pathological sigmoid halo fit rejected")
    if st["halo_rejection"] is not None:
        hdr["HALOQA"] = (st["halo_rejection"].get("stage", "unknown"), "halo rejection stage")
    hdr["COMMENT"] = "Image cubes are (channel R,G,B; y; x) in display units of the RGB press image."
    hdus = [fits.PrimaryHDU(header=hdr)]
    for name, arr in [("DATA", data), ("MODEL", model), ("BACKGROUND", bkg), ("SUB", sub), ("RESIDUAL", resid)]:
        h = w.copy()
        h["BUNIT"] = "display"
        hdus.append(
            fits.CompImageHDU(np.moveaxis(arr, -1, 0).astype(np.float32), header=h, name=name, quantize_level=QUANTIZE)
        )
    hdus.append(fits.CompImageHDU(patch.valid.astype(np.uint8), header=w.copy(), name="VALID"))
    hdus.append(
        fits.CompImageHDU(
            np.moveaxis(st["modelled"].astype(np.uint8), -1, 0),
            header=w.copy(),
            name="MODELLED",
            quantize_level=0.0,
        )
    )
    hdus.append(
        fits.BinTableHDU(
            fits.FITS_rec.from_columns(
                fits.ColDefs(
                    [
                        fits.Column(
                            name=c,
                            format="30A" if c in ("NAME", "SNR_NAME") else "D",
                            array=cat[c].astype(str).values
                            if c in ("NAME", "SNR_NAME")
                            else cat[c].astype(float).values,
                        )
                        for c in [
                            "NAME",
                            "LII",
                            "BII",
                            "x",
                            "y",
                            "POS_ERR",
                            "FLUX_1",
                            "FLUX_1_LO",
                            "FLUX_1_HI",
                            "SNR",
                            "SAT_AREA_ARCMIN2",
                            "PSF_WIDTH",
                            "SNR_NAME",
                        ]
                    ]
                )
            ),
            name="SOURCES",
        )
    )
    fits.HDUList(hdus).writeto(fpath, overwrite=True)

    # PDF
    ncols = 4 if de else 3
    fig, axg = plt.subplots(2, ncols, figsize=figure_size(2, ncols))
    axs = axg[:, :2]
    ext = (-0.5, patch.npix - 0.5, -0.5, patch.npix - 0.5)
    outline = core_outline(tile, patch)
    panels = [(data, "Data"), (model, "Model"), (bkg, "Background")]
    for ax, (img, title) in zip(axs.ravel(), panels):
        ax.imshow(np.nan_to_num(img).astype(np.uint8), origin="lower", extent=ext)
        ax.set_title(title)
    lum = np.where(np.isnan(resid).all(-1), np.nan, np.nansum(resid, -1))
    im = axs[1, 1].imshow(lum, origin="lower", extent=ext, cmap="RdBu_r", vmin=-3 * RESID_RANGE, vmax=3 * RESID_RANGE)
    axs[1, 1].set_title("Residual")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.02)
    axg[0, 2].imshow(np.nan_to_num(sub).astype(np.uint8), origin="lower", extent=ext)
    axg[0, 2].set_title("Source-subtracted")
    axg[1, 2].axis("off")
    used = [*axs.ravel(), axg[0, 2]]
    if de:
        hext = ext  # the hires grid spans the same sky area; draw it in patch pixel units
        axg[0, 3].imshow(asinh_rgb(cube, HIRES_SOFT, HIRES_TOP, HIRES_SMOOTH), origin="lower", extent=hext)
        axg[0, 3].set_title('eRASS1-DE HiPS (0.2-0.5/0.5-1/1-2 keV), 30"/px + sources')
        psf_sigma = np.sqrt(calib["psf"]["t2"]) * HIRES_FACTOR
        axg[1, 3].imshow(asinh_rgb(cube, PSF_SOFT, PSF_TOP, psf_sigma), origin="lower", extent=hext)
        axg[1, 3].set_title("eRASS1-DE HiPS smoothed to the image PSF")
        axg[0, 3].scatter(cat.x, cat.y, marker="+", s=70, c="cyan", lw=1.2)
        used += [axg[0, 3], axg[1, 3]]
    for ax in used:
        ax.add_patch(Polygon(outline, closed=True, fill=False, ec="w", lw=0.8, ls="--"))
        ax.set_xlim(ext[:2])
        ax.set_ylim(ext[2:])
    ax = axs[0, 0]
    if tile["de"]:
        e = near(ref["erass1"], patch)
        ax.scatter(
            e.x,
            e.y,
            s=8 + 25 * np.clip(np.log10(e.ML_FLUX_1) + 13, 0, 3),
            facecolors="none",
            edgecolors=np.where(e.EXT > 0, "magenta", "yellow"),
            lw=0.6,
            label="eRASS1 (F>1e-13; magenta: extended)",
        )
    x = near(ref["xmmsl3"], patch)
    ax.scatter(x.x, x.y, marker="s", s=40, facecolors="none", edgecolors="orange", lw=0.9, label="XMMSL3")
    bs = near(ref["bsc"], patch)
    ax.scatter(bs.x, bs.y, marker="D", s=50, facecolors="none", edgecolors="lime", lw=0.9, label="RASS BSC")
    for s_ in near(ref["snr"], patch, "l", "b").itertuples():
        r = s_.size_arcmin / 2 / (patch.scale * 60)
        ax.add_patch(Ellipse((s_.x, s_.y), 2 * r, 2 * r, fill=False, ec="red", lw=1, ls=":"))
    for a in list(axs.ravel()[:3]) + [axg[0, 2]]:
        a.scatter(cat.x, cat.y, marker="+", s=70, c="cyan", lw=1.2, label="this catalogue" if a is ax else None)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    scale_px = 1.0 / patch.scale
    axs[0, 0].plot([3, 3 + scale_px], [3, 3], "w-", lw=2)
    axs[0, 0].text(3 + scale_px / 2, 5, "1 deg", color="w", ha="center")
    fig.tight_layout()
    save_figure(fig, ppath, keep_titles=True)
    plt.close(fig)
    return {
        "tile": tid,
        "hires": f"hires/tile_{tid:05d}_hires.fits" if de else "",
        "l_c": patch.lc,
        "b_c": patch.bc,
        "half": hdr["HALF"],
        "split": tile["split"],
        "n_src": len(cat),
        "resid_rms": float(np.sqrt(np.nanmean(lum**2))),
        "halo_rejected": bool(st["halo_rejected"]),
        "fits": str(fpath.relative_to(OUTDIR)),
        "pdf": str(ppath.relative_to(OUTDIR)),
    }


def main(limit=None, with_hires=False):
    """Run step 14."""
    (OUTDIR / "fits").mkdir(exist_ok=True)
    (OUTDIR / "pdf").mkdir(exist_ok=True)
    (OUTDIR / "hires").mkdir(exist_ok=True)
    tiles = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    if limit:
        tiles = tiles.sample(limit, random_state=0)
    det_params, calib = load_params()
    rows = [(t._asdict(), det_params, calib, with_hires) for t in tiles.itertuples(index=False)]
    with ProcessPoolExecutor(WORKERS) as ex:
        index = pd.DataFrame(list(ex.map(product, rows, chunksize=4)))
    index.to_csv(OUTDIR / ("index_sample.csv" if limit else "index.csv"), index=False)
    print(index.describe())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="process a random sample of this many tiles")
    ap.add_argument("--hires", action="store_true", help="fetch eRASS1-DE HiPS maps for DE tiles")
    a = ap.parse_args()
    main(a.limit, a.hires)
