"""Step 18: publication figures from the input image and saved catalogue/map products.

Write separate PDFs for the unannotated input image, tile detail, spatial split,
catalogue sky distribution, source density, number counts, and count ratios.
The two HEALPix maps are separate figures sharing one display stretch. Bright
source cutouts retain short panel identifiers. Captions belong in the manuscript.
Outputs are saved in out/18_figures/.
"""

import warnings

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from epngcat.config import OUT, is_german, load_json, save_json, step_dir
from epngcat.image import load_rgb
from epngcat.patches import tile_table
from epngcat.plotting import figure_size, save_figure
from epngcat.projection import SQ2, inverse_norm, load_alignment
from epngcat.s12_rosat import load as load_2rxs
from epngcat.sky import crossmatch, snr_mask

OUTDIR = step_dir("18_figures")
B_CUT = 20.0  # |b| of the number-counts sample: away from the Galactic plane
ZOOM = (120.0, -45.0, 14.0)  # (l, b, size) of the tile-grid inset [deg]
STRETCH_K = 30.0  # asinh hardness of the HEALPix rendering
SPLIT_COLORS = {"train": "#3b6ea5", "val": "#e8a33d", "test": "#c0392b", "ru": "0.75"}
# The brightest persistent X-ray sources saturate the press image over degrees and survive the
# point-source subtraction as blobs that dominate the clean map. They are named rather than
# selected by flux because the catalogues fail exactly here: the ROSAT BSC entry for Sco X-1, the
# brightest of them all, carries no count rate at all. Positions come from SIMBAD.
BRIGHT = [
    "Sco X-1", "Crab", "Cyg X-1", "Cyg X-2", "Cyg X-3", "Vela X-1", "Cen X-3", "Her X-1",
    "GX 5-1", "GX 9+9", "GX 17+2", "GX 349+2", "GX 340+0", "Ser X-1", "Cir X-1", "4U 1700-377",
    "LMC X-1", "SMC X-1", "Aql X-1", "GX 301-2",
]  # fmt: skip
# Extended sources large enough to survive a PSF-sized subtraction essentially intact. These are
# detected and catalogued - the cleaning removes 0-40% of them against 96% for a point source -
# so they are labelled, not masked: they are the real extragalactic sky, not an artefact.
EXTENDED_NAMED = [
    "Virgo Cluster", "Coma Cluster", "Centaurus Cluster", "Fornax Cluster", "Perseus Cluster",
    "Ophiuchus Cluster", "Hydra Cluster", "Norma Cluster", "Antlia Cluster", "Shapley Supercluster",
    "NGC 5128", "M31", "LMC", "SMC",
]  # fmt: skip
EXT_PSF_WIDTH = 3.0  # catalogue sources broader than this (and brighter than EXT_FLUX) are outlined
EXT_FLUX = 5e-12
MASK_R_DEFAULT = 1.0  # deg, if the source has no saturated blob of its own in the catalogue
MASK_PAD = 1.3  # blob radius is scaled by this before masking
SNR_MIN_ARCMIN = 12.0  # remnants smaller than this are invisible at all-sky scale
LABEL_MIN_DEG = 0.7  # only masked sources bigger than this are named, to keep the plane readable


def fig_input():
    """Write the unannotated input image and separate tile/split diagnostics."""
    psf_fwhm = np.sqrt(8 * np.log(2)) * load_json(OUT / "05_calibrate" / "calibration.json")["psf"]["sky_sigma_arcmin"]
    img = load_rgb()
    proj = load_alignment()
    ny, nx = img.shape[:2]
    fig = plt.figure(figsize=(figure_size(wide=True)[0], figure_size(wide=True)[0] * ny / nx))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img.astype(np.uint8), interpolation="nearest")
    ax.axis("off")
    save_figure(fig, OUTDIR / "fig01_input_image.pdf", layout=False)

    # Detail: the tile grid and the PSF at the same scale, on a piece of sky away from the plane.
    lc, bc, size = ZOOM
    fig, ax2 = plt.subplots(figsize=(figure_size()[0], 3.5))
    t = pd.DataFrame(tile_table())
    sel = t[(np.abs(((t.l_c - lc + 180) % 360) - 180) < size) & (np.abs(t.b_c - bc) < size)]
    for r in sel.itertuples():
        ll = np.linspace(r.l_lo, r.l_lo + r.l_width, 30)
        bb = np.linspace(r.b_lo, r.b_hi, 30)
        box_l = np.r_[ll, np.full(30, ll[-1]), ll[::-1], np.full(30, ll[0])]
        box_b = np.r_[np.full(30, bb[0]), bb, np.full(30, bb[-1]), bb[::-1]]
        x, y = proj.sky2pix(box_l % 360, box_b)
        ax2.plot(x, y, c="C0", lw=0.5, alpha=0.8)
    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    near = cat[(np.abs(((cat.LII - lc + 180) % 360) - 180) < size) & (np.abs(cat.BII - bc) < size)]
    sx, sy = proj.sky2pix(near.LII.values, near.BII.values)
    ax2.scatter(sx, sy, s=14, facecolors="none", edgecolors="C1", lw=0.7, label=f"catalogue sources (N={len(near)})")
    cx, cy = proj.sky2pix([lc], [bc])
    px_per_deg = np.hypot(*(np.array(proj.sky2pix([lc + 1], [bc])) - np.array([cx, cy]))).item()
    ax2.add_patch(
        plt.Circle(
            (cx[0], cy[0] + 0.35 * size * px_per_deg),
            psf_fwhm / 60 / 2 * px_per_deg,
            color="w",
            fill=False,
            lw=1.2,
        )
    )
    ax2.text(cx[0], cy[0] + 0.35 * size * px_per_deg - 0.9 * px_per_deg, "PSF FWHM", color="w", ha="center", fontsize=7)
    ax2.imshow(img.astype(np.uint8), interpolation="nearest", zorder=-1)
    ax2.set_xlim(cx[0] - size / 2 * px_per_deg, cx[0] + size / 2 * px_per_deg)
    ax2.set_ylim(cy[0] + size / 2 * px_per_deg, cy[0] - size / 2 * px_per_deg)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=7, labelcolor="w")
    ax2.set_title(f"{size:.0f} deg field: 2 × 2 deg tiles", fontsize=9)

    save_figure(fig, OUTDIR / "fig01_tile_grid.pdf")

    fig_sky_split()


def fig_sky_split():
    """Show sampled tile footprints at 20% opacity, projected once before drawing."""
    fig, ax3 = plt.subplots(figsize=(figure_size(wide=True)[0], 4.0), subplot_kw={"projection": "aitoff"})
    tt = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    for name, color in SPLIT_COLORS.items():
        polygons = []
        for tile in tt[tt.split == name].itertuples():
            polygons.extend((ax3.transProjection + ax3.transAffine).transform(p) for p in _tile_polygons(tile))
        ax3.add_collection(
            PolyCollection(
                polygons,
                facecolor=color,
                edgecolor="none",
                alpha=0.2,
                antialiased=False,
                rasterized=True,
                transform=ax3.transAxes,
            )
        )
    ax3.text(*_aitoff(90, 0), "RU (eastern)", ha="center", va="center", fontsize=10, color="0.3")
    ax3.text(*_aitoff(270, 0), "DE (western)", ha="center", va="center", fontsize=10, color="0.3")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.grid(alpha=0.3)
    ax3.set_title("Spatial train / validation / test split", fontsize=9, pad=6)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.96)
    save_figure(fig, OUTDIR / "fig01_sky_split.pdf", layout=False)
    plt.close(fig)


def _tile_polygons(tile, samples=8):
    """Sample a tile's sky footprint, splitting polygons at the Aitoff seam."""
    lo = (tile.l_lo + 180) % 360 - 180
    hi = lo + tile.l_width
    intervals = [(lo, min(hi, 180))]
    if hi > 180:
        intervals.append((-180, hi - 360))
    polygons = []
    for left, right in intervals:
        longitude = np.linspace(left, right, samples)
        latitude = np.linspace(tile.b_lo, tile.b_hi, samples)
        lon = np.r_[longitude, np.full(samples, right), longitude[::-1], np.full(samples, left)]
        lat = np.r_[np.full(samples, tile.b_lo), latitude, np.full(samples, tile.b_hi), latitude[::-1]]
        polygons.append(np.column_stack((-np.radians(lon), np.radians(lat))))
    return polygons


def _aitoff(l, b):
    """Galactic (l, b) [deg] -> Aitoff axes coordinates, l increasing to the left."""
    return -np.radians(((np.asarray(l) + 180) % 360) - 180), np.radians(np.asarray(b))


def _hammer_grid(nx, sup=1):
    """Galactic (lon, lat) of every pixel of a Hammer image, and the mask of pixels on the sky."""
    x = np.linspace(-2 * SQ2, 2 * SQ2, nx * sup)
    y = np.linspace(SQ2, -SQ2, nx // 2 * sup)
    lam, phi = inverse_norm(*np.meshgrid(x, y), "hammer")
    lon, lat = (-np.degrees(lam)) % 360.0, np.degrees(phi)  # l increases to the left
    return lon, lat, np.isfinite(lon)


def _graticule_overlay(ax, lon, lat, dl=60, db=30, c="w", alpha=0.25):
    """Draw a Galactic graticule on an imshow'd Hammer image (pixel coordinates)."""
    wrapped = ((lon + 180) % 360) - 180
    ax.contour(wrapped, levels=np.arange(-180 + dl, 180, dl), colors=c, linewidths=0.4, alpha=alpha)
    ax.contour(lat, levels=np.arange(-60, 90, db), colors=c, linewidths=0.4, alpha=alpha)
    ax.contour(np.isfinite(lon).astype(float), levels=[0.5], colors=c, linewidths=0.7, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def fig_skymap():
    """Figure 9: the catalogue on the sky, and its surface density per half."""
    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    fig, ax = plt.subplots(figsize=(figure_size(wide=True)[0], 4.0), subplot_kw={"projection": "aitoff"})
    for half, c in (("DE", "C0"), ("RU", "C3")):
        s = cat[half == cat.HALF]
        x, y = _aitoff(s.LII.values, s.BII.values)
        ax.scatter(x, y, s=0.8, c=c, alpha=0.5, lw=0, label=f"{half}  N={len(s)}")
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])
    ax.legend(loc="lower right", fontsize=9, markerscale=14, framealpha=0.6)
    ax.set_title(f"EIMG catalogue, {len(cat)} sources (Galactic, l increasing to the left)", fontsize=11, pad=18)

    save_figure(fig, OUTDIR / "fig09_sky_distribution.pdf")

    # HEALPix cells, so every cell has the same area and the poles carry no wedge artefacts.
    import healpy as hp

    fig, ax = plt.subplots(figsize=(figure_size(wide=True)[0], 4.0))
    nside = 16
    n = np.bincount(
        hp.ang2pix(nside, cat.LII.values, cat.BII.values, lonlat=True), minlength=hp.nside2npix(nside)
    ).astype(float)
    dens = n / hp.nside2pixarea(nside, degrees=True)
    lon, lat, good = _hammer_grid(1200)
    img = np.full(lon.shape, np.nan)
    img[good] = dens[hp.ang2pix(nside, lon[good], lat[good], lonlat=True)]
    m = ax.imshow(np.ma.masked_invalid(img), cmap="magma", norm=LogNorm(vmin=0.3, vmax=10), interpolation="nearest")
    _graticule_overlay(ax, lon, lat)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, label=r"Sources [deg$^{-2}$]")
    cb.set_ticks([0.3, 0.5, 1, 2, 3, 5, 10])
    cb.ax.set_xticklabels(["0.3", "0.5", "1", "2", "3", "5", "10"])
    ax.set_title(f"surface density in HEALPix cells of {hp.nside2pixarea(nside, degrees=True):.1f} deg2", fontsize=11)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "fig09_surface_density.pdf")
    plt.close(fig)


COUNT_NSIDE = 512
COUNT_JOIN_DEG = 0.5


def count_footprint(nside=COUNT_NSIDE):
    """Build one equal-area sky mask for source inclusion and count normalisation.

    Pixelise once, then apply this identical mask to each catalogue and to its area.
    The mask includes the actual rendered footprint, the catalogue edge margin,
    both joins, the latitude cut, and all SNRcat masks (including large remnants).
    """
    import healpy as hp
    from scipy import ndimage

    from epngcat.image import sky_mask
    from epngcat.s13_final_catalog import EDGE_PX

    projection = load_alignment()
    footprint = sky_mask()
    distance = ndimage.distance_transform_edt(footprint)
    keep = np.zeros(hp.nside2npix(nside), dtype=bool)
    de = np.zeros_like(keep)
    for start in range(0, len(keep), 262144):
        pixels = np.arange(start, min(start + 262144, len(keep)))
        lon, lat = hp.pix2ang(nside, pixels, lonlat=True)
        de[pixels] = is_german(lon)
        px, py = projection.sky2pix(lon, lat)
        margin = ndimage.map_coordinates(distance, [py, px], order=0, mode="constant", cval=0)
        valid = (
            (np.abs(lat) > B_CUT)
            & (margin >= EDGE_PX)
            & (np.abs(px - projection.de.x0) >= EDGE_PX)
            & (np.abs(((lon + 90) % 180) - 90) > COUNT_JOIN_DEG)
        )
        selected = np.flatnonzero(valid)
        valid[selected] &= snr_mask(lon[selected], lat[selected]) == ""
        keep[pixels] = valid
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    areas = {
        "DE": float(np.count_nonzero(keep & de) * pixel_area),
        "RU": float(np.count_nonzero(keep & ~de) * pixel_area),
    }
    return keep, areas


def in_count_footprint(lon, lat, mask):
    """Apply exactly the area mask to catalogue coordinates."""
    import healpy as hp

    return mask[hp.ang2pix(hp.npix2nside(len(mask)), np.asarray(lon), np.asarray(lat), lonlat=True)]


def _counts(flux, area, edges):
    """Cumulative N(>S) per deg2 and its Poisson error on a flux grid."""
    flux = np.asarray(flux)
    flux = flux[np.isfinite(flux) & (flux > 0)]
    n = np.array([(flux >= e).sum() for e in edges], float)
    return n / area, np.sqrt(n) / area


def fig_lognlogs():
    """Figure 11: number counts of both halves, eRASS1-DE and 2RXS at |b| > 20 deg."""
    mask, area = count_footprint()
    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    cat = cat[in_count_footprint(cat.LII, cat.BII, mask)]
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[in_count_footprint(er.LII, er.BII, mask) & is_german(er.LII.values)]
    rx = load_2rxs(geometry=False)
    rx = rx[in_count_footprint(rx.l, rx.b, mask)]
    area_2rxs = area["DE"] + area["RU"]
    save_json(
        {
            "nside": COUNT_NSIDE,
            "latitude_cut_deg": B_CUT,
            "join_exclusion_deg": COUNT_JOIN_DEG,
            "area_deg2": area,
            "snr_masks": "all",
            "source_counts": {"image": len(cat), "erass1": len(er), "2rxs": len(rx)},
        },
        OUTDIR / "number_counts.json",
    )

    edges = np.geomspace(2e-13, 3e-11, 40)
    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.5))
    series = {
        "image DE": (cat.FLUX_1[cat.HALF == "DE"], area["DE"], "C0", "-"),
        "image RU": (cat.FLUX_1[cat.HALF == "RU"], area["RU"], "C3", "-"),
        "eRASS1-DE (0.2-2.3 keV)": (er.ML_FLUX_1, area["DE"], "k", "--"),
        "2RXS (0.1-2.4 keV)": (rx.flux, area_2rxs, "C2", ":"),
    }
    out = {}
    rows = []
    for name, (f, a, c, ls) in series.items():
        n, e = _counts(f, a, edges)
        out[name] = n
        rows.extend(
            {
                "sample": name,
                "flux": float(fl),
                "area_deg2": a,
                "density_deg2": float(v),
                "poisson_error_deg2": float(err),
            }
            for fl, v, err in zip(edges, n, e)
        )
        ax.plot(edges, n, ls, c=c, label=f"{name}, {a:.0f} deg2")
        ax.fill_between(edges, n - e, n + e, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$S$ [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel(r"$N(>S)$ [deg$^{-2}$]")
    ax.set_title(rf"Number counts at $|b| > {B_CUT:.0f}^\circ$", fontsize=11)
    ax.grid(False, which="both")
    ax.legend(fontsize=8)

    pd.DataFrame(rows).to_csv(OUTDIR / "number_counts.csv", index=False)
    save_figure(fig, OUTDIR / "fig11_lognlogs.pdf")

    fig, ax = plt.subplots(figsize=(figure_size()[0], 3.5))
    ok = out["image DE"] > 0
    ax.axhline(1, c="k", lw=1)
    ax.plot(edges[ok], out["image RU"][ok] / out["image DE"][ok], c="C3", label="image RU / image DE")
    ref_ok = out["eRASS1-DE (0.2-2.3 keV)"] > 0
    ax.plot(
        edges[ref_ok],
        out["image DE"][ref_ok] / out["eRASS1-DE (0.2-2.3 keV)"][ref_ok],
        c="C0",
        label="image DE / eRASS1",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$S$ [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_ylabel(r"Ratio of $N(>S)$")
    ax.set_title("Number-count ratios", fontsize=11)
    ax.grid(False, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, OUTDIR / "fig11_count_ratios.pdf")
    plt.close(fig)


def _hammer_rgb(maps, nx=1600, sup=2, scale=None):
    """Render three HEALPix channels as an RGB Hammer image; returns the image and its scaling.

    `scale` reuses the levels of another map. Scaling each map on its own percentiles would make a
    before/after pair dishonest: the point-source-subtracted map has a lower bright end, so its own
    99.8th percentile stretches the faint sky harder and the leftovers look brighter than they are.
    """
    import healpy as hp

    ny = nx // 2
    lon, lat, good = _hammer_grid(nx, sup)
    out = np.full((ny * sup, nx * sup, 3), np.nan)
    pix = hp.ang2pix(hp.npix2nside(maps.shape[1]), lon[good], lat[good], lonlat=True)
    for k in range(3):
        v = maps[k][pix]
        out[..., k][good] = np.where(v < -1e30, np.nan, v)  # hp.UNSEEN
    # The clean map masks each channel over its own clipped extent (step 16), so a pixel can have
    # red but not blue. In a colour composite that reads as a colour change rather than as missing
    # data, so a pixel is shown only where all three channels survive.
    out[np.isnan(out).any(-1)] = np.nan
    # Average the supersampled grid so that nside-1024 pixels are not aliased.
    with warnings.catch_warnings():  # a fully masked cell averages to NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(out.reshape(ny, sup, nx, sup, 3), axis=(1, 3))
    if scale is None:
        scale = [np.nanpercentile(out[..., k], [30, 99.8]) for k in range(3)]
    rgb = np.zeros_like(out)
    for k in range(3):
        lo, hi = scale[k]
        s = np.clip((out[..., k] - lo) / (hi - lo), 0, 1)
        rgb[..., k] = np.arcsinh(STRETCH_K * s) / np.arcsinh(STRETCH_K)
    return np.nan_to_num(rgb), scale


def _sky_to_image(l, b, nx):
    """Galactic (l, b) [deg] -> pixel coordinates of a `_hammer_grid(nx)` image."""
    from epngcat.projection import forward_norm

    lam = -np.radians(((np.asarray(l) + 180) % 360) - 180)  # l increases to the left
    x, y = forward_norm(lam, np.radians(b), "hammer")
    return (x + 2 * SQ2) / (4 * SQ2) * (nx - 1), (SQ2 - y) / (2 * SQ2) * (nx // 2 - 1)


def _circle(l0, b0, r_deg, n=180):
    """Sample a small circle of angular radius r_deg around (l0, b0)."""
    pa = np.linspace(0, 2 * np.pi, n)
    r, b0r, l0r = np.radians(r_deg), np.radians(b0), np.radians(l0)
    b = np.arcsin(np.sin(b0r) * np.cos(r) + np.cos(b0r) * np.sin(r) * np.cos(pa))
    l = l0r + np.arctan2(np.sin(pa) * np.sin(r) * np.cos(b0r), np.cos(r) - np.sin(b0r) * np.sin(b))
    return np.degrees(l) % 360, np.degrees(b)


def _plot_circle(ax, l0, b0, r_deg, nx, **kw):
    """Draw a small circle on a Hammer image, split where it wraps across the l = 180 seam."""
    x, y = _sky_to_image(*_circle(l0, b0, r_deg), nx)
    x = np.where(np.abs(np.diff(x, prepend=x[0])) > nx / 4, np.nan, x)
    ax.plot(x, y, **kw)


def named_positions(names):
    """SIMBAD positions of a list of objects in Galactic coordinates; empty if SIMBAD is unreachable."""
    from astropy.coordinates import SkyCoord

    from epngcat.s01b_hard_catalogs import _simbad

    try:
        pos = _simbad(names)
    except Exception as e:  # a figure must not depend on a name resolver being up
        print(f"  SIMBAD unavailable ({e}); {len(names)} named objects skipped")
        return pd.DataFrame(columns=["name", "l", "b"])
    ok = pos.ra.notna().values
    g = SkyCoord(pos.ra.values[ok], pos.dec.values[ok], unit="deg").galactic
    return pd.DataFrame({"name": np.array(names)[ok], "l": g.l.deg, "b": g.b.deg})


def bright_sources():
    """Positions of the brightest persistent X-ray sources, which are masked out of the clean map."""
    return named_positions(BRIGHT)


def large_extended():
    """Catalogue sources broad and bright enough to survive the PSF-sized subtraction."""
    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    sel = cat.FLAG_EXTENDED & (cat.PSF_WIDTH > EXT_PSF_WIDTH) & (cat.FLUX_1 > EXT_FLUX)
    return cat[sel].reset_index(drop=True)


def mask_bright(m, src, nside):
    """Blank the saturated halo of each bright source in a HEALPix map; returns the masked radii.

    The radius is the source's own saturated blob in the catalogue, so a source that did not
    saturate the image is not masked more than it needs.
    """
    import healpy as hp

    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    sat = cat[cat.SAT_AREA_ARCMIN2 > 0]
    idx, _ = crossmatch(src.l.values, src.b.values, sat.LII.values, sat.BII.values, 1.5 * 3600)
    out = []
    for i, r in enumerate(src.itertuples()):
        area = sat.SAT_AREA_ARCMIN2.values[idx[i]] if idx[i] >= 0 else np.nan
        rad = MASK_PAD * np.sqrt(area / np.pi) / 60 if np.isfinite(area) else MASK_R_DEFAULT
        m[:, hp.query_disc(nside, hp.ang2vec(r.l, r.b, lonlat=True), np.radians(rad))] = np.nan
        out.append({"name": r.name, "l": r.l, "b": r.b, "r_deg": float(rad)})
    return out


def supernova_remnants():
    """SNRcat remnants large enough to be visible on an all-sky map."""
    snr = pd.read_parquet(OUT / "01_catalogs" / "snrcat.parquet")
    return snr[snr.size_arcmin >= SNR_MIN_ARCMIN].reset_index(drop=True)


def fig_healpix(nside=1024):
    """Figure A7: the calibrated all-sky HEALPix maps, with and without point sources."""
    import healpy as hp

    nx = 1600
    src, snr = bright_sources(), supernova_remnants()
    ext, named = large_extended(), named_positions(EXTENDED_NAMED)
    scale, masked = None, []
    for name, title in (
        ("data", "Calibrated all-sky image"),
        ("clean_filled", "Source-subtracted all-sky image"),
    ):
        fig, ax = plt.subplots(figsize=(figure_size(wide=True)[0], figure_size(wide=True)[0] / 2))
        m = np.asarray(hp.read_map(OUT / "16_healpix" / f"healpix_{name}_nside{nside}.fits", field=None, dtype=None))
        if name.startswith("clean") and len(src):
            m = m.astype(float)
            masked = mask_bright(m, src, nside)
        rgb, scale = _hammer_rgb(m, nx=nx, scale=scale)
        ax.imshow(rgb, interpolation="nearest")
        _graticule_overlay(ax, *_hammer_grid(nx)[:2])
        if name.startswith("clean"):
            # Remnants are the one extended population the PSF-only scene model cannot remove, so
            # outline them rather than mask them: most of what is left in the plane sits inside one.
            for r in snr.itertuples():
                _plot_circle(ax, r.l, r.b, r.size_arcmin / 120, nx, c="#4fc3f7", lw=0.5, alpha=0.8)
            # Broad catalogue sources: detected, but a PSF-sized subtraction barely touches them.
            for r in ext.itertuples():
                _plot_circle(ax, r.LII, r.BII, 0.6, nx, c="#ffd54f", lw=0.5, alpha=0.8)
            for r in named.itertuples():
                _plot_circle(ax, r.l, r.b, 1.2, nx, c="#ffd54f", lw=0.9)
                x, y = _sky_to_image(r.l, r.b, nx)
                ax.annotate(
                    r.name.replace(" Cluster", "").replace(" Supercluster", ""),
                    (x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color="#ffd54f",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="k")],
                )
            for v in masked:
                _plot_circle(ax, v["l"], v["b"], v["r_deg"], nx, c="#ff7043", lw=0.7)
                if v["r_deg"] < LABEL_MIN_DEG:  # the small holes would only clutter the plane
                    continue
                x, y = _sky_to_image(v["l"], v["b"], nx)
                ax.annotate(
                    v["name"],
                    (x, y),
                    xytext=(0, 6 + v["r_deg"] * nx / 360),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="w",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="k")],
                )
        ax.set_title(title)
        suffix = "data" if name == "data" else "clean"
        save_figure(fig, OUTDIR / f"figA7_healpix_{suffix}.pdf")


def _cutout(l, b, half_deg=0.75):
    """Return cutout images, their Galactic WCS, local source pixels and the target pixel."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from epngcat.patches import tile_of

    path = OUT / "14_tiles" / "fits" / f"tile_{int(tile_of(l, b)):05d}.fits"
    if not path.exists():
        return None
    with fits.open(path) as h:
        w = WCS(h["DATA"].header).celestial
        x0, y0 = (float(v) for v in w.all_world2pix(l, b, 0))
        n = int(round(half_deg / abs(h["DATA"].header["CDELT1"])))
        sl = (slice(max(int(y0) - n, 0), int(y0) + n + 1), slice(max(int(x0) - n, 0), int(x0) + n + 1))
        imgs = {k: np.moveaxis(h[k].data, 0, -1)[sl] for k in ("DATA", "MODEL", "SUB")}
        src = pd.DataFrame(
            np.asarray(h["SOURCES"].data).byteswap().view(np.asarray(h["SOURCES"].data).dtype.newbyteorder("="))
        )
    src = src.assign(plot_x=src.x - sl[1].start, plot_y=src.y - sl[0].start)
    center = (x0 - sl[1].start, y0 - sl[0].start)
    ny, nx = imgs["DATA"].shape[:2]
    inside = src.plot_x.between(-0.5, nx - 0.5) & src.plot_y.between(-0.5, ny - 0.5)
    return imgs, w.slice(sl), src[inside], center


def _in_saturated_blob(l, b, cat, pad=1.5):
    """Flag positions falling inside the saturated blob of any catalogue source."""
    from scipy.spatial import cKDTree

    from epngcat.sky import radec_unit

    sat = cat[cat.SATURATED & (cat.SAT_AREA_ARCMIN2 > 0)]
    tree = cKDTree(radec_unit(np.asarray(l), np.asarray(b)))
    out = np.zeros(len(l), bool)
    for lc, bc, area in zip(sat.LII.values, sat.BII.values, sat.SAT_AREA_ARCMIN2.values):
        r = pad * np.sqrt(area / np.pi) / 60  # blob radius in degrees
        out[tree.query_ball_point(radec_unit([lc], [bc])[0], 2 * np.sin(np.radians(r) / 2))] = True
    return out


def bright_cases():
    """Pick one bright eRASS1 source per bright-end failure mode, from the data.

    The modes are what the completeness measurement points at: bright sources are recovered as well
    as faint ones while they are point-like and unsaturated, and the apparent bright-end decline is
    the rising fraction of extended sources, of saturated cores, and of blends.
    """
    from epngcat.config import is_german
    from epngcat.evaluate import in_snr

    cat = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    cat = cat[cat.HALF == "DE"].reset_index(drop=True)
    er = pd.read_parquet(OUT / "01_catalogs" / "erass1.parquet")
    er = er[np.isfinite(er.ML_FLUX_1) & (er.ML_FLUX_1 > 0) & is_german(er.LII.values)].reset_index(drop=True)
    er = er[~in_snr(er.LII.values, er.BII.values)].reset_index(drop=True)
    idx, sep = crossmatch(er.LII.values, er.BII.values, cat.LII.values, cat.BII.values, 4 * 60.0)
    er["found"], er["sep"], er["match"] = idx >= 0, sep, idx
    er["sat"] = np.where(idx >= 0, cat.SATURATED.values[idx], False)
    # A blend: several bright eRASS1 sources whose nearest image source is the same one.
    bright = er.ML_FLUX_1 > 1e-12
    shared, counts = np.unique(er.match[bright & er.found], return_counts=True)
    er["blend"] = er.match.isin(shared[counts >= 2]) & bright
    # "Inside a saturated halo" has to be tested against the blob's own extent, not its centroid:
    # in the Sco X-1 region the field is white for degrees while the nearest blob centroid is 40'
    # away, and such a cutout shows nothing at all.
    clear = ~_in_saturated_blob(er.LII.values, er.BII.values, cat)
    rules = (
        (
            "point-like, unsaturated: subtracted cleanly",
            (er.EXT_LIKE == 0) & er.found & clear & (er.ML_FLUX_1 > 3e-12),
        ),
        ("point-like but saturated: the core carries no flux information", (er.EXT_LIKE == 0) & er.found & er.sat),
        ("extended in eRASS1: only a PSF-sized core is removed", (er.EXT_LIKE > 10) & clear & (er.ML_FLUX_1 > 2e-12)),
        ("blend: two bright eRASS1 sources, one image detection", er.blend & clear),
    )
    out, seen = [], set()
    for label, sel in rules:
        for r in er[sel].nlargest(6, "ML_FLUX_1").itertuples():
            if r.IAUNAME not in seen:
                out.append((label, r))
                seen.add(r.IAUNAME)
                break
    return out


def fig_bright_residuals():
    """Figure A8: bright-source cutouts with Galactic axes and one header per column."""
    cases = bright_cases()
    fig = plt.figure(figsize=figure_size(len(cases), 3))
    headers = ("Data", "Background + source model", "Source-subtracted")
    for row, (_, source) in enumerate(cases):
        cut = _cutout(source.LII, source.BII)
        if cut is None:
            continue
        imgs, wcs, src, center = cut
        for col, key in enumerate(("DATA", "MODEL", "SUB")):
            ax = fig.add_subplot(len(cases), 3, row * 3 + col + 1, projection=wcs)
            ax.imshow(np.clip(imgs[key], 0, 255).astype(np.uint8), origin="lower", interpolation="nearest")
            ax.scatter(src.plot_x, src.plot_y, s=40, marker="o", facecolors="none", edgecolors="#ff7043", lw=0.7)
            ax.plot(*center, "+", c="#4fc3f7", ms=9, mew=1.2)
            longitude, latitude = ax.coords
            for coord in (longitude, latitude):
                coord.set_major_formatter("d.dd")
                coord.set_ticks(number=3)
                coord.set_ticklabel(size=8, exclude_overlapping=True)
            longitude.set_axislabel(r"$l$", minpad=0.6)
            latitude.set_axislabel(r"$b$" if col == 0 else "", minpad=0.6)
            latitude.set_ticklabel_visible(col == 0)
            if row == 0:
                ax.set_title(headers[col])
            if row == 0 and col == 0:
                ax.plot([], [], "+", c="#4fc3f7", label="eRASS1 position")
                ax.scatter([], [], marker="o", facecolors="none", edgecolors="#ff7043", label="image catalogue")
                ax.legend(loc="upper right", fontsize=7, labelcolor="w")
    save_figure(fig, OUTDIR / "figA8_bright_residuals.pdf", keep_titles=True)


def main():
    """Run step 18."""
    for f in (fig_input, fig_skymap, fig_lognlogs, fig_healpix, fig_bright_residuals):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
