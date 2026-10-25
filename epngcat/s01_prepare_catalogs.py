"""Step 01: slim the input catalogues and fetch the ROSAT Bright Source Catalogue.

Outputs (out/01_catalogs/):
  erass1.parquet   eRASS1-DE main catalogue, columns needed downstream
  xmmsl3.parquet   XMM slew survey (clean detections, soft 0.2-2 keV band = B6; FLUX_B6 and
                   FLUX_B6_ERR are converted here to erg/s/cm2, as every other catalogue)
  rass_bsc.parquet ROSAT all-sky survey bright source catalogue (VizieR IX/10A)
  xmm_dr15.parquet 5XMM-DR15 clean point sources (HEASARC xmmssc) with a 0.2-2 keV flux
  snrcat.parquet   Galactic supernova remnants (SNRcat, Ferrand & Safi-Harb 2012) with mask radii
"""

import numpy as np
import pandas as pd
from astropy.io import fits

from epngcat.config import DATA, ERASS1_FITS, XMM_DR15_FITS, XMMSL_FITS, step_dir
from epngcat.sky import sep_arcsec

OUTDIR = step_dir("01_catalogs")

ERASS_COLS = ["IAUNAME", "RA", "DEC", "LII", "BII", "POS_ERR", "EXT", "EXT_LIKE", "DET_LIKE_0", "ML_EXP_1"]
ERASS_COLS += [f"{p}_{b}" for b in ["1", "P1", "P2", "P3"] for p in ["ML_FLUX", "ML_FLUX_ERR", "ML_RATE", "ML_BKG"]]
ERASS_COLS += ["DET_LIKE_P1", "DET_LIKE_P2", "DET_LIKE_P3"]
ERASS_COLS += ["FLAG_SP_SNR", "FLAG_SP_BPS", "FLAG_SP_SCL", "FLAG_SP_LGA", "FLAG_SP_GC_CONS", "FLAG_OPT"]

# XMMSL band names: B6 = soft 0.2-2 keV, B7 = hard 2-12 keV, B8 = total 0.2-12 keV. The soft
# band is the one that overlaps the image, so B6 is what we keep.
XMM_COLS = ["UNIQUE_SRCNAME", "RA", "DEC", "RADEC_ERR", "LII", "BII", "DATE_OBS", "FLUX_B6", "FLUX_B6_ERR"]
XMM_COLS += ["FLUX_B8", "FLUX_B8_ERR", "DET_ML_B6", "EXT_B6", "VAL_FLAG", "VER_FALSE", "VER_HALO", "VER_INEXT"]
# TUNIT of every XMMSL flux column is "(10E-12)ergs/s/cm2": FLUX_B6 has a median of 1.0 against
# the known ~6e-13 erg/s/cm2 slew soft-band limit, so the scale is 1e-12.
XMM_FLUX_UNIT = 1e-12


def _table(path, cols):
    data = fits.open(path, memmap=True)[1].data
    out = {}
    for c in cols:
        v = np.asarray(data[c])
        out[c] = v.astype(v.dtype.newbyteorder("=")) if v.dtype.kind != "S" else np.char.strip(v).astype(str)
    return pd.DataFrame(out)


def prepare_erass1():
    """Slim eRASS1-DE main catalogue."""
    df = _table(ERASS1_FITS, ERASS_COLS)
    df.to_parquet(OUTDIR / "erass1.parquet")
    return len(df)


def prepare_xmmsl3():
    """Keep clean XMMSL3 detections with a soft-band flux."""
    df = _table(XMMSL_FITS, XMM_COLS)
    good = (
        (df.VAL_FLAG == "CLEAN_SAMPLE")
        & ~df.VER_FALSE
        & (df.DET_ML_B6 > 10)
        & np.isfinite(df.FLUX_B6)
        & (df.FLUX_B6 > 0)
    )
    df = df[good].reset_index(drop=True)
    for c in ("FLUX_B6", "FLUX_B6_ERR", "FLUX_B8", "FLUX_B8_ERR"):
        df[c] *= XMM_FLUX_UNIT
    df.to_parquet(OUTDIR / "xmmsl3.parquet")
    return int(good.sum())


# 5XMM-DR15 EPIC bands 1, 2, 3 are 0.2-0.5, 0.5-1.0 and 1.0-2.0 keV: their sum is exactly the
# 0.2-2 keV band of the image catalogue's FLUX_P1 + FLUX_P2 + FLUX_P3, so no band conversion is
# needed on either side. sum_flag < 2 keeps detections without a serious quality warning and
# extent = 0 keeps the point sources.
# 5XMM-DR15 EPIC bands 1, 2, 3 are 0.2-0.5, 0.5-1.0 and 1.0-2.0 keV: their sum is exactly the
# 0.2-2 keV band of the image catalogue's FLUX_P1 + FLUX_P2 + FLUX_P3, so no band conversion is
# needed on either side. SUM_FLAG < 2 keeps detections without a serious quality warning and
# EXTENT = 0 keeps the point sources.
#
# The released catalogue is 3.4M detections over 421 columns (~8 GB). Only the columns below are
# read, and only detections above XMM_FLUX_MIN are kept, which is well under the image's own
# completeness limit; the result caches to a few MB in data/ so the download can be deleted.
XMM_DR15_COLS = [
    "SRCID", "IAUNAME", "RA", "DEC", "LII", "BII", "RADEC_ERR", "EXTENT", "SUM_FLAG", "EP_DET_ML",
    "EP_1_FLUX", "EP_1_FLUX_ERR", "EP_2_FLUX", "EP_2_FLUX_ERR", "EP_3_FLUX", "EP_3_FLUX_ERR",
]  # fmt: skip
XMM_FLUX_MIN = 1e-13  # erg/s/cm2, 0.2-2 keV
XMM_CHUNK = 250_000  # detections read at a time


def _xmm_dr15_rows():
    """Yield 5XMM-DR15 detections passing the quality and flux cuts, a chunk of rows at a time."""
    with fits.open(XMM_DR15_FITS, memmap=True) as h:
        d = h[1].data
        n = len(d)
        for lo in range(0, n, XMM_CHUNK):
            sl = slice(lo, min(lo + XMM_CHUNK, n))
            col = {c: np.asarray(d[c][sl]) for c in XMM_DR15_COLS}
            f = np.stack([col[f"EP_{i}_FLUX"].astype(float) for i in (1, 2, 3)])
            e = np.stack([col[f"EP_{i}_FLUX_ERR"].astype(float) for i in (1, 2, 3)])
            flux, err = np.nansum(f, 0), np.sqrt(np.nansum(e**2, 0))
            keep = (col["SUM_FLAG"] < 2) & (col["EXTENT"] == 0) & np.isfinite(flux) & (flux > XMM_FLUX_MIN)
            if not keep.any():
                continue
            out = {c: col[c][keep] for c in ("SRCID", "IAUNAME", "RA", "DEC", "LII", "BII", "RADEC_ERR", "EP_DET_ML")}
            out["flux_02_2"], out["flux_02_2_err"] = flux[keep], err[keep]
            df = pd.DataFrame(out)
            yield df.astype({c: df[c].dtype.newbyteorder("=") for c in df.columns if df[c].dtype.kind in "fiu"})


def prepare_xmm_dr15():
    """Filter 5XMM-DR15 to bright clean point sources (cached in data/), one row per source."""
    cache = DATA / "xmm_dr15_bright.parquet"
    if not cache.exists():
        if not XMM_DR15_FITS.exists():
            raise SystemExit(f"missing {XMM_DR15_FITS.name}: download 5XMM-DR15 or restore {cache}")
        DATA.mkdir(exist_ok=True)
        pd.concat(list(_xmm_dr15_rows()), ignore_index=True).to_parquet(cache)
    det = pd.read_parquet(cache)
    # One row per source: the catalogue lists a detection per observation.
    g = det.groupby("SRCID")
    df = pd.DataFrame(
        {
            "name": g.IAUNAME.first(),
            "ra": g.RA.median(),
            "dec": g.DEC.median(),
            "lii": g.LII.median(),
            "bii": g.BII.median(),
            "radec_err": g.RADEC_ERR.median(),
            "flux_02_2": g.flux_02_2.median(),
            "flux_02_2_err": g.flux_02_2_err.median(),
            "n_det": g.size(),
        }
    ).reset_index()
    df.to_parquet(OUTDIR / "xmm_dr15.parquet")
    return len(df)


def rass_flux(count_rate, hr1):
    """ROSAT count rate -> 0.1-2.4 keV flux, HR1-dependent conversion of Schmitt et al. (1995)."""
    return np.asarray(count_rate) * (5.30 * np.asarray(hr1) + 8.31) * 1e-12


def prepare_rass_bsc():
    """Download the ROSAT BSC (cached in data/)."""
    cache = DATA / "rass_bsc_IX10A_1rxs.fits"
    if not cache.exists():
        from astroquery.vizier import Vizier

        DATA.mkdir(exist_ok=True)
        tab = Vizier(columns=["**"], row_limit=-1).get_catalogs("IX/10A/1rxs")[0]
        tab.write(cache, overwrite=True)
    t = fits.open(cache)[1].data
    df = pd.DataFrame({c: np.asarray(t[c]).astype(np.asarray(t[c]).dtype.newbyteorder("=")) for c in t.names})
    df = df.rename(columns={"1RXS": "NAME", "RAJ2000": "RA", "DEJ2000": "DEC"})
    df["NAME"] = df["NAME"].astype(str).str.strip()
    df["FLUX"] = rass_flux(df["Count"], df["HR1"])
    df.to_parquet(OUTDIR / "rass_bsc.parquet")
    return len(df)


SNRCAT_URL = "http://snrcat.physics.umanitoba.ca/SNRdownload.php?table=SNR"
SNR_DEFAULT_SIZE = 10.0  # arcmin, when SNRcat gives no size
SNR_MASK_SCALE, SNR_MASK_PAD = 1.2, 6.0  # mask radius = scale * size / 2 + pad [arcmin]


def prepare_snrcat():
    """Download SNRcat (cached in data/) and derive Galactic positions and mask radii."""
    import io
    import urllib.request

    from astropy.coordinates import SkyCoord

    cache = DATA / "snrcat.csv"
    if not cache.exists():
        DATA.mkdir(exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(SNRCAT_URL, timeout=60).read())
    text = cache.read_bytes().decode("utf-8", errors="replace")
    body = text[text.index("G;") :]  # skip the download banner
    df = pd.read_csv(io.StringIO(body), sep=";")
    c = SkyCoord(df["J2000_ra (hh:mm:ss)"], df["J2000_dec (dd:mm:ss)"], unit=("hourangle", "deg"))
    size = pd.to_numeric(df["size_coarse (arcmin)"], errors="coerce").fillna(SNR_DEFAULT_SIZE)
    # Galactic position from the G-name (0.1 deg precision); the J2000 columns are more precise but
    # wrong for a few entries (e.g. Antlia), so use them only where both agree within 0.2 deg.
    gl = df["G"].str.slice(1, 6).astype(float).values
    gb = df["G"].str.slice(6, 11).astype(float).values
    jl, jb = c.galactic.l.deg, c.galactic.b.deg
    agree = sep_arcsec(gl, gb, jl, jb) < 0.2 * 3600
    out = pd.DataFrame(
        {
            "name": df["G"],
            "alt": df["id_alt"].fillna(""),
            "type": df["type"].fillna(""),
            "l": np.where(agree, jl, gl),
            "b": np.where(agree, jb, gb),
            "pos_from_name": ~agree,
            "size_arcmin": size,
            "mask_radius_arcmin": SNR_MASK_SCALE * size / 2 + SNR_MASK_PAD,
        }
    )
    out.to_parquet(OUTDIR / "snrcat.parquet")
    return len(out)


def main():
    """Run step 01."""
    print("eRASS1:", prepare_erass1())
    print("XMMSL3 clean:", prepare_xmmsl3())
    print("5XMM-DR15 clean point sources:", prepare_xmm_dr15())
    print("RASS BSC:", prepare_rass_bsc())
    print("SNRcat:", prepare_snrcat())


if __name__ == "__main__":
    main()
