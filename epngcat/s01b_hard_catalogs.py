"""Step 01b: high-energy all-sky catalogues for the completeness checks of step 12.

Outputs (out/01_catalogs/):
  maxi_gsc.parquet MAXI/GSC 7-year catalogue, 4-10 keV (Hori+ 2018; HEASARC maxigsc7yr)
  maxi_ssc.parquet MAXI/SSC catalogue, 0.7-7 keV (Tomida+ 2016, 2016PASJ...68S..32T; HEASARC maxissccat,
                   from CDS J/PASJ/68/S32)
  rxte_xss.parquet RXTE all-sky slew survey, 3-20 keV (Revnivtsev+ 2004; HEASARC xteasscat)
Each table carries the SIMBAD position of its identification (l_id, b_id) where resolvable, and a
1-sigma positional uncertainty of the catalogue position (pos_err_arcmin).
"""

import numpy as np
import pandas as pd

from epngcat.config import DATA, step_dir

OUTDIR = step_dir("01_catalogs")


HARD = {
    "maxi_gsc": ("maxigsc7yr", "ctrpart_name"),
    "maxi_ssc": ("maxissccat", "alt_name"),
    "rxte_xss": ("xteasscat", "alt_names"),
}


def _heasarc(table):
    """Full HEASARC table (cached in data/)."""
    cache = DATA / f"heasarc_{table}.parquet"
    if not cache.exists():
        from astroquery.heasarc import Heasarc

        DATA.mkdir(exist_ok=True)
        df = Heasarc.query_tap(f"SELECT * FROM {table}").to_table().to_pandas()
        df[[c for c in df.columns if not c.startswith("__")]].to_parquet(cache)
    return pd.read_parquet(cache)


def _simbad(names):
    """SIMBAD positions (ra, dec) for identification names (cached in data/); NaN if unresolved."""
    cache = DATA / "simbad_ids.parquet"
    known = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["name", "ra", "dec"])
    todo = sorted(set(names) - set(known.name) - {""})
    if todo:
        from astroquery.simbad import Simbad

        rows = []
        for i in range(0, len(todo), 200):
            t = Simbad.query_objects(todo[i : i + 200])
            ra = np.ma.filled(np.ma.asarray(t["ra"], dtype=float), np.nan)
            dec = np.ma.filled(np.ma.asarray(t["dec"], dtype=float), np.nan)
            rows.append(pd.DataFrame({"name": [str(x).strip() for x in t["user_specified_id"]], "ra": ra, "dec": dec}))
        known = pd.concat([known, *rows], ignore_index=True).drop_duplicates("name")
        known.to_parquet(cache)
    return known.set_index("name").reindex(names)


def prepare_hard_catalogs():
    """MAXI/GSC, MAXI/SSC and RXTE/XSS with identification positions."""
    from astropy.coordinates import SkyCoord

    out = {}
    for name, (table, id_col) in HARD.items():
        df = _heasarc(table)
        ids = df[id_col].fillna("").astype(str).str.split(";").str[0].str.strip()
        pos = _simbad(ids.tolist())
        ra, dec = pos.ra.values, pos.dec.values
        if name == "maxi_gsc":  # the catalogue gives counterpart coordinates for most sources
            has = df.ctrpart_ra.notna().values
            ra = np.where(has, df.ctrpart_ra, ra)
            dec = np.where(has, df.ctrpart_dec, dec)
        ra, dec = np.asarray(ra, float), np.asarray(dec, float)
        ok = np.isfinite(ra)
        g = SkyCoord(np.where(ok, ra, 0), np.where(ok, dec, 0), unit="deg").galactic
        df["id_name"] = ids.values
        df["l_id"] = np.where(ok, g.l.deg, np.nan)
        df["b_id"] = np.where(ok, g.b.deg, np.nan)
        df = df.rename(columns={"lii": "l", "bii": "b"})
        # Positional uncertainty (arcmin, 1 sigma) of the catalogue position.
        if name == "maxi_gsc":
            df["pos_err_arcmin"] = np.hypot(df.error_radius.fillna(df.error_radius.median()) / 60, 3.0)
        elif name == "rxte_xss":
            df["pos_err_arcmin"] = df.error_radius / 2  # localisation radius ~ 2 sigma
        else:
            df["pos_err_arcmin"] = 0.3 * 60 / 2  # SSC localisation ~0.3 deg (Tomida+ 2016)
        df.to_parquet(OUTDIR / f"{name}.parquet")
        out[name] = (len(df), int(ok.sum()))
    return out


def main():
    """Run step 01b."""
    print("hard X-ray catalogues (n, n with identification position):", prepare_hard_catalogs())


if __name__ == "__main__":
    main()
