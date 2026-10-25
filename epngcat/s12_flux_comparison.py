"""Publication flux comparisons from saved catalogues, without refitting models.

External panels use the final catalogue (including its RU flux-scale correction).
The eRASS1 panel uses precisely the base-model DE test flux sample from step 10.
Write the plotted pairs as well as the figure so selections remain inspectable.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from epngcat.config import OUT, step_dir
from epngcat.evaluate import catalog, match_detections
from epngcat.plotting import figure_size, save_figure
from epngcat.s12_external import MATCH_IMG, external
from epngcat.s12_rosat import RADIUS
from epngcat.s12_rosat import load as load_2rxs
from epngcat.sky import crossmatch

OUTDIR = step_dir("12_external")
UNITS = r"$[\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$"


def external_pairs(reference, image, radius, name):
    """Nearest image match for each external source, retaining finite positive fluxes."""
    rows = []
    for half in ("DE", "RU"):
        ref = reference[reference.half == half].reset_index(drop=True)
        cat = image[half == image.HALF].reset_index(drop=True)
        idx, sep = crossmatch(ref.l.values, ref.b.values, cat.LII.values, cat.BII.values, radius)
        ok = idx >= 0
        pairs = pd.DataFrame(
            {
                "reference": name,
                "half": half,
                "source_id": cat.NAME.values[idx[ok]],
                "reference_flux": ref.flux.values[ok],
                "image_flux": cat.FLUX_1.values[idx[ok]],
                "separation_arcsec": sep[ok],
            }
        )
        rows.append(pairs)
    return pd.concat(rows, ignore_index=True)


def load_pairs():
    """Recreate the documented external and held-out eRASS1 selections."""
    image = pd.read_csv(OUT / "13_final" / "erass1_image_catalog.csv", low_memory=False)
    image = image[image.SNR_NAME.isna() | (image.SNR_NAME == "")]
    rx = load_2rxs()
    rosat = external_pairs(rx[~rx.extended], image, RADIUS, "2RXS")
    slew = external_pairs(external("xmmsl3"), image, MATCH_IMG, "XMMSL3")

    de = pd.read_parquet(OUT / "10_validate_de" / "de_catalog.parquet")
    test = de[(de.SPLIT == "test") & ~de.IN_SNR_MASK].reset_index(drop=True)
    matched = match_detections(test.rename(columns={"LII": "l", "BII": "b"}), catalog("test"))
    pt = matched[(matched.cat_EXT == 0) & (matched.sep < 150) & (matched.cat_ML_FLUX_1 > 0)]
    erass = pd.DataFrame(
        {
            "reference": "eRASS1",
            "half": "DE",
            "source_id": pt.cat_IAUNAME.values,  # eRASS1 ID; base product has no final EIMG name
            "reference_flux": pt.cat_ML_FLUX_1.values,
            "image_flux": pt.FLUX_1.values,
            "separation_arcsec": pt.sep.values,
        }
    )
    pairs = pd.concat([rosat, slew, erass], ignore_index=True)
    valid = np.isfinite(pairs.reference_flux) & np.isfinite(pairs.image_flux)
    return pairs[valid & (pairs.reference_flux > 0) & (pairs.image_flux > 0)].reset_index(drop=True)


def plot(pairs):
    """Three horizontal log-flux panels, with dots and a common ordinate scale."""
    fig, axes = plt.subplots(1, 3, figsize=(figure_size(wide=True)[0], 2.85), sharey=True)
    panels = (("2RXS", "0.1–2.4"), ("XMMSL3", "0.2–2.0"), ("eRASS1", "0.2–2.3"))
    # Enclose every plotted pair; use identical flux limits to keep slopes comparable.
    values = pairs[["reference_flux", "image_flux"]].to_numpy()
    limits = 10 ** np.array([np.floor(np.log10(values.min())), np.ceil(np.log10(values.max()))])
    for ax, (name, band) in zip(axes, panels):
        ax.plot(limits, limits, color="0.3", lw=0.8, zorder=1)
        for half, colour in (("DE", "C0"), ("RU", "C3")):
            sample = pairs[(pairs.reference == name) & (pairs.half == half)]
            if sample.empty:
                continue
            label = f"{half}" + (" test" if name == "eRASS1" else "") + rf" ($N={len(sample)}$)"
            ax.scatter(
                sample.reference_flux,
                sample.image_flux,
                marker=".",
                s=3,
                linewidths=0,
                color=colour,
                alpha=0.3,
                rasterized=True,
                label=label,
            )
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.set_xlabel(rf"$F_{{\mathrm{{{name}}}}}$ ({band} keV)" + "\n" + UNITS)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="upper left", fontsize=7, markerscale=3, handletextpad=0.2, borderaxespad=0.3)
        ax.grid(False)
    axes[0].set_ylabel(r"$F_{\mathrm{image}}$ (0.2–2.3 keV)" + "\n" + UNITS)
    save_figure(fig, OUTDIR / "flux_comparison.pdf")


def main():
    """Build the paper figure and its exact plotted-pair table from saved products."""
    pairs = load_pairs()
    pairs.to_csv(OUTDIR / "flux_comparison_pairs.csv", index=False)
    plot(pairs)
    print(pairs.groupby(["reference", "half"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
