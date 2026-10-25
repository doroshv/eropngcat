"""Shared A&A-style figures, based on paperdraft/plot_script_example.py.

Use native Matplotlib text so a clean build does not require a TeX installation.
STIX supplies a bundled Times-like serif and matching mathematical glyphs.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation, Text

COLUMN_IN = 255.76535 / 72.27
PAGE_IN = 2 * COLUMN_IN
ASPECT = 1.2 * (5**0.5 - 1) / 2


def use_style():
    """Apply the reference typography and print-quality export defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.usetex": False,  # TeX math is rendered by MathText; no external TeX binary is required.
            "axes.formatter.use_mathtext": True,
            "figure.figsize": [COLUMN_IN, COLUMN_IN * ASPECT],
            "axes.linewidth": 0.6,
            "lines.linewidth": 1,
            "lines.markersize": 3,
            "scatter.marker": ".",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figure_size(rows=1, columns=1, *, wide=False):
    """Size a panel grid in inches at the intended publication width."""
    width = PAGE_IN if wide or columns > 1 else COLUMN_IN
    height = width / columns * ASPECT + (0.7 if columns > 1 else 0)
    return width, max(2.8 if columns > 1 else 2.6, height) * rows


def tex_label(value):
    """Typeset numbers and units consistently, preserving existing TeX math.

    All labels pass through this function, including legends and annotations.
    Scientific notation is converted only outside existing dollar-delimited math.
    """
    units = {
        "erg s$^{-1}$ cm$^{-2}$": r"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}$",
        "erg cm$^{-2}$ s$^{-1}$": r"$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$",
        "erg s-1 cm-2": r"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}$",
        "counts s-1 arcmin-2": r"$\mathrm{counts}\,\mathrm{s}^{-1}\,\mathrm{arcmin}^{-2}$",
        "ct s$^{-1}$ arcmin$^{-2}$": r"$\mathrm{ct}\,\mathrm{s}^{-1}\,\mathrm{arcmin}^{-2}$",
    }
    for plain, math in units.items():
        value = value.replace(plain, math)
    parts = re.split(r"(\$[^$]*\$)", value)
    for i in range(0, len(parts), 2):
        part = parts[i]
        part = re.sub(
            r"(?<![\w.])([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)\b",
            lambda m: rf"${m[1]}\times 10^{{{int(m[2])}}}$",
            part,
        )
        part = part.replace("+-", r"$\pm$").replace("->", r"$\rightarrow$").replace("×", r"$\times$")
        part = re.sub(
            r"\b(deg2|keV|arcsec|arcmin|deg|dex)\b",
            lambda m: r"$\mathrm{deg}^{2}$" if m[1] == "deg2" else rf"$\mathrm{{{m[1]}}}$",
            part,
        )
        part = re.sub(r"\bF(50|90)\b", lambda m: rf"$F_{{{m[1]}}}$", part)
        part = re.sub(r"\bsigma\b", lambda _: r"$\sigma$", part)
        part = re.sub(r"(?<=\d)sigma\b", lambda _: r"$\sigma$", part)
        parts[i] = part
    return "".join(parts)


def _wrap_label(label, max_width, renderer):
    """Wrap text to its measured width without breaking TeX expressions."""
    math = []

    def protect(match):
        math.append(match[0])
        return f"MATHPLACEHOLDER{len(math) - 1}END"

    protected = re.sub(r"\$[^$]*\$", protect, label.get_text())

    def restore(value):
        return re.sub(r"MATHPLACEHOLDER(\d+)END", lambda m: math[int(m[1])], value)

    lines = []
    for original in protected.split("\n"):
        line = ""
        for word in original.split():
            candidate = f"{line} {word}".strip()
            display = restore(candidate)
            width = renderer.get_text_width_height_descent(display, label.get_fontproperties(), "$" in display)[0]
            if line and width > max_width:
                lines.append(restore(line))
                line = word
            else:
                line = candidate
        lines.append(restore(line))
    label.set_text("\n".join(lines))


def _place_annotations(fig):
    """Place map names inside their axes without overlapping other names."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        occupied = []
        for annotation in ax.texts:
            if not isinstance(annotation, Annotation) or not annotation.get_visible():
                continue
            original = annotation.get_position()
            candidates = [
                original,
                (0, -14),
                (20, -14),
                (-20, -14),
                (30, 12),
                (-30, 12),
                (0, 25),
                (0, -28),
                (45, 0),
                (-45, 0),
            ]
            for position in candidates:
                annotation.set_position(position)
                box = annotation.get_window_extent(renderer).expanded(1.08, 1.15)
                if (
                    ax.bbox.contains(box.x0, box.y0)
                    and ax.bbox.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in occupied)
                ):
                    occupied.append(box)
                    break
            else:
                annotation.set_visible(False)


def save_figure(fig, path, *, layout=True, keep_titles=False):
    """Write a PDF with vector text and 300-dpi rasterised dense points.

    Keep the physical canvas width fixed (no tight bounding-box cropping). Wrap
    plain-text annotations before the final layout pass; math labels stay intact.
    """
    if not keep_titles:
        for ax in fig.axes:
            for title in (ax.title, ax._left_title, ax._right_title):
                title.set_text("")
        if fig._suptitle is not None:
            fig._suptitle.set_text("")
    for label in fig.findobj(match=Text):
        label.set_text(tex_label(label.get_text()))
    renderer = fig.canvas.get_renderer()
    width = fig.get_figwidth()
    for ax in fig.axes:
        panel_width = ax.get_position().width * width
        titles = (ax.title, ax._left_title, ax._right_title)
        for label in (*titles, ax.xaxis.label, ax.yaxis.label):
            if label in titles:
                label.set_fontsize(9)
            span = panel_width if label is not ax.yaxis.label else ax.get_position().height * fig.get_figheight()
            _wrap_label(label, span * fig.dpi, renderer)
        for collection in ax.collections:
            if isinstance(collection, PathCollection) and len(collection.get_offsets()) > 500:
                collection.set_rasterized(True)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(10)
        _wrap_label(fig._suptitle, width * fig.dpi * 0.95, renderer)
    if layout:
        fig.tight_layout(pad=0.8)
    _place_annotations(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


use_style()
