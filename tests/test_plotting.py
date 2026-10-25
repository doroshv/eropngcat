"""Check TeX notation remains valid in labels with units and scientific numbers."""

import pytest
from matplotlib.mathtext import MathTextParser

from epngcat.plotting import tex_label


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3.4 deg2", r"3.4 $\mathrm{deg}^{2}$"),
        ("F50=5.4e-13", r"$F_{50}$=$5.4\times 10^{-13}$"),
        ("5.4e13 +- 2e+12", r"$5.4\times 10^{13}$ $\pm$ $2\times 10^{12}$"),
        (r"$\log_{10}(F_1/F_2)$", r"$\log_{10}(F_1/F_2)$"),
        ("[erg s$^{-1}$ cm$^{-2}$]", r"[$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}$]"),
    ],
)
def test_tex_labels_parse_and_are_idempotent(source, expected):
    result = tex_label(source)
    assert result == expected
    assert tex_label(result) == result
    MathTextParser("agg").parse(result)
