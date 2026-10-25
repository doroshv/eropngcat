"""Regression checks for incremental rebuilds and failure recovery."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_tree(tmp_path):
    """Create a tiny stand-in for the real DAG without copying scientific data."""
    text = (ROOT / "Makefile").read_text()
    (tmp_path / "Makefile").write_text(text)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/check_tile_products.py").write_text("pass\n")
    (tmp_path / "tools/stage_deps.py").write_text((ROOT / "tools/stage_deps.py").read_text())
    for path in ("pyproject.toml", "uv.lock", "eRASS1_Main.v1.2.fits", "xmmsl3_v1.0.fits", "erass1_allsky.png"):
        (tmp_path / path).touch()
    products = re.findall(r"out/[\w/]+\.[\w]+", text)
    for path in products:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
    stamps = tmp_path / "out/.stamps"
    stamps.mkdir(parents=True, exist_ok=True)
    for module in re.findall(r"call stage,(\w+)", text):
        (stamps / module).touch()
    return tmp_path


def run_make(root, *args):
    return subprocess.run(["make", *args], cwd=root, text=True, capture_output=True)


def test_missing_secondary_figure_rebuilds_producer(tmp_path):
    root = make_tree(tmp_path)
    assert "python -m epngcat." not in run_make(root, "-n", "all").stdout
    (root / "out/18_figures/fig11_lognlogs.pdf").unlink()
    result = run_make(root, "-n", "all")
    assert result.returncode == 0, result.stderr
    assert "python -m epngcat.s18_figures" in result.stdout


def test_missing_model_rebuilds_training_and_downstream(tmp_path):
    root = make_tree(tmp_path)
    (root / "out/09_models/models.pkl").unlink()
    result = run_make(root, "-n", "all")
    assert "python -m epngcat.s09_models" in result.stdout
    assert "python -m epngcat.s13_final_catalog" in result.stdout


def test_failed_stage_removes_completion_stamp(tmp_path):
    root = make_tree(tmp_path)
    (root / "out/18_figures/fig11_lognlogs.pdf").unlink()
    (root / "fail.py").write_text("raise SystemExit(1)\n")
    result = run_make(root, "all", f"PY={sys.executable} fail.py --module")
    assert result.returncode != 0
    assert not (root / "out/.stamps/s18_figures").exists()
    assert "python -m epngcat.s18_figures" in run_make(root, "-n", "all").stdout
