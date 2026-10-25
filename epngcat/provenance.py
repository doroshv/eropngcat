"""Content fingerprints and a compact record of the completed Make run."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def fingerprint(paths):
    """Hash ordered input paths and bytes without loading large images into memory."""
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main():
    """Record code, locked environment, input files, and retained HiPS cutouts."""
    from epngcat.config import DATA, ERASS1_FITS, IMAGE, OUT, ROOT, SEED, WORKERS, XMMSL_FITS, save_json

    def files(paths):
        records = {}
        for path in sorted(paths):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            records[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": checksum}
        return records

    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = [s.split(">=")[0] for s in project["project"]["dependencies"]]
    code = [
        *ROOT.glob("epngcat/*.py"),
        *ROOT.glob("tools/*.py"),
        ROOT / "Makefile",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    inputs = [IMAGE, ERASS1_FITS, XMMSL_FITS, *DATA.glob("*")]
    # Preserve the downloaded/reprojected HiPS inputs used in stage 15. A clean
    # build can fetch them, but reproducing this run uses these exact cutouts.
    hires = sorted((OUT / "14_tiles/hires").glob("*.fits"))
    record = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "command": "make all",
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "seed": SEED,
        "workers": WORKERS,
        "wings_enabled": bool(int(os.environ.get("EPNGCAT_WINGS", "1"))),
        "source_versions": {
            "eRASS1": ERASS1_FITS.name,
            "XMMSL3": XMMSL_FITS.name,
            "5XMM": "5XMM-DR15 v1.0, prepared bright-source subset",
            "2RXS": "VizieR J/A+A/588/A103, cached in data/rass_2rxs.parquet",
            "HiPS": "eRASS1-DE DR1 Rate_c010 bands 025/026/027, order 6",
        },
        "code": files(code),
        "inputs": files(inputs),
        "hires_inputs": files(hires),
        "results": files(
            [
                OUT / "10_validate_de/metrics.json",
                OUT / "13_final/erass1_image_catalog.csv",
                OUT / "13_final/uncertainty_calibration.json",
                OUT / "17_orphans/orphans.json",
                OUT / "18_figures/number_counts.json",
            ]
        ),
    }
    save_json(record, OUT / "run_manifest.json")
    print(f"Recorded {len(record['inputs'])} inputs and {len(hires)} retained HiPS cutouts", flush=True)


if __name__ == "__main__":
    main()
