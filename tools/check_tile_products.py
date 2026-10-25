"""Check the dynamic products listed by the tile stage, without scientific dependencies."""

import csv
from pathlib import Path


def missing_products(index):
    """Return absent tile images/maps, including PDF previews."""
    index = Path(index)
    if not index.exists():
        return [index]
    missing = []
    with index.open() as stream:
        for row in csv.DictReader(stream):
            if "pdf" not in row:
                missing.append(index.parent / "pdf" / f"tile_{int(row['tile']):05d}.pdf")
            for key in ("fits", "pdf", "hires"):
                if row.get(key):
                    path = index.parent / row[key]
                    if not path.is_file():
                        missing.append(path)
    return missing


if __name__ == "__main__":
    if missing_products("out/14_tiles/index.csv"):
        print("missing")
