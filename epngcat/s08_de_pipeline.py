"""Step 08: run the final image-only pipeline (detection + measurements) on all DE tiles.

Outputs (out/08_de_pipeline/): de_pipeline.parquet (split and SNR-mask columns attached)
"""

import pandas as pd

from epngcat.config import OUT, step_dir
from epngcat.evaluate import in_snr
from epngcat.pipeline import process_tiles

OUTDIR = step_dir("08_de_pipeline")


def main():
    """Run step 08."""
    tiles = pd.read_parquet(OUT / "04_tiles" / "tiles.parquet")
    tiles = tiles[tiles.de]
    det = process_tiles(tiles)
    det["split"] = tiles.set_index("tile").split.reindex(det.tile).values
    det["in_snr"] = in_snr(det.l.values, det.b.values)
    det.to_parquet(OUTDIR / "de_pipeline.parquet")
    print(det.split.value_counts().to_dict(), "in SNR:", int(det.in_snr.sum()))


if __name__ == "__main__":
    main()
