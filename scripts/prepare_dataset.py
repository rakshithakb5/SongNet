"""Build a CSV index for the local FMA-small dataset.

Expected layout:
data/raw/fma_small/
data/raw/fma_metadata/tracks.csv
"""

from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_loader import build_index, stratified_splits

METADATA = RAW_DATA_DIR / "fma_metadata" / "tracks.csv"
AUDIO = RAW_DATA_DIR / "fma_small"

def main():
    index = build_index(METADATA, AUDIO)
    train, validation, test = stratified_splits(index)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(PROCESSED_DATA_DIR / "index.csv", index=False)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"Indexed tracks: {len(index)}")
    print(f"Train: {len(train)} | Validation: {len(validation)} | Test: {len(test)}")

if __name__ == "__main__":
    main()
