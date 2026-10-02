"""FMA-small dataset indexing and split utilities."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import GENRES, GENRE_TO_INDEX


def load_fma_metadata(metadata_path):
    """Load FMA tracks metadata and keep the eight target genres."""
    tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    genre_top = tracks[("track", "genre_top")]
    selected = tracks.loc[genre_top.isin(GENRES)].copy()
    selected["genre"] = genre_top.loc[selected.index]
    return selected


def audio_path(raw_audio_dir, track_id):
    """Map an FMA track ID to its two-level FMA path."""
    track_id = int(track_id)
    return Path(raw_audio_dir) / f"{track_id // 1000:03d}" / f"{track_id:05d}.mp3"


def build_index(metadata_path, raw_audio_dir):
    tracks = load_fma_metadata(metadata_path)
    rows = []
    for track_id, row in tracks.iterrows():
        path = audio_path(raw_audio_dir, track_id)
        if path.exists():
            rows.append({
                "track_id": int(track_id),
                "path": str(path),
                "genre": row["genre"],
                "label": GENRE_TO_INDEX[row["genre"]],
            })
    return pd.DataFrame(rows)


def stratified_splits(index_df, seed=42):
    """Create 70/20/10 train/validation/test splits."""
    train, temp = train_test_split(
        index_df, test_size=0.30, stratify=index_df["label"], random_state=seed
    )
    validation, test = train_test_split(
        temp, test_size=1/3, stratify=temp["label"], random_state=seed
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
