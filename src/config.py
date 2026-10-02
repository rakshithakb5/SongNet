from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLE_RATE = 22050
CLIP_DURATION_SECONDS = 30
N_MELS = 128
HOP_LENGTH = 512

GENRES = ["Electronic", "Pop", "Experimental", "Rock", "Folk", "Instrumental", "Hip-Hop", "International"]
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}
