"""Mel-spectrogram feature extraction for SongNet."""

import numpy as np
import librosa
from .config import CLIP_DURATION_SECONDS, HOP_LENGTH, N_MELS, SAMPLE_RATE

def load_audio(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION_SECONDS)
    return audio, sr

def mel_spectrogram(audio, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def extract_mel_from_file(path):
    audio, sr = load_audio(path)
    return mel_spectrogram(audio, sr)

def prepare_model_input(mel):
    return mel[..., np.newaxis]
