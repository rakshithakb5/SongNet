"""SongNet - real-time music genre classification (Streamlit demo)

Run from the project folder (where songnet.ipynb is):   streamlit run app.py
Needs the trained model at outputs/songnet.pt
"""
import os, time, tempfile
import numpy as np, pandas as pd
import streamlit as st

# draw the page FIRST, so it appears immediately (heavy libraries load afterwards)
st.set_page_config(page_title='SongNet', layout='wide')
st.title('SongNet: real-time music genre classification')
st.caption('CNN + GRU trained on FMA-small (8 genres) | Team 21, UE24CS352A Machine Learning, PES University')

MODEL_PATH = 'outputs/songnet.pt'
SR, N_FFT, HOP, N_MELS = 22050, 2048, 512, 128      # same settings as in the notebook
STEP_SEC = 8 * HOP / SR                              # one model step = 0.186 s


@st.cache_resource(show_spinner='Loading PyTorch, librosa and the model (first time only, ~30 s)...')
def load_everything():
    """heavy imports + model loading, done once and then kept in memory"""
    import torch, torch.nn as nn, torch.nn.functional as F
    import librosa

    class SongNet(nn.Module):                        # same model as in the notebook
        def __init__(self, n_genres=8):
            super().__init__()
            def block(c_in, c_out):
                return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(c_out), nn.ReLU(),
                                     nn.MaxPool1d(2), nn.Dropout(0.3))
            self.cnn = nn.Sequential(block(N_MELS, 256), block(256, 256), block(256, 256))
            self.gru = nn.GRU(256, 128, num_layers=2, batch_first=True, dropout=0.3)
            self.out = nn.Linear(128, n_genres)

        def forward(self, x):
            h = self.cnn(x)
            h, _ = self.gru(h.transpose(1, 2))
            return self.out(h)

    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model = SongNet(len(ckpt['genres']))
    model.load_state_dict(ckpt['model'])
    model.eval()

    # warm-up: the first spectrogram is slow (librosa compiles some code), so do it now
    librosa.feature.melspectrogram(y=np.zeros(SR, dtype=np.float32), sr=SR,
                                   n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return dict(torch=torch, F=F, librosa=librosa, model=model,
                genres=ckpt['genres'], mean=ckpt['mean'], std=ckpt['std'])


def predict_over_time(path, L, seconds=30):
    """genre probabilities averaged over everything heard so far, at every 0.19 s step"""
    librosa, torch, F = L['librosa'], L['torch'], L['F']
    y, _ = librosa.load(path, sr=SR, mono=True, duration=seconds)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS), ref=np.max)
    x = torch.from_numpy((mel - L['mean']) / L['std']).float()[None]
    with torch.no_grad():
        step_probs = F.softmax(L['model'](x), dim=-1)[0].numpy()
    running = step_probs.cumsum(0) / np.arange(1, len(step_probs) + 1)[:, None]
    seconds_axis = np.round(np.arange(1, len(running) + 1) * STEP_SEC, 2)
    return running, seconds_axis


# ---------------- page ----------------
if not os.path.exists(MODEL_PATH):
    st.error(f'Model not found at {MODEL_PATH}. Run "streamlit run app.py" from the project folder '
             f'(the one containing the outputs folder). Current folder: {os.getcwd()}')
    st.stop()

L = load_everything()
GENRES = L['genres']

uploaded = st.file_uploader('Upload a song (MP3, WAV, ...)', type=['mp3', 'wav', 'ogg', 'flac', 'm4a'])
if uploaded is None:
    st.info('Upload a song to start. The model will guess its genre and update the guess as the song plays.')
    st.stop()

suffix = os.path.splitext(uploaded.name)[1]          # save upload to a temp file so librosa can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
    f.write(uploaded.getvalue())
    path = f.name

with st.spinner('Analysing the song...'):
    running, t = predict_over_time(path, L)

if st.button('Play and classify live', type='primary'):
    st.audio(uploaded.getvalue(), autoplay=True)
    left, right = st.columns([3, 1])
    left.subheader('Probability of each genre so far (%)')
    chart = left.empty()
    right.subheader('Current guess')
    guess = right.empty()
    bars = right.empty()

    start = time.time()
    while True:
        n = min(len(running), max(1, int((time.time() - start) / STEP_SEC)))
        best = running[n - 1].argmax()
        chart.line_chart(pd.DataFrame(running[:n] * 100, index=t[:n], columns=GENRES), height=380)
        guess.markdown(f'## {GENRES[best]}\n**{running[n - 1, best] * 100:.0f}%** after {t[n - 1]:.1f} s')
        bars.bar_chart(pd.Series(running[n - 1] * 100, index=GENRES), height=260)
        if n >= len(running):
            break
        time.sleep(0.5)

    st.success(f'Final answer: **{GENRES[running[-1].argmax()]}** ({running[-1].max() * 100:.0f}%)')