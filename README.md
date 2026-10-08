# SongNet: Real-time Music Genre Classification

**UE24CS352A Machine Learning, Mini-Project, PES University**
Section 5E · Team 21 · Problem #33

| USN | Name |
|---|---|
| PES2UG24CS272 | Mayukh Vinay |
| PES2UG24CS918 | Rakshitha K B |

A reproduction of *SongNet: Real-time Music Classification* (Chi Zhang, Yue Zhang, Chen Chen, Stanford CS229, 2018), with a live demo app added.
[Original report](https://cs229.stanford.edu/proj2018/report/53.pdf) · [Original poster](https://cs229.stanford.edu/proj2018/poster/53.pdf)

The model listens to a song and predicts its genre **every 0.19 seconds**, using only the audio heard so far, so the prediction can update live while the song plays.

---

## Results (test set: 800 songs never seen in training)

| Model | Input | Test accuracy |
|---|---|---|
| Random guess | – | 12.5% |
| KNN | 140 MFCC statistics | 50.4% |
| MLP | 140 MFCC statistics | 49.5% |
| Logistic Regression | 140 MFCC statistics | 53.5% |
| SVM | 140 MFCC statistics | 59.5% |
| **SongNet (ours, CNN + GRU)** | **mel-spectrogram** | **63.0%** |
| SongNet (Stanford paper) | mel-spectrogram | 65.2% |

**Per genre (SongNet, test set):**

| Genre | Precision | Recall | F1 |
|---|---|---|---|
| Hip-Hop | 0.809 | 0.720 | 0.762 |
| Rock | 0.653 | 0.770 | 0.706 |
| International | 0.643 | 0.740 | 0.688 |
| Folk | 0.673 | 0.700 | 0.686 |
| Instrumental | 0.576 | 0.680 | 0.624 |
| Electronic | 0.591 | 0.650 | 0.619 |
| Experimental | 0.557 | 0.490 | 0.521 |
| Pop | 0.500 | 0.290 | 0.367 |

**Findings**
- SongNet beats every baseline, including the strong SVM (+3.5 points), and is 2.2 points below the original paper.
- As in the paper, **Pop** and **Experimental** are the hardest genres. Pop is confused with Rock, International and Folk because it borrows from many styles; Experimental is often confused with Instrumental.
- Hip-Hop is sometimes confused with Electronic (both use programmed beats).
- Training accuracy reached ~77% while validation levelled off at ~60%, so the model overfits somewhat. Early stopping, dropout and random-crop augmentation limit this.

Charts: [`outputs/training_curves.png`](outputs/training_curves.png) · [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png) · [`outputs/example_spectrograms.png`](outputs/example_spectrograms.png) · [`outputs/genre_distribution.png`](outputs/genre_distribution.png)

---

## Dataset

[**FMA: Free Music Archive**](https://github.com/mdeff/fma) (Defferrard et al., ISMIR 2017), `fma_small` subset:
- 8,000 clips of 30 seconds, 8 balanced genres with 1,000 each: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- Download: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MB) and [fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip) (7.2 GB, SHA1 `ade154f733639d52e35e32f5593efe5be76c6d70`)
- The dataset is **not** included in this repo because of its size. Unzip both files into `data/`.

## Method

**1. Data cleaning**
- Checked for missing genre labels, duplicate tracks and missing MP3 files: none found.
- Classes are perfectly balanced (1,000 per genre), so no resampling was needed.
- 6 corrupt or too-short audio files were detected and removed automatically (tracks 98565, 98567, 98569, 99134, 108925, 133297), leaving **7,994 songs**.

**2. Preprocessing**
- Load each clip at 22,050 Hz, mono, 30 seconds.
- Convert it to a **log-mel-spectrogram**: 128 mel bands, 2,048-sample window, 512-sample hop, in decibels.
- Pad or trim to 1,280 frames (about 29.7 s), giving one 128 × 1,280 "picture" per song.
- Stratified **70 / 20 / 10** split into train (5,595), validation (1,599) and test (800), with seed 42.
- Normalise with the mean and standard deviation of the **training** set only (−43.9 dB, 16.3 dB), to avoid data leakage.

**3. Model: SongNet (C-RNN, 742,152 parameters)**
```
mel-spectrogram (128 x time)
 -> 3 x [Conv1D (256 filters, kernel 3) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout 0.3]
 -> GRU (2 layers, 128 units, left to right)
 -> Linear -> 8 genre scores at every time step (one step = 0.186 s)
 -> clip prediction = average of the per-step probabilities
```
Because the GRU only reads left to right, the prediction at time *t* uses only audio up to *t*. That is what makes real-time classification possible.

**4. Training**
- Loss: negative log-likelihood of the time-averaged probability.
- Optimiser: Adam (learning rate 1e-3, weight decay 1e-4), batch size 32.
- ReduceLROnPlateau halves the learning rate when validation loss stalls for 3 epochs.
- Early stopping after 15 epochs without improvement. The best validation accuracy was **60.5%** at epoch 39, and training stopped at epoch 54.
- **Data augmentation:** each training step uses a random 15-second excerpt of each song.
- Trained on a Google Colab T4 GPU (about 10 s per epoch; about 5.5 min per epoch on a laptop CPU).

**5. Baselines:** KNN, Logistic Regression, MLP and SVM (scikit-learn, standardised inputs) on the 140 MFCC statistics provided in FMA's `features.csv`, using the same train/test split.

---

## Repository contents

```
songnet.ipynb        main notebook: cleaning, preprocessing, baselines, model, evaluation, demo
app.py               Streamlit web app: upload a song and watch the genre prediction live
requirements.txt     Python libraries
outputs/
  songnet.pt         trained model (best epoch)
  results.csv        test accuracy of all models
  history.csv        loss and accuracy per epoch
  split.npz          train/val/test indices + normalisation values
  tracks_used.csv    the 7,994 cleaned tracks with genres
  *.png              charts
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

- **Live demo (no dataset needed):** `streamlit run app.py`, then upload any MP3 and click **Play and classify live**.
- **Full pipeline:** download FMA into `data/` (see Dataset), open `songnet.ipynb` and run the cells from top to bottom. Training on a CPU takes several hours, so we trained the same cells on Google Colab's GPU (see the notebook).

## References
1. C. Zhang, Y. Zhang, C. Chen. *SongNet: Real-time Music Classification.* Stanford CS229, 2018.
2. M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson. *FMA: A Dataset for Music Analysis.* ISMIR 2017.
3. K. Choi, G. Fazekas, M. Sandler, K. Cho. *Convolutional Recurrent Neural Networks for Music Classification.* ICASSP 2017.
4. B. McFee et al. *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015.

## Acknowledgements
 All experiments were run, and all results analysed, by the team.