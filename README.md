# BirdCLEF+ 2026 — minimal baseline

The simplest end-to-end pipeline:
audio file → 5s slice → mel spectrogram → CNN classifier → 234 species probabilities → submission CSV.

No ensembles, no Perch, no fancy heads. Get something working first, *then* improve.

## One-time setup

Python 3.12 is recommended (PyTorch's support for 3.13/3.14 lags). On macOS:

```bash
brew install python@3.12               # if you don't already have it
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Don't have Python 3.12? On macOS: `brew install python@3.12`.

## Quick smoke test (~10 min on CPU, 2-3 min on Apple Silicon GPU)

Trains on 10 species / 500 clips so you can watch the full loop end-to-end:

```bash
source .venv/bin/activate
jupyter lab
```

This opens a browser tab. Open the `notebooks/` folder.

## The two notebooks

1. **`notebooks/01_train.ipynb`** — Walk through training cell by cell. Starts in `DEBUG` mode
   (10 species, ~500 clips) so you can see the full loop in ~10 minutes on CPU before
   committing to a real run. Flip `DEBUG = False` once it works end-to-end.

2. **`notebooks/02_infer.ipynb`** — Loads the trained checkpoint, predicts on the soundscapes,
   writes `submission.csv`. Falls back to `train_soundscapes/` since `test_soundscapes/` is
   empty locally (Kaggle reveals real test audio only at submission time).

## File map

- `data/birdclef-2026/` — the unzipped competition data.
- `notebooks/01_train.ipynb`, `notebooks/02_infer.ipynb` — the baseline.
- `notebooks/birdclef-2026-eos-2/` — the top reference notebook (pulled from Kaggle, for study).
- `checkpoints/` — trained model weights (created by 01_train).
- `submission.csv` — output of 02_infer; what you'd upload to Kaggle.