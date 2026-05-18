### The mental model to keep

  1. Audio → picture (mel spectrogram).
  2. Picture → fingerprint (Perch, or a CNN that learned to mimic Perch).
  3. Fingerprint → probabilities (compare to learned per-species prototypes / classifier head).
  4. Ensemble (average several models' opinions).
  5. Smooth across time (predictions for nearby windows should look similar).


  Gaps
  Loss function - BCE
  Adam Optimiser, LR(learning rate)
  Evaluation Metric - AUC ROC - https://www.kaggle.com/code/metric/birdclef-roc-auc
  Dataset
  Dataloaders
  
## What just happened

You trained a CNN to recognize bird species from 5-second audio clips.
The model:

1. Sees mel spectrograms ("pictures of sound").
2. Outputs 234 numbers per clip (one logit per species).
3. Learned by adjusting its millions of knobs to make those numbers match the true labels.

**Next step:** open `02_infer.ipynb` to load this checkpoint, run it on the soundscapes,
and write a `submission.csv`.

## Ideas to improve from here

In rough order of cost/benefit:

1. **Use the full data** — flip `DEBUG = False` and train on all 234 species.
2. **Train longer** — bump `EPOCHS` from 8 to 20-30. Watch val_auc; stop when it plateaus.
3. **Use secondary_labels** — multi-label your one-hot vector to include all species in a clip (free positive examples).
4. **Audio augmentation** — add background noise, random gain, pitch shift.
5. **Use the labeled soundscapes** — `train_soundscapes_labels.csv` has 60s soundscapes labeled per-window. That's closer to the test distribution than isolated training clips.
6. **Bigger backbone** — try `efficientnet_b1` or `resnet34`. Slower, better.
7. **K-fold ensemble** — train 5 models on different splits, average their predictions.
8. **Add Perch embeddings** — like the top notebook does. Big jump in quality.
