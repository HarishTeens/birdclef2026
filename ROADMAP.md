# BirdCLEF+ 2026 — Roadmap

The plan for taking the model from a working baseline toward a medal. Each phase has a status, the moves it contains, and an expected score range. Phases are layered: each one *adds* to the previous, doesn't replace it.

Top score to beat: **0.963**. The leaderboard is brutally compressed — top 30% sits at 0.946+, top 5% sits at 0.949+ (yes, the top is *that* tight). Realistic target: **bronze/silver medal (top 5-10%)**, requires **~0.94+**. Beating top requires either flawless execution of every phase + a novel idea, or major luck.

**Current standing**: LB-best is **0.842** (combined 10-model ensemble = 5 seeds + 5 K-folds). Stacking two different diversity sources gave +0.003 over either alone (0.839 → 0.842). Total trajectory: 0.497 → 0.714 (P0) → 0.772 (P1) → 0.836 (P3) → 0.839 (P4) → 0.842 (combined).

---

## Status at a glance

| Phase | Description | Realistic LB target | Status |
|---|---|---|---|
| 0 | Baseline pipeline | 0.50 → 0.71 | ✅ shipped, LB **0.714** |
| 1 | Distribution alignment | 0.78-0.85 | ✅ shipped, LB **0.772** (soundscape val 0.882 — gap 0.11) |
| 2a | SpecAugment + REPLICAS=4 | 0.80-0.83 | ❌ regressed, LB **0.757** (-0.015 from Phase 1, see lessons below) |
| 2b | Background mixing (clip + soundscape ambience) | 0.79-0.83 | ❌ regressed, LB **0.696** (epoch-1 ckpt; mixing introduced label noise) |
| 2c-e | SpecAugment isolated / energy trimming / mixup / TTA | TBD | ⏸ deferred (augmentation not paying off — Perch is the better lever) |
| 3 | Perch v2 foundation model | 0.85-0.89 | ✅ shipped, LB **0.836** (+0.064 from Phase 1; val/LB gap *shrank* from 0.110 to 0.092 — Perch generalizes better) |
| 3.5 | Head tweaks (LR sched / arch / mixup / ensemble) | 0.85-0.87 | ✅ shipped, LB **0.833** (-0.003, within noise — val gain was below noise floor) |
| 4 | Sound Event Detection (SED) head | 0.88-0.92 | ✅ shipped, LB **0.839** (new best, +0.003 vs Phase 3; val 0.9401, gap 0.101). Skipped variants — moving to K-fold for diversity gain. |
| 5 | Iterative refinement (pseudo-labeling) | 0.89-0.92 | ❌ 3 attempts, all within ±0.01 of Phase 3.5. Pseudo-labels from the Phase 3.5 ensemble don't carry new info. Closed. |
| 6 | K-fold ensemble | 0.91-0.94 | ✅ shipped, LB **0.839** (tied with Phase 4 SED — same diversity gain, different source) |
| 6b | Combined ensemble (5 seeds + 5 folds = 10 models) | TBD | ✅ shipped, LB **0.842** (new best, +0.003 over Phase 4/K-fold's 0.839) |
| 4-variant | GeMFreq pooling over freq dim | TBD | ❌ ensemble val 0.9397 (≈ Phase 4's 0.9401). Learned p ≈ 3 across seeds — mean pool was approximately optimal. Not shipped. |
| 7 | Calibration & post-processing | +0.005-0.02 | ⏳ planned |
| 8 | Novel edge / lottery tickets | wildcard | ⏳ open-ended |

LB targets revised downward after Phase 1 — the val/LB gap turned out to be ~0.11 (not ~0.03 as in Phase 0), so my earlier estimates were rosy. Bronze medal (top ~10%) requires ~0.94+, which means we need Phases 2-6 to all land near the top of their ranges and stack reasonably well.

Update this table after each phase ships.

---

## Phase 0 — Baseline pipeline ✅

Goal: working end-to-end submission. Everything below this depends on it.

- Single CNN (`efficientnet_b0`, ImageNet pretrained, in_chans=1, 234-way output).
- Train on `train_audio/` clips, random 5s crop, one-hot primary label.
- Validate on held-out 15% of clips (`train.csv` split, stratified by species).
- Inference: 60s soundscape → 12 windows → sigmoid → Gaussian smooth in logit space → `submission.csv`.
- Submission flow: checkpoint → Kaggle Dataset → Kaggle Notebook → Submit.

**LB**: 0.714.
**Files**: `notebooks/01_train.ipynb` (cells 0-23), `notebooks/02_infer.ipynb`, `kaggle-uploads/baseline-{ckpt,submit}/`.

---

## Phase 1 — Distribution alignment ✅

Goal: make the training distribution look more like the test distribution. This is where 80% of the early-stage gains come from.

**Shipped**: LB **0.772** (+0.058 over Phase 0). Soundscape val_auc 0.882 at epoch 3 (kernel crashed before later epochs but the "save best" logic preserved the peak).

**Lessons learned**:
- The +0.058 LB gain was real and meaningful, but ~42% of the val gain. Local val (0.882) overstates the model's true LB by ~0.11.
- Cause **revised after Phase 2a**: it's NOT primarily memorization. The 20-file val is too small and too closely related to the 46-file train portion to be a faithful LB proxy. Improvements on this val don't necessarily generalize to the *different* recording sessions in `test_soundscapes/`.

---

## Phase 2a — SpecAugment + REPLICAS=4 ❌

Goal: close the val/LB gap by reducing memorization of specific train soundscape files. Combined SpecAugment (regularization) + reducing soundscape replicas from 10 → 4.

**Result**: regressed. LB 0.757 (−0.015 from Phase 1). Local soundscape val 0.9057 (+0.023 over Phase 1's 0.8824 baseline-on-this-val). Gap *widened* to 0.149.

**What went wrong**:
- Two changes bundled into one experiment — couldn't attribute success/failure.
- The REPLICAS reduction was likely the culprit: with 10 replicas, the model saw ~10,000 soundscape examples per epoch. With 4 replicas, only ~4,000. The model spent more capacity on the (less test-like) clip data and less on the soundscape data that matches the test distribution. Net: less exposure to target distribution.
- SpecAugment alone is probably fine — its effect was confounded by REPLICAS.

**Lessons compounded with Phase 1's**:
- **The 20-file ss_val_loader is not a faithful LB proxy.** It measures generalization across files in `train_soundscapes/`, not generalization from train to test (which are different recording sessions). Treat local val as a sanity check, not an optimization target.
- **Don't bundle changes.** Test one variable at a time when LB is the only ground truth.
- **More exposure to target-distribution data > less, even at the cost of some memorization.** Until we have a way to add target-distribution data without trade-offs (Phase 2b's background mixing, Phase 5's pseudo-labeling), keep `SOUNDSCAPE_REPLICAS = 10`.

---

## Phase 2b — Background mixing ❌

Goal: synthesize training examples that combine clean species signal (clips) with realistic test ambience (soundscapes). Implements the "clips for species, soundscapes for ambience" idea.

**Result**: regressed. LB **0.696** (-0.076 from Phase 1, -0.018 from raw Phase 0). Only trained 1 epoch (val diverged after epoch 1, so we shipped the peak checkpoint).

**What went wrong**:
- **Label noise from the "background" pool.** Pre-cached 256 random 5s segments from unlabeled soundscape files. Critical mistake: *unlabeled doesn't mean silent*. Those segments very likely contained unlabeled bird calls. Mixing them into a clip labeled species X teaches: "audio with X + Y calls → species X (and definitely not Y)." The model learns *anti-correlations* between species → catastrophic for multi-label.
- **Too-aggressive α range** (0.3-0.8). At α=0.8 the background is 80% as loud as the call. Calls partially drowned.
- **Under-trained**. Only 1 epoch (vs Phase 1 needing 3). The training environment also misbehaved (epoch 2 took 3 hours instead of 25 min — likely sleep/throttle), which compounded uncertainty.

**Lesson — the broader pattern**:
- **Phase 1 (data change) gained +0.058 LB. Both Phase 2 attempts (augmentation changes) lost LB.**
- For our setup, *more / better training data* > *clever augmentation of existing data*. The big lever we haven't pulled yet is **Perch v2** — a foundation model trained on millions of bird recordings. That's the right next move.

**To rescue background mixing later** (if we want to come back to it):
- Filter the background pool to *low-energy* segments (likely silence) before caching. Removes the label-noise issue.
- Drop α to ~0.1-0.3 (background is a quiet acoustic stain, not a competing signal).
- Make mixing probability smaller (~0.25) so the model still sees many clean clips.

---

## Phase 3 — Perch v2 foundation model ✅

**Result**: LB **0.836** (+0.064 from Phase 1, our prior best).

Approach: precompute Perch v2 embeddings (1536-d) for every clip + every labeled soundscape window once. Train a small MLP head (LayerNorm + Linear 1536→512 + ReLU + Linear 512→234) on top of frozen embeddings.

**Lessons from Phase 3**:
- Each epoch now takes **~1-3 seconds** (vs ~25 minutes in Phase 0-2). ~1,000× speedup. This unlocks fast iteration.
- val/LB gap *shrank* from 0.110 to 0.092 — Perch's foundation-model generalization helps across recording sessions. Theoretical prediction confirmed empirically.
- Best epoch was 12/30; training got noisier after that. Adding a cosine LR schedule would let us train longer cleanly.
- Same/diff species cosine similarity in raw Perch embeddings: 0.255 vs 0.112 — Perch encodes species patterns out of the box.

**Setup files**:
- `embeddings/clip_embeddings.npy` (218 MB) + `clip_index.csv`
- `embeddings/ss_window_embeddings.npy` (9 MB) + `ss_window_index.csv`
- `checkpoints/model_v5.pt` (head weights only, 3.4 MB)
- Kaggle inference: bundle Perch ONNX wheel + use `tuckerarrants/perch-v2-no-dft-onnx`

---

## Phase 3.5 — Head refinements 🟡 (in progress)

Goal: squeeze the most out of Perch embeddings before moving to harder phases. Each
experiment is <5 min wall-clock thanks to cached embeddings, so we run them one at a
time, controlled, and learn from each.

### Experiment results (Phase 3 baseline = `ss_val_auc 0.9276`)

| # | Change | best val | Δ vs base | LB | Verdict |
|---|---|---|---|---|---|
| 1 | Cosine LR schedule (5e-4 → 1e-6) | 0.9193 | −0.008 | not shipped | ❌ Peak came earlier *and* lower. Smaller late steps locked the model into a worse local optimum, not a better one. **Concluded**: Phase 3's late-epoch noise was the model running out of signal, not optimizer overshoot. Save-best logic already handled it. |
| 2 | Linear head only (`LayerNorm + Linear(1536→234)`) | 0.9200 | −0.008 | not shipped | ❌ as a regression but ✅ as a **diagnostic** — Perch features are essentially linearly separable. The MLP's small edge (~0.008) may be noise. **Concluded**: capacity isn't the limiter, Perch is doing 99% of the work. Skip Experiment 3 (larger head). |
| 3 | Larger head (`hidden_dim=1024`) | — | — | — | ⏸ skipped — Exp 2 told us capacity isn't the bottleneck |
| 4 | Embedding mixup (Beta(0.4, 0.4), p=0.5) | **0.9358** | **+0.008** | pending | ✅ Helped. Peak hit earlier (ep 8 vs 12) AND held longer (multi-epoch plateau above 0.93). Classic regularization signature. |
| 5 | 5-seed ensemble of Exp 4 heads | — | — | pending | ⏳ Next. Reliable +0.01-0.02 from diversity. |

### Lessons compounded

- **Mixup works because it's a different *kind* of intervention** than what we tried before. SpecAugment / background mixing were aimed at making the *input* harder; mixup makes the *label* softer (interpolated). On rich pretrained features, label smoothing via mixup is a near-free regularizer.
- **Linear probe results matter more than you'd think.** Confirming Perch features are linearly separable changes the strategy: we stop investing in head architecture and double down on Perch-side improvements (Phase 4 SED features, Phase 5 pseudo-labeling).
- **Phase 3.5 ensemble shipped at LB 0.833 (−0.003 vs Phase 3's 0.836).** Local val went +0.005, LB went −0.003. Both moves are within the val noise floor (~±0.005 per-seed std on our 19-file val). **The biggest lesson: don't optimize past your val noise floor.** Future phases need to target >0.02 val improvements to be detectable above noise. Phases that gave real LB gains (Phase 1 +0.058, Phase 3 +0.064) all involved *data* changes; head tweaks compound less. **Strategy going forward: prioritize data-side interventions (Phase 5 pseudo-labels, Phase 4 SED features).**

---



### 1a. Soundscape-based validation
- Switched val set from held-out clips to held-out `train_soundscapes_labels.csv` windows.
- Split by **file** (not row) to prevent leakage: 70% files train, 30% files val.
- Validation now correlates within ~0.03 of LB; trustworthy for offline experiments.

### 1b. Multi-hot training on clips (`secondary_labels`)
- 4,372 clips (~12%) have secondary species listed.
- `BirdDatasetV2` parses them and sets multiple positions to 1 in the label vector.

### 1c. Train on labeled soundscape windows
- `SoundscapeTrainDataset` reads exact 5s windows from `train_soundscapes/`, applies mild random-gain augmentation, returns multi-hot labels.
- Mixed via `ConcatDataset([clip_train, soundscape_train] * SOUNDSCAPE_REPLICAS=10)`.
- Soundscape windows now ~22% of each epoch's training data.
- Checkpoint selection by **soundscape val_auc**, not clip val_auc.

**LB target**: 0.78-0.85.
**Current**: epoch 3 soundscape val_auc = 0.8824 → estimated LB ~0.85 (well above target).
**Files**: `notebooks/01_train.ipynb` (cells 14-24, sections 14 and 15).

---

## Phase 2 — Augmentation & cleaning ⏳

Goal: synthesize training examples that look more like test conditions; remove dead/silent training data.

### 2a. Background mixing
**The big one.** During training, for each clip, overlay a random "background-only" segment from a soundscape:

```
training_input = clip_5s + α * background_5s    # α ~ Uniform(0.3, 0.8)
training_label = clip's species (unchanged)
```

The model sees clean species calls embedded in realistic ambience — bridges the train/test gap without needing more labels.

Implementation: pre-extract background-only segments from soundscapes (windows where the labels CSV says "no labeled species" — or just any window from the unlabeled portion of `train_soundscapes/`). Sample one per training step, mix in.

Expected: +0.02 to +0.04.

### 2b. Energy-based silence trimming
For each clip, compute per-second RMS energy. When choosing a random 5s crop, weight selection toward higher-energy windows. Dramatically reduces "silence labeled as species X" noise in the training signal.

Implementation: precompute per-second energy for all 35k clips once (~30 min), cache. Sample crops with probability proportional to energy.

Expected: +0.005 to +0.015.

### 2c. SpecAugment
Standard audio augmentation: randomly mask out a few frequency bands and a few time bands in the mel spectrogram during training. Forces the model to use multiple cues instead of relying on any single spectral feature.

```python
self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
```

Expected: +0.01 to +0.02.

### 2d. Mixup
With small probability, take two training examples and blend them:

```
mixed_mel    = λ * mel_a    + (1-λ) * mel_b
mixed_label  = λ * label_a  + (1-λ) * label_b      where λ ~ Beta(0.2, 0.2)
```

Trains the model to handle clips with overlapping species, exactly what soundscape windows look like.

Expected: +0.01 to +0.03.

### 2e. Test-time augmentation (TTA)
At inference, run each 5s window through the model with small perturbations (e.g., shifted crops, gain variations), average the predictions. Free score boost at inference time only.

Expected: +0.005 to +0.015. Costs: ~2-4x slower inference.

**LB target**: 0.83-0.88.
**Files**: new section 16 in `notebooks/01_train.ipynb`.

---

## Phase 3 — Perch v2 foundation model ⏳

Goal: stop training feature extraction from scratch. Use Google's Perch v2 (a transformer pretrained on millions of bird recordings) as a fixed feature extractor.

### 3a. Embedding pipeline
- Run Perch v2 over every clip (random 5s crop) and every soundscape window once.
- Store 1,536-d embeddings on disk as `.npy` files. Total disk: maybe 1-3 GB.
- This is the expensive part — ~hours of one-time compute.

### 3b. Linear head training
- New model: `Linear(1536 → 234)` on top of frozen Perch embeddings.
- Trains in **minutes** on CPU (no audio decoding, no big network forward pass).
- Lets us iterate on the classifier head, augmentation, etc. fast.

### 3c. Embedding-aware augmentation
- Mixup on embeddings (much faster than mel-mixup).
- Background mixing in audio space → recompute embeddings on the fly (slow) OR pre-compute many backgrounds and add their embeddings (fast, approximation).

**LB target**: 0.88-0.92.
**Files**: new `notebooks/03_perch_embed.ipynb` (compute embeddings) + extension to `01_train.ipynb`.
**Reference**: Perch ONNX is in the EoS notebook's dataset list — `tuckerarrants/perch-v2-no-dft-onnx`. macOS arm64 needs `pip install onnxruntime` (the bundled Linux wheel won't run locally).

---

## Phase 4 — Sound Event Detection (SED) head 🟡 shipped, awaiting LB

Goal: use Perch's per-timestep features (its `spatial_embedding` output) instead of the pooled `embedding` output. Add an attention head that learns which timesteps matter for each species.

**Implementation**: Re-extract Perch outputs to get `spatial_embedding (B, 16, 4, 1536)` for clips + labeled soundscapes. Mean-pool over the freq dimension → `(B, 16, 1536)`. SED head: LayerNorm + bottleneck Linear → two 1×1 Conv1d's (att + cla) → `clip_logits = sum(softmax(tanh(att)) * cla, dim=time)`.

**Result** (5-seed ensemble):
- Per-seed bests: 0.9410 / 0.9416 / 0.9347 / 0.9234 / 0.9234 (std 0.0080 — wider than Phase 3.5's 0.0045)
- Mean individual:  0.9328
- Ensemble val_auc: **0.9401**  Δ vs Phase 3.5 ensemble: **+0.0072**
- This is 1.6× our noise std — the first non-noise architectural win since Phase 3.

LB pending. Expected: 0.84-0.86 range (central estimate ~0.848 if val/LB gap holds at Phase 3's 0.092).

### Architecture (1st-place 2025-inspired, used by the EoS notebook)
```
mel spectrogram (1, 128, 313)
    ↓ CNN backbone (timm)
feature map (C, F, T)
    ↓ GeMFreq pooling over freq axis (learnable p, sharper than mean)
(C, T)
    ↓ Linear + Dropout
(512, T)
    ↓ Conv1d(att) + Conv1d(cla)
attention weights (234, T) + frame logits (234, T)
    ↓ clip_logit = sum(softmax(tanh(att)) * frame_logits, dim=-1)
clip logits (234,)
```

### What this buys us
- The model can focus on the actual call moments and ignore silence — without us hand-engineering frequency filters.
- Frame-level outputs can be max-pooled or smoothed across windows at inference for stronger predictions.
- Combined with Perch in Phase 5 below, this is the basis of the top notebook's Distilled-SED model (LB 0.917 standalone).

### Knowledge distillation from Perch (combine with Phase 3)
- Add a second head: `GAP + Linear → 1536-d`.
- During training, the *backbone* is supervised by MSE against Perch's embedding for the same clip.
- The *SED head* uses stop-gradient on backbone features.
- Result: a fast mel-CNN that approximates Perch's understanding, doesn't need Perch at inference time.

**LB target**: 0.90-0.93 (standalone), more in an ensemble.
**Files**: new `notebooks/04_sed_train.ipynb`.

---

## Phase 5 — Iterative refinement (pseudo-labeling) ❌ (1st attempt)

Goal: use the Phase 3.5 ensemble to pseudo-label the ~10,500 unlabeled soundscape files, then train on the much larger dataset.

**1st attempt result**: Phase 5 ensemble ss_val_auc = **0.9243** vs Phase 3.5's 0.9329. Δ = **−0.0086** — clearly worse than noise floor. Not submitted.

### Pseudo-label diagnostic (threshold 0.75)

- 127,104 unlabeled SS windows embedded with Perch ✓
- 42.5% of windows got ≥1 positive ✓ (sweet spot)
- **Species coverage was severely skewed** ❌:
  - 122/206 species got 0 pseudo-labels
  - Top 3 species (65380, 22973, 555146) accounted for ~50% of all positives
  - Top 10 species accounted for ~90% of all positives

### What went wrong

With pseudo-labels at 73% of training data and 90% of positives concentrated in 10 species, the model spent its capacity on those few species and *lost ground* on the rare ones. Mean per-seed val dropped from Phase 3.5's 0.9278 → Phase 5's 0.9169.

The ensemble lift worked normally (+0.0074 over mean individual), but it couldn't compensate for a fundamentally biased training mix.

### Lessons + recovery options

**Lesson**: pseudo-labeling at a fixed threshold (0.75) reflects the *prior model's* biases. The Phase 3.5 ensemble is already confident on common species and uncertain on rare ones, so naive pseudo-labels reinforce that. We need to *correct* for the bias, not just propagate it.

Recovery options to try (lowest effort first):
1. **Lower threshold (0.5 or 0.6)** to capture more rare-species pseudo-labels. Risk: more noise.
2. **Per-species threshold calibration**: pick a per-species threshold that gives each species at least K positive labels in the pool. Restores coverage.
3. **Downsample pseudo-labels**: take only ~30k pseudo windows per epoch instead of 127k. Restores Phase 3.5-style mix where pseudo is supplement, not bulk.
4. **Soft pseudo-labels**: use raw probabilities (continuous values) instead of hard thresholding. Preserves uncertainty.
5. **Move to Phase 4 (SED with Perch spatial_embedding)** as a different lever entirely — sometimes a wrong approach is the wrong approach.

Best next attempt is probably **(3) downsample pseudo + retry**, since it preserves the working Phase 3.5 mix while adding some target-distribution signal. ~30 min to retest.

### 2nd attempt: downsampled to 20k pseudo windows

**Result**: ensemble val = **0.9333** vs Phase 3.5's 0.9329 — Δ = **+0.0004**, statistically zero.

The downsample fixed the regression (mean per-seed went from 0.9169 → 0.9284, back to Phase 3.5 levels) but didn't extract any *new* signal. Not submitted.

**Updated lesson**: pseudo-labels at threshold 0.75 are information-redundant with what Phase 3.5 already knows. The model's confident-prediction set ≈ the pseudo-label set ≈ the species it was already going to classify well. To extract new info from unlabeled soundscapes, we'd need either:
- Lower threshold (gets more diverse but noisier labels)
- Per-species calibration (forces broader coverage)
- A different model class doing the labeling (Phase 4 SED would have spatial-temporal info that Phase 3.5's clip-level head doesn't capture)
- Or accept that pseudo-labeling isn't the right lever for this dataset and move to Phase 4 (SED) or Phase 8 (SSL).

### 3rd attempt: lower threshold (0.5) + downsample (20k)

Hypothesis: lower threshold → broader species coverage → more diverse pseudo-label signal.

**Result**: ensemble val = **0.9343** vs Phase 3.5's 0.9329 — Δ = **+0.0014**, within noise. Not submitted.

Coverage did improve: species with 0 pseudo-labels dropped from 122 → 98 (+24 species). But the top-10 dominance pattern was unchanged (still 91% of all positives concentrated there). The 24 newly-covered species each got <10 pseudo-labels — not enough new signal to move the needle.

### Phase 5 conclusion (closed)

Three attempts, all flat or worse. The pattern:

| Attempt | Pseudo size | Threshold | Ensemble val | Δ vs Phase 3.5 |
|---|---|---|---|---|
| 1 | 127k (full) | 0.75 | 0.9243 | −0.0086 ❌ |
| 2 | 20k (downsample) | 0.75 | 0.9333 | +0.0004 ⚠ |
| 3 | 20k (downsample) | 0.50 | 0.9343 | +0.0014 ⚠ |

**Pseudo-labeling from Phase 3.5 doesn't extract new information.** This is consistent with the broader pattern in our results — phases that worked (1, 3) brought new audio distributions or new model classes; phases that didn't (2a, 2b, 3.5, 5) rearranged or augmented existing information. For the next phase we need a *new* information source. The best candidate is Perch's `spatial_embedding` output, which we haven't used yet (Phase 4).

### 5a. Within-clip refinement
1. Use the current best model.
2. For each `train_audio/` clip: slide a 5s window across the whole clip, record the probability for the clip's labeled species at each position.
3. Keep only windows where probability > threshold (e.g., 0.7). Drop the rest.
4. Retrain on this filtered dataset.

Effect: the model trains on fewer, *cleaner* examples instead of more, *noisier* examples. Counterintuitive but works because the training signal per example is so much stronger.

Expected: +0.01 to +0.03.

### 5b. Pseudo-label the unlabeled soundscapes
- We have ~9,000 unlabeled soundscape files (only 66 are labeled per-window).
- Run our best model on every 5s window of every file. Save predictions.
- For windows where the model is very confident (e.g., max prob > 0.85), treat that prediction as a pseudo-label.
- Add these pseudo-labeled windows to training.

Expected: +0.02 to +0.05. Risk: the model amplifies its own biases. Always validate on the real held-out soundscape val set.

### 5c. Iterate
Repeat 5a + 5b a few times. Diminishing returns after 2-3 rounds.

**LB target**: 0.91-0.94.
**Files**: new `notebooks/05_pseudo_label.ipynb`.

---

## Phase 6 — K-fold ensemble ⏳

Goal: train 5 models on different splits, average their predictions. Reliable bread-and-butter gain.

### 6a. 5-fold CV
- Split clips + soundscape files into 5 folds.
- Train one model per fold (5 models total). Each fold's val set is different.
- At inference, average the 5 models' predictions in logit space.

### 6b. Architecture / seed diversity
- Within the 5 folds, vary backbone (efficientnet_b0, b1, resnet18).
- Vary random seed.
- More diverse models → less correlated errors → better ensemble.

### 6c. Two-stage smoothing at inference
- Smooth across the 12 windows per file (already doing this in Phase 0).
- Optionally also smooth across folds (rank-average instead of logit-average).

**LB target**: 0.93-0.95.
**Cost**: 5x training time.
**Files**: new `notebooks/06_kfold.ipynb` (or convert `01_train.ipynb` to take a `--fold` argument).

---

## Phase 7 — Calibration & post-processing ⏳

Goal: squeeze the last drops without retraining.

### 7a. Isotonic regression per species
- Use the val set to fit isotonic regression models that map model probabilities → calibrated probabilities.
- Each species gets its own calibrator.
- Apply at inference.

### 7b. File-level confidence thresholding
- If a species has probability > 0.3 in ANY of the 12 windows of a file but < 0.1 in the others, push the low ones down further (a confident call should dominate; isolated weak ones are likely noise).

### 7c. Logit smoothing tuning
- We already do Gaussian smoothing with kernel `[0.1, 0.2, 0.4, 0.2, 0.1]`. Try wider/narrower kernels, optimize on val.

**LB target**: +0.005 to +0.02 over Phase 6.
**Files**: extension to `02_infer.ipynb` and the Kaggle kernel.

---

## Phase 8 — Novel edge / lottery tickets ⏳

Goal: find something nobody else has tried. This is where competitions are *won* (not just placed in).

Ideas to try if there's time / compute:

### 8a. SSL pretraining on unlabeled soundscapes
- ~10,000 unlabeled soundscape files = huge unlabeled corpus.
- Train a model with masked spectrogram modeling (like BERT for audio): randomly mask patches, predict them.
- Fine-tune the SSL-pretrained model on the labeled task.
- Could rival Perch for this specific domain.

### 8b. Two-stage cascade
- First stage: binary classifier "is there any bird in this 5s window?" — trained on all clips (positive) + soundscape silence (negative).
- Second stage: only run species classifier where stage 1 says "yes."
- Decouples the "find the call" and "identify the species" problems.

### 8c. Domain-specific synthetic data
- Mix multiple species' calls into one synthetic soundscape window at controlled SNRs.
- Label is the union of source species.
- Generates limitless training examples that look exactly like noisy multi-label test windows.

### 8d. Test-time meta-learning
- For each test file, use the first few seconds to estimate the recording's background noise floor and adapt the threshold dynamically.
- Risk: this borders on "leakage of test characteristics into predictions"; check competition rules first.

### 8e. Anything you come up with
This list is open-ended. The roadmap is a starting point, not a script.

---

## Tracking progress

Update the status table at the top after each phase ships. Format suggestion:

```
| 2 | Augmentation & cleaning | 0.83-0.88 | ✅ shipped, LB 0.857 (+0.143 from Phase 1) |
```

Also useful per phase:
- **Local soundscape val_auc** (offline signal)
- **LB score** (when submitted)
- **What worked, what didn't, what to try next** (the lessons)

A short retrospective after each phase (5 lines in NOTES.md) compounds learning across the competition.

---

## Compute budget

A practical constraint. Each full training run is ~3-5 hours on Apple Silicon MPS. Per phase the typical wall-clock budget:

| Phase | Training runs | Cumulative time |
|---|---|---|
| 2 | 2-3 | ~10 hours |
| 3 | 2-3 (head is fast; Perch embedding extraction is the slow one-time cost) | ~6 hours |
| 4 | 2-3 | ~12 hours |
| 5 | 2-3 (each refinement round = retraining) | ~12 hours |
| 6 | 5+ (k-fold + ensemble variants) | ~25 hours |
| 7 | 0 (no retraining) | ~30 min |
| 8 | varies | varies |

Total to silver-medal target: **~65-80 hours of training**. Spread across 1-2 weeks of overnight runs and weekend bursts.

If you want faster turnaround: rent a single A100 on Lambda Labs ($1-2/hour); training drops to ~30 minutes per run. ~$100 total to compress 80 hours into ~10 hours. Worth considering if pushing for the top.

---

## Stretch goals beyond the roadmap

If you're chasing 0.96+:
- Read every public BirdCLEF 2025 (and earlier) write-up. Steal everything.
- Form a team with someone strong on the model side.
- Reproduce the EoS notebook locally first; understand every piece before trying to beat it.
- Find one thing that's truly your own contribution. That's the differentiator.

The score gap from 0.93 (silver) to 0.96 (gold/winning) is bigger than the gap from 0.71 (current) to 0.93. The last 3 score points cost more than the first 22. Plan accordingly.
