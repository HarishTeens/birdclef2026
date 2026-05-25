# BirdCLEF+ 2026

A learning-by-doing run through my first Kaggle competition.

**Final result**: LB **0.842** (≈ top 70%). Trajectory: `0.497 → 0.714 → 0.772 → 0.836 → 0.839 → 0.842` across 7 submitted phases.

The pipeline that produced 0.842:

```
audio (5s)
   ↓ Perch v2 ONNX (foundation model, frozen)
spatial_embedding (16, 4, 1536)
   ↓ mean-pool over freq
(16, 1536)
   ↓ 10 SED heads (5 random seeds + 5 K-fold splits)
     each: LayerNorm → MLP bottleneck → Conv1d attention pooling over 16 timesteps
clip logits (234) per head
   ↓ average across 10 heads (in logit space)
   ↓ Gaussian smooth across 12 windows per soundscape
   ↓ sigmoid
probabilities → submission.csv
```

---

## File map

| Path | Purpose |
|---|---|
| `notebooks/01_train.ipynb` | Phase 4 + Phase 6 K-fold training. Produces 10 SED head checkpoints. |
| `notebooks/02_infer.ipynb` | Combined 10-model ensemble inference, writes `submission.csv`. |
| `notebooks/03_perch_embed.ipynb` | Perch v2 embedding extraction (both global + spatial). Run first. |
| `notebooks/birdclef-2026-eos-2/` | Top-of-leaderboard reference notebook (pulled via Kaggle CLI, for study). |
| `kaggle-uploads/baseline-ckpt/` | Trained model checkpoints uploaded as a Kaggle Dataset. |
| `kaggle-uploads/baseline-submit/` | Kaggle Notebook (the actual submission code, with `/kaggle/input/` paths). |
| `data/birdclef-2026/` | Unzipped competition data (35.5k clips + 10.6k soundscapes). |
| `data/perch/` | Perch v2 ONNX model + bundled `onnxruntime` wheel for Kaggle. |
| `embeddings/` | Cached Perch outputs — `clip_embeddings.npy`, `spatial_emb_clips.npy`, etc. |
| `checkpoints/` | Local trained models (`model_v8_seed*.pt`, `model_v9_fold*.pt`). |
| `ROADMAP.md` | Detailed phase-by-phase plan and per-phase retrospectives. |
| `NOTES.md` | Personal notes / scratchpad. |
| `requirements.txt` | Python 3.12 dependencies. |
| `.venv/` | Virtual environment. |

---

## Setup

```bash
brew install python@3.12                  # macOS only
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate the Kaggle CLI (`~/.kaggle/access_token` or `~/.kaggle/kaggle.json`).

Then download the data and Perch model:

```bash
kaggle competitions download -c birdclef-2026 -p data/
unzip data/birdclef-2026.zip -d data/birdclef-2026/

kaggle datasets download tuckerarrants/perch-v2-no-dft-onnx -p data/perch --unzip
```

## Running the pipeline

```bash
jupyter lab
```

Open the notebooks **in order**:

1. `03_perch_embed.ipynb` — extract embeddings once (~30-50 min, resumable).
2. `01_train.ipynb` — train all 10 SED heads (~20-30 min on Apple Silicon MPS).
3. `02_infer.ipynb` — local inference + sanity-check submission.csv.

To submit to Kaggle:

```bash
# Upload checkpoints
kaggle datasets version -p kaggle-uploads/baseline-ckpt -m "..."

# Push the Kaggle Notebook
kaggle kernels push -p kaggle-uploads/baseline-submit

# Submit (after the kernel preview-runs cleanly)
kaggle competitions submit -c birdclef-2026 \
  -k harishteens/birdclef-2026-baseline-submit -v <N> \
  -f submission.csv -m "Submission description"
```

---

## Phase summary

| # | Phase | LB | Δ | Verdict |
|---|---|---|---|---|
| 0 | Baseline (mel-CNN on isolated clips) | 0.714 | — | ✅ working pipeline |
| 1 | + Train on labeled soundscapes + multi-hot labels | 0.772 | +0.058 | ✅ big win |
| 2a | + SpecAugment + REPLICAS=4 | 0.757 | −0.015 | ❌ regressed |
| 2b | + Background mixing (clip + soundscape ambience) | 0.696 | −0.076 | ❌ regressed (label noise from unlabeled "backgrounds" that contained other birds) |
| 3 | Replace CNN with Perch v2 + small MLP head | 0.836 | +0.064 | ✅ huge win (foundation model leverage) |
| 3.5 | Mixup + 5-seed head ensemble | 0.833 | −0.003 | ⚠ within noise floor |
| 4 | SED head on Perch spatial_embedding, 5 seeds | 0.839 | +0.003 | ✅ small but real architectural win |
| 5 | Pseudo-label unlabeled soundscapes (3 attempts) | not shipped | flat to −0.05 | ❌ pseudo-labels redundant with model's existing confidence |
| 6 | K-fold ensemble (5 splits) of SED heads | 0.839 | 0 | ✅ tied with seed ensemble |
| 6b | Combined 10-model ensemble (5 seeds + 5 folds) | **0.842** | +0.003 | ✅ **current best** |

Failed variants explored (all within ±0.005 of baseline at the time): cosine LR schedule, linear-only head, larger hidden_dim, GeMFreq pooling. Most of these confirmed Perch features were already near-optimally extracted by simpler choices.

See `ROADMAP.md` for the detailed per-phase retrospectives.

---

## Learnings

These are what I'd want to remember for the **next** competition. Each was earned at the cost of either a wasted submission, a wasted training run, or both.

### 1. Data > model. Always.

The two changes that gave double-digit LB jumps (Phase 1: +0.058, Phase 3: +0.064) brought **new information** to the model:
- Phase 1: training data that looks like the test distribution (soundscapes vs isolated clips).
- Phase 3: a foundation model pretrained on millions of bird recordings.

Everything that just rearranged existing data (pseudo-labels, augmentation, head architecture tweaks) gave at most +0.005 LB, often nothing, sometimes regressions.

**For the next comp**: first prioritize anything that brings genuinely new information into training. Architectural cleverness is the polish at the end.

### 2. The val noise floor is real

Our held-out soundscape val set was only 19 files / 422 windows. The per-seed standard deviation of "best val_auc" was ~0.005. So any improvement of less than ~0.01 in local val was **statistically indistinguishable from noise**.

This burned us in Phase 3.5: we saw a +0.005 val gain from mixup ensembling, shipped it, and LB came back at −0.003. The "improvement" was always within noise — we just couldn't tell from one measurement.

**For the next comp**:
- Quantify your val noise floor (train the same config with 5 seeds; the std is your floor).
- Don't burn submissions on improvements smaller than 2× the noise floor.
- Trust local val only as a **change detector**, not an optimization target.

### 3. Submission discipline

Code competitions like BirdCLEF cap you at 5 submissions/day (resets at midnight UTC). Each one matters more than you think.

- **Never** use blind retry loops. We wasted 3 submissions because a retry script checked for "Successfully" in the API response but successful submissions returned empty output. The script kept re-submitting the same model every 30 minutes.
- Always parse the actual submission list (`kaggle competitions submissions -c ...`) to confirm a submission landed before deciding what to do next.
- Plan your day's submission budget before iterating. Hold one in reserve for the "what-if".

### 4. Cache features for 1000× iteration speed

Phase 0–2 trained a CNN on raw audio → mel → CNN backbone. Each epoch took ~25 minutes. We managed ~3 experiments per day.

Phase 3 onward used pre-computed Perch embeddings. Each epoch took **~1-3 seconds**. We ran 30+ experiments per day.

That 1000× speedup is what made it possible to test linear-probe ablations, multi-seed ensembles, and K-fold across just one day.

**For the next comp**: the moment you commit to a frozen feature extractor (foundation model, fixed backbone), cache its outputs once and never run it again during training. Iteration speed compounds into score.

### 5. Run linear probes early

In Phase 3.5 we replaced our MLP head with a single `Linear(1536 → 234)` layer. It scored 0.9200 vs the MLP's 0.9276 — within the noise floor of each other.

That single experiment told us: **Perch's features are essentially linearly separable for our task.** The MLP's small edge was probably noise. This immediately stopped us from investing in head architecture variants (`hidden_dim=1024`, etc.) and pushed us to spend effort on the *backbone side* (Phase 4 SED, multi-backbone ensembles).

**For the next comp**: when using a pretrained foundation model, train a *linear* head first to measure how good the features are. The gap between linear-probe and your best head tells you whether to invest in head architecture or in something else entirely.

### 6. Distribution alignment is the single biggest lever

In Phase 0, our val_auc was 0.97 but LB was 0.71 — a 0.26 gap. The training data (isolated clips) was so different from the test data (soundscape windows) that almost nothing transferred.

In Phase 1, we just **added some labeled soundscape data to training**. Val dropped to 0.88 (correctly — the harder val is closer to real test). But LB jumped to 0.77 (+0.06). The same code, just a different data mix.

**For the next comp**: the second you have a working baseline, the very next move is to look at the distribution gap between your train and test sets. If they look meaningfully different (different recording conditions, different sampling, different label format), that gap is your biggest lever.

### 7. Save-best logic is non-negotiable

Always save the checkpoint at the epoch with the highest val score, not the last epoch. Models routinely peak around epoch 8-12 and then drift down. With save-best, late-epoch noise doesn't hurt you. Without it, you ship a worse model than you trained.

```python
if val_auc > best_val_auc:
    best_val_auc = val_auc
    torch.save(model.state_dict(), checkpoint_path)
```

3 lines. Always.

### 8. Ensembling helps, but compounds less than data changes

Our 5-seed ensemble (Phase 4) was +0.003 LB. 5-fold ensemble (Phase 6) was +0.000 LB (tied with 5-seed — different diversity sources, same gain). Combined 10-model (5+5) was +0.003 LB over either.

Reliable, modest, additive. But also bounded — you can't ensemble your way to a +0.05 jump.

**For the next comp**: ensembling is the closing move, not the opening move. Get your single-model right first.

### 9. The "I don't know" intervention

Several times during BirdCLEF, I shipped something because val was up by 0.005 and assumed it would translate to LB. It didn't. The Phase 5 pseudo-label saga was 3 separate experiments all hovering at "+0.001 to +0.005 val, doesn't translate to LB".

The pattern was: I was over-trusting a noisy signal because I wanted there to be progress.

**For the next comp**: if val improvement is small, ask "could this be noise?" *before* shipping. Run with multiple seeds. Check the std. If std > Δ, don't ship.

---

## Where this could go next

If you ever come back to this comp (or apply the lessons to another):

- **Multi-backbone ensemble**: combine Perch with BirdNET or another foundation model. The two model classes have different biases → strongest ensemble play available.
- **EoS-style distillation**: train a small mel-CNN to mimic Perch's embeddings, then add an SED head on top. This is what the top notebooks do.
- **SSL pretraining**: train a masked-spectrogram autoencoder on the ~10,000 unlabeled soundscapes, fine-tune on labels. High-effort, potentially big win.
- **TTA at inference**: average across small input perturbations. Cheap free gain.

But realistically, the medal cutoff (top 10%, LB ~0.949) is +0.107 from our current 0.842. That's larger than any single move we made. Getting there would require either a novel idea or several weeks of careful execution stacking multiple medium-confidence moves.

---

## Acknowledgments

Built jointly with Claude Code as a learning exercise — the goal was to *understand* every layer of a Kaggle competition rather than crib from public notebooks. The collaboration produced detailed line-by-line walkthroughs of every concept (BCE loss, Adam, mel spectrograms, attention pooling, K-fold, mixup, pseudo-labeling, foundation models) — those are in the notebooks' markdown cells and in `ROADMAP.md`.
