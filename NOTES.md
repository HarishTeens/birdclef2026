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


---------------------------------
 The Phased Plan                                                                                                                                                                                                           
                  
  Phase 0 — Foundations ✓ (you're here)
                                                                                                                                                                                                                            
  - Working pipeline: 0.714
  - Honest validation (soundscape val ≈ LB): 0.745 → 0.714                                                                                                                                                                  
  - Status: You can iterate.                                                                                                                                                                                                
                                                                                                                                                                                                                            
  Phase 1 — Distribution alignment (target: 0.78-0.83)                                                                                                                                                                      
                                                                                                                                                                                                                            
  Goal: train the model on data that looks like the test set.                                                                                                                                                               
  - Train on labeled soundscapes (1,478 windows) mixed with clips.
  - Use secondary_labels for multi-hot training on clips.                                                                                                                                                                   
  - Expected gain: +0.05 to +0.10. The biggest single jump you'll see without bringing in foundation models.
                                                                                                                                                                                                                            
  Phase 2 — Augmentation (target: 0.80-0.85)                                                                                                                                                                                
                                                                                                                                                                                                                            
  Goal: make the model robust to test-time noise/variation.                                                                                                                                                                 
  - Random gain, background noise mixing, time/frequency masks (SpecAugment).
  - Mixup / CutMix between clips.                                                                                                                                                                                           
  - Test-time augmentation (TTA) — score multiple shifted windows, average.
  - Expected gain: +0.02 to +0.04.                                                                                                                                                                                          
                                                                                                                                                                                                                            
  Phase 3 — Perch (target: 0.88-0.92)                                                                                                                                                                                       
                                                                                                                                                                                                                            
  Goal: stop training the audio-feature-extractor from scratch.                                                                                                                                                             
  - Use Perch v2 to compute embeddings of every clip and every soundscape window.
  - Train a small head on top: embedding → 234 logits.                                                                                                                                                                      
  - This is the single biggest model upgrade. The top notebook does this; you should too.
  - Expected gain: +0.05 to +0.10.                                                                                                                                                                                          
                                                                                                                                                                                                                            
  Phase 4 — Better head (target: 0.90-0.93)                                                                                                                                                                                 
                                                                                                                                                                                                                            
  Goal: get more out of the Perch features.
  - SED-style attention head (frame-level logits + attention pooling).                                                                                                                                                      
  - Use Perch embeddings as input and as a distillation target for a small CNN (the EoS approach).                                                                                                                          
  - Expected gain: +0.01 to +0.03.                                                                
                                                                                                                                                                                                                            
  Phase 5 — Ensemble (target: 0.93-0.95)
                                                                                                                                                                                                                            
  Goal: average diverse models.
  - K-fold cross-validation: train 5 models on different splits.                                                                                                                                                            
  - Mix architectures (one EfficientNet, one ResNet, one Perch-head).
  - Mix random seeds within each architecture.                       
  - Average predictions in logit space + Gaussian smooth across windows.                                                                                                                                                    
  - Expected gain: +0.02 to +0.04. This is reliable bread-and-butter.                                                                                                                                                       
                                                                                                                                                                                                                            
  Phase 6 — Calibration & post-processing (target: 0.94-0.95)                                                                                                                                                               
                                                                                                                                                                                                                            
  Goal: squeeze the last bits.                                                                                                                                                                                              
  - Isotonic regression to calibrate probabilities per species.                                                                                                                                                             
  - Smoothing in logit space (we already do this lightly).                                                                                                                                                                  
  - Threshold tuning at the 60s-file level (mark species as absent if confidence is low across all 12 windows).
  - Expected gain: +0.005 to +0.02.                                                                                                                                                                                         
                                                                                                                                                                                                                            
  Phase 7 — Novel edge (target: depends on idea, +0.00 to +0.05)
                                                                                                                                                                                                                            
  This is the wildcard. Some ideas worth trying:                                                                                                                                                                            
  - SSL pretraining on the 10,500+ unlabeled soundscape files with masked spectrogram modeling.                                                                                                                             
  - Domain-specific data augmentation — synthesize soundscapes by mixing clips at controlled SNRs with real backgrounds.                                                                                                    
  - Two-stage classification: first "is there a bird here at all?" then "which species?" — handles the silence-dominated test set explicitly.
  - Pseudo-labeling: run a strong model on the unlabeled soundscapes, take confident predictions as additional training labels, retrain.                                                                                    
                                                                                                                                                                                                                            
  If you find a novel idea that works here, you might genuinely beat the top. That's where Kaggle is won.        