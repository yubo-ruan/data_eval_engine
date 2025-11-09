# Reward Model Implementation TODO

Detailed task breakdown for implementing the complete reward model pipeline.

---

## 📋 Task Categories

- 🟢 **Ready to implement** - No blockers
- 🟡 **In progress** - Currently being worked on
- 🔴 **Blocked** - Waiting on dependencies
- ✅ **Complete** - Implemented and tested

---

## Phase 1: Data & Clip Preparation

### 1.1 Local Video Download & Extraction 🟢

**Goal**: Allow users to download videos and extract clips locally.

**Tasks**:
- [ ] `scripts/01_download_videos.py`
  - [ ] Download from EPIC-Kitchens dataset
  - [ ] Download from Ego4D (future)
  - [ ] Download from custom sources (future)
  - [ ] Progress tracking & resume capability
  - [ ] Checksum verification
  - [ ] Save to `data/raw_videos/<source>/`

- [ ] `scripts/02_extract_clips.py`
  - [ ] Load video from `data/raw_videos/`
  - [ ] Extract clips based on annotations
  - [ ] Extract random temporal windows
  - [ ] Resize to 480x480, resample to 15fps
  - [ ] Save to `data/extracted_clips/<source>/`
  - [ ] Support batch extraction (process multiple videos)
  - [ ] Generate clip metadata (clip_id, source, timestamp, action, etc.)

**Files to create**:
- `scripts/01_download_videos.py`
- `scripts/02_extract_clips.py`

**Dependencies**:
- `opencv-python`, `pandas`, `tqdm`

---

### 1.2 S3 Sync Integration 🟢

**Goal**: Seamless upload/download of extracted clips to/from S3.

**Tasks**:
- [ ] `scripts/03_sync_with_s3.py`
  - [ ] Wrapper around existing `upload_to_cloud.py` and `download_from_cloud.py`
  - [ ] Update paths to work with `reward_model/data/extracted_clips/`
  - [ ] Support multi-source sync (EPIC, Ego4D, GoPro)
  - [ ] Add `--upload` and `--download` flags
  - [ ] Progress tracking for large batches

**Files to create**:
- `scripts/03_sync_with_s3.py`

**Dependencies**:
- Existing S3 scripts in `scripts/upload_to_cloud.py` and `scripts/download_from_cloud.py`

---

### 1.3 Unified Metadata Preparation 🟢

**Goal**: Create standardized metadata file across all video sources.

**Tasks**:
- [ ] `scripts/04_prepare_metadata.py`
  - [ ] Load annotations from multiple sources
  - [ ] Map to unified schema (clip_id, source, video_id, start_frame, stop_frame, action, etc.)
  - [ ] Handle different annotation formats (EPIC CSV, Ego4D JSON, etc.)
  - [ ] Save to `data/metadata/unified_metadata.csv`
  - [ ] Include clip statistics (duration, fps, resolution)

**Files to create**:
- `scripts/04_prepare_metadata.py`
- `data/metadata/unified_metadata.csv` (output)

**Dependencies**:
- `pandas`, understanding of EPIC-100 annotation format

---

## Phase 2: Feature Extraction

### 2.1 Individual Feature Extractors 🟢

**Goal**: Modular feature extractors, one per feature type.

**Tasks**:
- [ ] `features/extractors/blur_extractor.py`
  - [ ] Laplacian variance method
  - [ ] Input: video path, Output: blur score (scalar)
  - [ ] Test on sample clips

- [ ] `features/extractors/exposure_extractor.py`
  - [ ] Histogram entropy method
  - [ ] Input: video path, Output: exposure score (scalar)

- [ ] `features/extractors/motion_extractor.py`
  - [ ] Optical flow variance (motion_var)
  - [ ] Jerk score (motion smoothness)
  - [ ] Input: video path, Output: (motion_var, jerk_score)

- [ ] `features/extractors/hand_extractor.py`
  - [ ] MediaPipe hand detection
  - [ ] Compute % frames with hands visible
  - [ ] Input: video path, Output: hand_visibility (0-1)

- [ ] `features/extractors/clip_embedding_extractor.py`
  - [ ] CLIP ViT-B/32 embeddings
  - [ ] Average over all frames
  - [ ] Input: video path, Output: 512-D vector

- [ ] `features/extractors/__init__.py`
  - [ ] Import all extractors
  - [ ] Provide unified interface: `extract_feature(video_path, feature_name)`

**Files to create**:
- 5 extractor files + `__init__.py`

**Dependencies**:
- `opencv-python`, `mediapipe`, `transformers`, `torch`, `PIL`

---

### 2.2 Main Feature Extraction Pipeline 🟢

**Goal**: Run all feature extractors on all clips in batch.

**Tasks**:
- [ ] `features/extract_all_features.py`
  - [ ] Load all clips from `data/extracted_clips/`
  - [ ] For each clip, run all extractors
  - [ ] Handle errors gracefully (skip corrupted clips)
  - [ ] Save features to `features/outputs/all_features.csv`
  - [ ] Save full CLIP embeddings to `features/outputs/clip_embeddings.npy`
  - [ ] Progress bar for batch processing
  - [ ] Support resume (skip already processed clips)

**Files to create**:
- `features/extract_all_features.py`

**Dependencies**:
- Feature extractors from 2.1

---

### 2.3 Composite Score Calculator 🟢

**Goal**: Compute weighted composite score from auto-features (for comparison with human preferences).

**Tasks**:
- [ ] `features/compute_composite_score.py`
  - [ ] Load features from `features/outputs/all_features.csv`
  - [ ] Normalize each feature (z-score or min-max)
  - [ ] Weighted average: `composite = w1*blur + w2*exposure + w3*motion + w4*hands`
  - [ ] Default weights: equal (0.25 each)
  - [ ] Support custom weights via config
  - [ ] Add composite score column to CSV

**Files to create**:
- `features/compute_composite_score.py`

**Dependencies**:
- `pandas`, `numpy`

---

## Phase 3: Human Preference Calibration

### 3.1 Annotation Integration 🟢

**Goal**: Interface with existing human annotation tool.

**Tasks**:
- [ ] `human_preferences/collect_annotations.py`
  - [ ] Launch human annotation tool from reward_model directory
  - [ ] Automatically use clips from `data/extracted_clips/`
  - [ ] Save annotations to `human_preferences/annotations.csv`

**Files to create**:
- `human_preferences/collect_annotations.py`

**Dependencies**:
- Existing `human_annotation_tool/app.py`

---

### 3.2 Annotation Processing 🟢

**Goal**: Convert raw annotations to training-ready format.

**Tasks**:
- [ ] `human_preferences/process_annotations.py`
  - [ ] Load annotations CSV (columns: clip_a_id, clip_b_id, preference)
  - [ ] Map clip IDs to feature vectors
  - [ ] Create pairwise training samples
  - [ ] Handle ties (skip or treat as 50/50)
  - [ ] Split into train/val (80/20)
  - [ ] Save to `human_preferences/train_pairs.csv` and `val_pairs.csv`

**Files to create**:
- `human_preferences/process_annotations.py`

**Dependencies**:
- `pandas`, feature outputs from Phase 2

---

### 3.3 PyTorch Pair Dataset 🟢

**Goal**: PyTorch dataset for loading pairwise preference data.

**Tasks**:
- [ ] `human_preferences/pair_dataset.py`
  - [ ] PyTorch Dataset class: `PairwiseDataset`
  - [ ] `__getitem__`: Returns (features_a, features_b, label)
    - features_a: [517-D] (5 auto + 512 CLIP)
    - features_b: [517-D]
    - label: 1 if A preferred, 0 if B preferred
  - [ ] DataLoader with batching
  - [ ] Handle CLIP embeddings (load from .npy file)

**Files to create**:
- `human_preferences/pair_dataset.py`

**Dependencies**:
- `torch`, `pandas`, `numpy`

---

## Phase 4: Reward Model Architecture

### 4.1 Model Components 🟢

**Goal**: Implement trunk + two-head architecture.

**Tasks**:
- [ ] `models/architectures/trunk.py`
  - [ ] MLP: input_dim (517) → 256 → 128
  - [ ] ReLU activations
  - [ ] Optional dropout (configurable)
  - [ ] Layer normalization

- [ ] `models/architectures/human_head.py`
  - [ ] MLP: 128 → 64 → 1
  - [ ] No activation on output (logit)
  - [ ] Used for pairwise preference prediction

- [ ] `models/architectures/robot_head.py`
  - [ ] MLP: 128 → 64 → 1 (stub)
  - [ ] Future: predict ΔM from robot data
  - [ ] For now: not trained, just placeholder

- [ ] `models/architectures/reward_model.py`
  - [ ] Full model: Trunk + HumanHead (+ RobotHead stub)
  - [ ] Forward pass: `forward(features) -> reward_score`
  - [ ] For pairs: `forward_pair(feat_a, feat_b) -> (R_a, R_b, logit_diff)`
  - [ ] Load/save checkpoints

**Files to create**:
- 4 architecture files

**Dependencies**:
- `torch`, `torch.nn`

---

### 4.2 Loss Function 🟢

**Goal**: Implement Bradley-Terry pairwise loss.

**Tasks**:
- [ ] `models/losses.py`
  - [ ] `BradleyTerryLoss(β=1.0)`
  - [ ] Formula: `loss = -log σ(β * (R_a - R_b))` if A preferred
  - [ ] Support batch processing
  - [ ] Option to weight ties (if included)

**Files to create**:
- `models/losses.py`

**Dependencies**:
- `torch`

---

### 4.3 Training Script 🟢

**Goal**: Main training loop with logging and checkpointing.

**Tasks**:
- [ ] `models/train.py`
  - [ ] Load train/val datasets
  - [ ] Initialize model, optimizer (Adam), loss function
  - [ ] Training loop:
    - [ ] Forward pass on pairs
    - [ ] Compute BT loss
    - [ ] Backward pass, optimizer step
    - [ ] Log: loss, pair-AUC, learning rate
  - [ ] Validation loop:
    - [ ] Compute val loss, pair-AUC
    - [ ] Early stopping based on val loss
  - [ ] Checkpointing: save best model to `models/checkpoints/best_model.pth`
  - [ ] TensorBoard logging to `models/logs/`
  - [ ] Generate score histograms (top/bottom clips)

**Files to create**:
- `models/train.py`

**Dependencies**:
- `torch`, `tensorboard`, model components from 4.1-4.2

---

### 4.4 Evaluation Script 🟢

**Goal**: Evaluate trained model on validation set.

**Tasks**:
- [ ] `models/evaluate.py`
  - [ ] Load trained model from checkpoint
  - [ ] Compute metrics:
    - [ ] Pair-AUC on validation set
    - [ ] Loss
    - [ ] Accuracy (% correct preferences)
  - [ ] Generate reward score distribution
  - [ ] Find top-10 and bottom-10 clips by reward
  - [ ] Save results to `models/logs/evaluation_results.json`

**Files to create**:
- `models/evaluate.py`

**Dependencies**:
- `torch`, `sklearn.metrics`, model from 4.1

---

## Phase 5: Validation – Simulate Downstream Gain

### 5.1 Toy Behavior Cloning Model 🟢

**Goal**: Simple CNN to predict actions from video frames (proxy for robot policy).

**Tasks**:
- [ ] `validation/toy_bc/bc_model.py`
  - [ ] Simple CNN architecture (ResNet18 or MobileNet backbone)
  - [ ] Input: video frames (480x480x3)
  - [ ] Output: action probabilities (or regression target)
  - [ ] For simplicity: predict verb/noun from EPIC annotations

- [ ] `validation/toy_bc/train_bc.py`
  - [ ] Train BC model on random 80% of clips
  - [ ] Fine-tune separately with:
    - Top 10% clips by R_h (high reward)
    - Bottom 10% clips by R_h (low reward)
  - [ ] Measure validation loss difference
  - [ ] ΔM = val_loss(top) - val_loss(bottom)
  - [ ] Expect: ΔM > 0 (high reward clips improve BC performance)

- [ ] `validation/toy_bc/evaluate_bc.py`
  - [ ] Evaluate BC model on validation set
  - [ ] Compute success rate or loss per clip
  - [ ] Save per-clip ΔM to `validation/results/bc_delta_m.csv`

**Files to create**:
- 3 files in `validation/toy_bc/`

**Dependencies**:
- `torch`, `torchvision`, `opencv-python`

---

### 5.2 Correlation Computation 🟢

**Goal**: Measure correlation ρ(R_h, ΔM_sim).

**Tasks**:
- [ ] `validation/compute_correlation.py`
  - [ ] Load reward scores R_h from model
  - [ ] Load ΔM from BC evaluation
  - [ ] Compute Pearson correlation: ρ(R_h, ΔM_sim)
  - [ ] Compute Spearman rank correlation (alternative)
  - [ ] Target: ρ ≥ 0.5
  - [ ] Save results to `validation/results/correlation.json`
  - [ ] Plot scatter: R_h vs ΔM

**Files to create**:
- `validation/compute_correlation.py`

**Dependencies**:
- `scipy.stats`, `matplotlib`

---

### 5.3 Downstream Simulation Pipeline 🟢

**Goal**: End-to-end validation pipeline.

**Tasks**:
- [ ] `validation/simulate_downstream.py`
  - [ ] Load trained reward model
  - [ ] Score all clips
  - [ ] Select top/bottom clips by reward
  - [ ] Train BC models on different subsets
  - [ ] Compute ΔM
  - [ ] Compute correlation
  - [ ] Generate report with plots

**Files to create**:
- `validation/simulate_downstream.py`

**Dependencies**:
- BC model (5.1), correlation script (5.2)

---

## Phase 6: Ablation Studies

### 6.1 Feature Ablation 🟢

**Goal**: Verify that motion/hand features matter.

**Tasks**:
- [ ] Modify `models/train.py` to support feature masking
- [ ] Train model with:
  - [ ] All features (baseline)
  - [ ] Remove motion features (no motion_var, jerk)
  - [ ] Remove hand features (no hand_visibility)
  - [ ] Remove both motion + hand
- [ ] Compare ρ(R_h, ΔM_sim) for each variant
- [ ] Expect: ≥ 0.1 drop in ρ when removing motion/hand
- [ ] Document results in `validation/results/ablation_results.json`

**Dependencies**:
- Trained baseline model, validation pipeline (5.3)

---

## Phase 7: Visualization & Dashboard

### 7.1 Training Visualization 🟢

**Goal**: Monitor training progress with TensorBoard.

**Tasks**:
- [ ] Log to TensorBoard:
  - [ ] Loss curves (train & val)
  - [ ] Pair-AUC over time
  - [ ] Reward score histograms
  - [ ] Learning rate schedule
- [ ] Launch with: `tensorboard --logdir models/logs/`

**Dependencies**:
- `tensorboard`, training script (4.3)

---

### 7.2 Qualitative Review Dashboard 🟢

**Goal**: Visualize top/bottom clips for qualitative review.

**Tasks**:
- [ ] `utils/visualization.py`
  - [ ] Function: `show_top_bottom_clips(model, n=10)`
  - [ ] Score all clips with trained model
  - [ ] Find top-10 and bottom-10 by reward
  - [ ] Generate HTML page with embedded videos
  - [ ] Side-by-side comparison
  - [ ] Include: reward score, auto-features, action label

**Files to create**:
- `utils/visualization.py`
- `validation/results/clip_review.html` (output)

**Dependencies**:
- `matplotlib`, `jinja2` (for HTML templates)

---

## Phase 8: Configuration & Utilities

### 8.1 Configuration Files 🟢

**Goal**: YAML configs for all pipeline components.

**Tasks**:
- [ ] `configs/data_config.yaml`
  - [ ] Paths: raw_videos, extracted_clips, metadata
  - [ ] S3 bucket name
  - [ ] Supported sources: epic_kitchens, ego4d, gopro

- [ ] `configs/feature_config.yaml`
  - [ ] Enabled features: blur, exposure, motion, hands, clip
  - [ ] Feature-specific settings (e.g., CLIP model variant)
  - [ ] Composite score weights

- [ ] `configs/model_config.yaml`
  - [ ] Trunk: hidden_dims, dropout, activation
  - [ ] Human head: hidden_dims
  - [ ] Robot head: hidden_dims (stub)

- [ ] `configs/training_config.yaml`
  - [ ] Batch size, learning rate, epochs
  - [ ] β for BT loss
  - [ ] Optimizer settings (Adam, weight decay)
  - [ ] Early stopping patience
  - [ ] Checkpoint frequency

**Files to create**:
- 4 YAML config files

**Dependencies**:
- `pyyaml`

---

### 8.2 Shared Utilities 🟢

**Goal**: Common helper functions.

**Tasks**:
- [ ] `utils/metrics.py`
  - [ ] Pair-AUC calculation
  - [ ] Accuracy for pairwise preferences
  - [ ] Correlation metrics (Pearson, Spearman)

- [ ] `utils/io_utils.py`
  - [ ] Load/save CSV, JSON, YAML
  - [ ] Load checkpoint safely
  - [ ] Video I/O helpers

**Files to create**:
- 2 utility files

**Dependencies**:
- `pandas`, `torch`, `sklearn`

---

## Phase 9: Testing & Documentation

### 9.1 Unit Tests 🟡

**Goal**: Test critical components.

**Tasks**:
- [ ] Test feature extractors on sample clip
- [ ] Test BT loss computation
- [ ] Test model forward pass
- [ ] Test dataset loading

**Dependencies**:
- `pytest`

---

### 9.2 Integration Tests 🟡

**Goal**: End-to-end pipeline test.

**Tasks**:
- [ ] Run full pipeline on 10 clips
- [ ] Verify output formats
- [ ] Check for errors

---

### 9.3 Documentation 🟡

**Goal**: Complete documentation for all scripts.

**Tasks**:
- [ ] Add docstrings to all functions
- [ ] Update README with examples
- [ ] Create usage guide for each script

---

## Phase 10: Deployment & Production

### 10.1 Requirements File 🟢

**Goal**: Pin all dependencies.

**Tasks**:
- [ ] `requirements.txt`
  - [ ] List all Python packages with versions
  - [ ] Include: torch, opencv-python, mediapipe, transformers, pandas, etc.

**Files to create**:
- `requirements.txt`

---

### 10.2 Docker Container 🔴

**Goal**: Containerize for reproducibility.

**Tasks**:
- [ ] Create Dockerfile
- [ ] Test container on fresh machine
- [ ] Document Docker usage

---

## Summary: Implementation Order

**Recommended order**:

1. ✅ **Phase 1.1-1.2**: Data preparation (download, extract, sync)
2. ✅ **Phase 2.1-2.3**: Feature extraction
3. ✅ **Phase 3.1-3.3**: Human preference integration
4. ✅ **Phase 4.1-4.4**: Model architecture & training
5. ✅ **Phase 5.1-5.3**: Validation pipeline
6. ✅ **Phase 6.1**: Ablation studies
7. ✅ **Phase 7.1-7.2**: Visualization
8. ✅ **Phase 8.1-8.2**: Configuration & utilities

**Total estimated tasks**: ~70-80 individual tasks

**Estimated timeline**: 2-3 weeks of focused work

---

**Next Step**: Start with Phase 1.1 (download videos script) and work sequentially through the phases.
