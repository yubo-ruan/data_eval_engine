# Reward Model Training Pipeline

Production-ready reward model for evaluating egocentric video clips for robot manipulation training.

---

## Project Structure

```
reward_model/
├── data/                           # Data management
│   ├── raw_videos/                 # Downloaded full videos (gitignored)
│   ├── extracted_clips/            # Processed clips (gitignored)
│   │   ├── epic_kitchens/
│   │   ├── ego4d/
│   │   └── gopro/
│   ├── annotations/                # Source annotations from datasets
│   └── metadata/                   # Unified metadata files
│
├── features/                       # Feature extraction
│   ├── extractors/                 # Individual feature extractors
│   │   ├── blur_extractor.py
│   │   ├── exposure_extractor.py
│   │   ├── motion_extractor.py
│   │   ├── hand_extractor.py
│   │   ├── clip_embedding_extractor.py
│   │   └── __init__.py
│   ├── outputs/                    # Extracted features (CSV/JSON)
│   ├── extract_all_features.py    # Main feature extraction script
│   └── compute_composite_score.py # Composite score calculator
│
├── human_preferences/              # Human annotation integration
│   ├── collect_annotations.py     # Interface to human annotation tool
│   ├── process_annotations.py     # Convert annotations to training format
│   └── pair_dataset.py            # PyTorch dataset for pairwise data
│
├── models/                         # Model architecture & training
│   ├── architectures/
│   │   ├── trunk.py               # Shared MLP trunk
│   │   ├── human_head.py          # Human preference head
│   │   ├── robot_head.py          # Robot performance head (stub)
│   │   └── reward_model.py        # Full model definition
│   ├── checkpoints/               # Saved model weights (gitignored)
│   ├── logs/                      # Training logs (tensorboard)
│   ├── train.py                   # Main training script
│   ├── losses.py                  # Bradley-Terry loss implementation
│   └── evaluate.py                # Model evaluation script
│
├── validation/                     # Validation pipeline
│   ├── toy_bc/                    # Toy behavior cloning model
│   │   ├── bc_model.py            # Simple CNN for BC
│   │   ├── train_bc.py            # BC training script
│   │   └── evaluate_bc.py         # BC evaluation
│   ├── results/                   # Validation results
│   ├── simulate_downstream.py     # Simulate ΔM via BC
│   └── compute_correlation.py     # Compute ρ(R_h, ΔM_sim)
│
├── configs/                        # Configuration files
│   ├── data_config.yaml           # Data sources & paths
│   ├── feature_config.yaml        # Feature extraction settings
│   ├── model_config.yaml          # Model architecture & hyperparameters
│   └── training_config.yaml       # Training settings
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── exploratory_analysis.ipynb
│   ├── feature_visualization.ipynb
│   └── reward_distribution.ipynb
│
├── utils/                          # Shared utilities
│   ├── visualization.py           # Plotting & dashboards
│   ├── metrics.py                 # Evaluation metrics
│   └── io_utils.py                # File I/O helpers
│
├── scripts/                        # Data preparation scripts
│   ├── 01_download_videos.py      # Download videos to raw_videos/
│   ├── 02_extract_clips.py        # Extract clips from videos
│   ├── 03_sync_with_s3.py         # Upload/download clips to/from S3
│   └── 04_prepare_metadata.py     # Create unified metadata
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── TODO.md                         # Detailed task list
```

---

## Quick Start

### 1. Setup Environment

```bash
cd reward_model
pip install -r requirements.txt
```

### 2. Setup EPIC-Kitchens Download Scripts (One-Time)

```bash
# From the project root (data_eval_engine/)
cd ..  # Go to data_eval_engine root
mkdir -p external
cd external
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git

# Verify installation
ls epic-kitchens-download-scripts/epic_downloader.py
# Should show: epic-kitchens-download-scripts/epic_downloader.py
```

**Directory structure after setup:**
```
data_eval_engine/
├── external/
│   └── epic-kitchens-download-scripts/    # Official download scripts
│       └── epic_downloader.py
└── reward_model/
    └── scripts/
        └── 01_download_videos.py          # Our wrapper
```

### 3. Prepare Data

```bash
cd reward_model  # Back to reward_model directory

# Option A: Download specific video IDs
python scripts/01_download_videos.py --video_ids P01_101,P01_102,P01_103

# Option B: Download all videos from EPIC-100 annotations
python scripts/01_download_videos.py --from_annotations data/annotations/epic_100_train.csv

# Option C: Download clips from S3 (if shared by team)
python scripts/03_sync_with_s3.py --download --all
```

### 3. Extract Features

```bash
# Extract all features for all clips
python features/extract_all_features.py --input data/extracted_clips/ --output features/outputs/
```

### 4. Collect Human Annotations

```bash
# Launch annotation tool
cd ../human_annotation_tool
python app.py

# After collecting annotations, process them
cd ../reward_model
python human_preferences/process_annotations.py --annotations ../human_annotation_tool/results/annotations.csv
```

### 5. Train Reward Model

```bash
# Train model with human preferences
python models/train.py --config configs/training_config.yaml

# Monitor training
tensorboard --logdir models/logs/
```

### 6. Validate Model

```bash
# Run validation pipeline
python validation/simulate_downstream.py --model models/checkpoints/best_model.pth

# Compute correlation
python validation/compute_correlation.py --results validation/results/
```

---

## Workflow

### Phase 1: Data Collection
1. Extract clips from videos → `data/extracted_clips/`
2. Upload to S3 for team sharing → `scripts/03_sync_with_s3.py --upload`
3. Team members download clips → `scripts/03_sync_with_s3.py --download`

### Phase 2: Feature Extraction
1. Run feature extractors → `features/extract_all_features.py`
2. Generate composite scores → `features/compute_composite_score.py`
3. Features saved to → `features/outputs/all_features.csv`

### Phase 3: Human Annotation
1. Launch annotation tool → `human_annotation_tool/app.py`
2. Collect pairwise comparisons (target: 200-500 pairs)
3. Process annotations → `human_preferences/process_annotations.py`

### Phase 4: Model Training
1. Configure hyperparameters → `configs/model_config.yaml`
2. Train model → `models/train.py`
3. Monitor metrics: pair-AUC, loss, score histograms
4. Save checkpoints → `models/checkpoints/`

### Phase 5: Validation
1. Train toy BC model → `validation/toy_bc/train_bc.py`
2. Simulate ΔM → `validation/simulate_downstream.py`
3. Compute correlation ρ(R_h, ΔM_sim)
4. Target: ρ ≥ 0.5

### Phase 6: Ablation Studies
1. Remove motion/hand features
2. Retrain model
3. Measure drop in ρ (expect ≥ 0.1 drop)

---

## Configuration

Edit `configs/*.yaml` files to customize:

- **Data sources**: Which datasets to use (EPIC, Ego4D, GoPro)
- **Features**: Which feature extractors to enable
- **Model architecture**: Hidden dims, dropout, activation functions
- **Training**: Learning rate, batch size, epochs, β for BT loss
- **Validation**: BC model architecture, train/val split

---

## Key Design Decisions

### Architecture
- **Trunk**: Shared MLP (input_dim → 256 → 128)
- **Human Head**: MLP (128 → 64 → 1) for pairwise preferences
- **Robot Head**: Stub for future ΔM data integration

### Loss Function
- **Bradley-Terry Loss** with β = 1
- P(A > B) = σ(β * (R(A) - R(B)))

### Features
- **Visual Quality**: Blur score, exposure score
- **Motion**: Variance, jerk (smoothness)
- **Hands**: MediaPipe hand detection
- **Semantics**: CLIP embeddings (512-D)
- **Extensible**: Easy to add new features via `features/extractors/`

### Validation Strategy
- Can't measure real robot ΔM yet
- Proxy: Train toy BC model, measure validation loss difference
- Correlation ρ(R_h, ΔM_sim) should be ≥ 0.5

---

## Dependencies

Core libraries:
- `torch` - Model training
- `opencv-python` - Video processing
- `mediapipe` - Hand detection
- `transformers` - CLIP embeddings
- `pandas` - Data management
- `tensorboard` - Training visualization
- `pyyaml` - Config files
- `boto3` - S3 integration

See `requirements.txt` for full list.

---

## Next Steps

See [TODO.md](TODO.md) for detailed task breakdown and implementation checklist.

---

**Status**: Initial structure setup complete. Ready for implementation.
