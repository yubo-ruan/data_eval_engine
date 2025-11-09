# Future Improvements & TODO

## Project Goal
Build a production-ready reward model for evaluating egocentric video clips for robot manipulation training, using a diverse dataset from multiple sources.

---

## 🎥 **Data Sources & Infrastructure**

### Current State
- Using single EPIC-Kitchens-100 video (P01_102.mp4) locally
- Manual download process
- Limited to 20 clips (10 annotated + 10 random)

### Required Changes

#### 1. **Cloud Video Storage**
- [ ] Set up cloud storage (AWS S3, Google Cloud Storage, or Azure Blob)
- [ ] Upload diverse video dataset:
  - EPIC-Kitchens-100 videos
  - Ego4D dataset videos
  - Self-recorded GoPro footage
- [ ] Organize by source and metadata
- [ ] Implement access controls and versioning

**Structure:**
```
cloud-storage/
├── epic_kitchens/
│   ├── P01_101.mp4
│   ├── P01_102.mp4
│   └── ...
├── ego4d/
│   ├── video_001.mp4
│   └── ...
└── gopro/
    ├── session_001.mp4
    └── ...
```

#### 2. **Video Download Script (00_download_videos.py)**
- [ ] Replace `01_download_sample.py` with robust downloader
- [ ] Download from cloud storage instead of manual process
- [ ] Support multiple data sources (EPIC, Ego4D, GoPro)
- [ ] Implement:
  - Progress tracking
  - Resume capability
  - Parallel downloads
  - Checksum verification
  - Automatic retry on failure
- [ ] Store videos in organized local structure
- [ ] **Do NOT commit MP4 files to git** (use .gitignore)

**New Script Features:**
```python
# 00_download_videos.py
- download_from_cloud(source='epic_kitchens', video_ids=[...])
- download_from_cloud(source='ego4d', video_ids=[...])
- download_from_cloud(source='gopro', session_ids=[...])
- verify_downloads()
- cleanup_old_videos()
```

#### 3. **Metadata Management (01_prepare_metadata.py)**
- [ ] Create unified metadata file for all sources
- [ ] Map annotations across different datasets:
  - EPIC-Kitchens: CSV annotations
  - Ego4D: JSON annotations
  - GoPro: Manual annotations or auto-generated
- [ ] Standardize action labels across datasets
- [ ] Track video source, quality metrics, action types

**Unified Metadata Format:**
```json
{
  "clip_id": "epic_P01_102_001",
  "source": "epic_kitchens",
  "video_id": "P01_102",
  "start_frame": 1545,
  "stop_frame": 1866,
  "action": "wash knife",
  "verb": "wash",
  "noun": "knife",
  "original_annotation": {...}
}
```

---

## 📁 **Repository Structure Redesign**

### Current Structure
```
data_eval_engine/
├── epic_prototype/
│   ├── data/
│   │   ├── videos/          # Local videos (gitignored)
│   │   └── mixed_clips/
│   └── scripts/
└── human_annotation_tool/
```

### Proposed New Structure
```
data_eval_engine/
├── data/                              # All data (gitignored)
│   ├── raw_videos/                    # Downloaded source videos
│   │   ├── epic_kitchens/
│   │   ├── ego4d/
│   │   └── gopro/
│   ├── annotations/                   # Source annotations
│   │   ├── epic_100_train.csv
│   │   ├── ego4d_annotations.json
│   │   └── gopro_manual_labels.csv
│   ├── extracted_clips/               # Processed clips
│   │   ├── annotated/
│   │   └── random/
│   └── metadata/
│       ├── unified_metadata.json
│       ├── clip_features.csv
│       └── download_manifest.json
│
├── scripts/                           # All processing scripts
│   ├── 00_download_videos.py         # NEW: Download from cloud
│   ├── 01_prepare_metadata.py        # NEW: Unify annotations
│   ├── 02_extract_clips.py           # UPDATED: Multi-source extraction
│   ├── 03_compute_features.py
│   ├── 04_visualize_clip.py
│   ├── 05_extract_mixed_clips.py
│   ├── 06_compute_all_features.py
│   └── 07_reencode_clips_for_web.py
│
├── annotation_tool/                   # Renamed from human_annotation_tool
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── results/
│
├── reward_model/                      # NEW: Model training
│   ├── train.py
│   ├── model.py
│   ├── evaluate.py
│   └── checkpoints/
│
├── configs/                           # NEW: Configuration files
│   ├── cloud_storage.yaml
│   ├── data_sources.yaml
│   └── model_config.yaml
│
├── notebooks/                         # Analysis notebooks
│   └── exploratory_analysis.ipynb
│
├── tests/                             # Unit tests
│   └── test_data_processing.py
│
├── .gitignore                         # Ignore videos, models
├── README.md
├── requirements.txt
└── FUTURE_IMPROVEMENTS.md             # This file
```

---

## 🔧 **Script Updates Required**

### 02_extract_clips.py (formerly 02_extract_single_clip.py)
**Current:** Extracts single clip from one video
**Needed:**
- [ ] Accept multiple video sources
- [ ] Handle different annotation formats
- [ ] Batch extraction across all videos
- [ ] Support different FPS and resolutions
- [ ] Maintain source tracking in metadata

```python
# Updated interface
extract_clips(
    source='epic_kitchens',
    video_ids=['P01_101', 'P01_102'],
    clip_type='annotated',  # or 'random'
    output_dir='data/extracted_clips/'
)
```

### 05_extract_mixed_clips.py
**Current:** 50/50 mix from single video
**Needed:**
- [ ] Sample clips across ALL video sources
- [ ] Stratified sampling by source and action type
- [ ] Configurable ratios (not just 50/50)
- [ ] Ensure diversity in action types

---

## 🧠 **Reward Model Training** (NEW)

### To Implement
- [ ] Create `reward_model/train.py`
- [ ] Use pairwise preferences from annotations
- [ ] Train MLP on extracted features
- [ ] Implement Bradley-Terry model or similar
- [ ] Cross-validation and evaluation metrics
- [ ] Model checkpointing
- [ ] Hyperparameter tuning

### Model Architecture
```python
# Input: Concatenated features from both clips
# - Visual quality (2 features)
# - Motion (2 features)
# - Hand detection (1 feature)
# - CLIP embedding (512 features)
# Total: ~517 features per clip × 2 = 1034 input features

# Output: Preference probability (which clip is better)
```

---

## 📊 **Annotation Tool Improvements**

### Current State
- Basic pairwise comparison
- 30 annotations collected
- Single annotator

### Future Improvements
- [ ] Multi-annotator support with tracking
- [ ] Inter-annotator agreement metrics (Cohen's Kappa)
- [ ] Active learning: prioritize uncertain pairs
- [ ] Show clip features during annotation (optional)
- [ ] Export to common formats (Labelbox, CVAT)
- [ ] Progress dashboard with coverage heatmap
- [ ] Keyboard shortcuts documentation page

---

## 🔬 **Feature Extraction Improvements**

### Current Features
- Blur score (Laplacian variance)
- Exposure score (histogram entropy)
- Motion variance (optical flow)
- Jerk score (motion smoothness)
- Hand visibility (MediaPipe)
- CLIP embeddings (512-D)

### Additional Features to Consider
- [ ] Object detection (detect tools, ingredients)
- [ ] Action recognition (pre-trained model scores)
- [ ] Audio features (if available)
- [ ] 3D hand pose estimation
- [ ] Scene change detection
- [ ] Temporal consistency metrics
- [ ] Task-specific features (grasp quality, object manipulation)

---

## 🏗️ **Infrastructure & Deployment**

### Development Environment
- [ ] Docker containerization
- [ ] Requirements pinning and dependency management
- [ ] Environment variables for cloud credentials
- [ ] Logging and monitoring setup

### Cloud Deployment
- [ ] Deploy annotation tool to cloud (Heroku, AWS, etc.)
- [ ] Set up CI/CD pipeline
- [ ] Automated testing
- [ ] Model serving API (FastAPI)

---

## 📝 **Documentation Needs**

- [ ] API documentation for scripts
- [ ] Data schema documentation
- [ ] Model training guide
- [ ] Annotation guidelines for new annotators
- [ ] Deployment guide
- [ ] Contributing guidelines

---

## ⚠️ **Known Issues & Considerations**

### Current Issues
1. **Composite score doesn't reflect task utility**
   - High sharpness doesn't mean useful for robot training
   - Need human labels to learn what matters

2. **Video codec compatibility**
   - Had to re-encode from mp4v to H.264
   - Future: extract directly to H.264

3. **Limited diversity**
   - Single video source
   - Limited action types
   - Need more varied scenarios

### Design Decisions to Revisit
- [ ] Should we weight features in composite score?
- [ ] How to handle ties in pairwise comparisons?
- [ ] Optimal clip duration (currently 3s)?
- [ ] Resolution vs. file size tradeoff (currently 480x480)?
- [ ] FPS for analysis (currently 15fps)?

---

## 📅 **Implementation Priority**

### Phase 1: Data Infrastructure (Next Sprint)
1. Set up cloud storage
2. Create download scripts
3. Restructure repository
4. Update .gitignore

### Phase 2: Multi-Source Support
1. Unify metadata across sources
2. Update extraction scripts
3. Collect diverse clips

### Phase 3: Scale Annotation
1. Deploy annotation tool
2. Recruit multiple annotators
3. Collect 200-500 annotations

### Phase 4: Model Training
1. Implement reward model
2. Train on collected preferences
3. Evaluate and iterate

### Phase 5: Production
1. Model serving API
2. Integration with robot training pipeline
3. Continuous improvement loop

---

## 🔗 **External Resources**

- EPIC-Kitchens-100: https://epic-kitchens.github.io/
- Ego4D: https://ego4d-data.org/
- Reward Learning Papers: [Add relevant papers]
- Cloud Storage Docs: [Add links when chosen]

---

## 💡 **Ideas to Explore**

- Active learning for annotation efficiency
- Self-supervised pre-training on unlabeled videos
- Multi-task learning (predict both quality and action)
- Temporal modeling (not just single clips)
- Cross-dataset transfer learning
- Uncertainty estimation in reward predictions

---

**Last Updated:** 2025-11-08
**Status:** Work in Progress - Prototype Phase Complete
