# EPIC-Kitchens Single-Clip Prototype Workflow

## Project Context
End Goal: Build a reward model that predicts which egocentric video clips are most useful for robot manipulation training.

Current Task: Validate the feature extraction pipeline on a single clip before scaling to thousands.

## Directory Structure
```
epic_prototype/
├── data/
│   ├── videos/          # Place downloaded videos here
│   ├── annotations/     # EPIC-Kitchens annotations
│   ├── test_clip.mp4    # Extracted test clip
│   └── test_clip_frames.npy  # Frames array
├── scripts/
│   ├── 01_download_sample.py
│   ├── 02_extract_single_clip.py
│   ├── 03_compute_features.py
│   └── 04_visualize_clip.py
└── outputs/
    ├── test_clip_features.json
    └── test_clip_annotated.mp4
```

## Execution Order

### Step 1: Download Sample Clip Info
```bash
cd epic_prototype
python scripts/01_download_sample.py
```
This script will:
- Download EPIC-Kitchens annotation file
- Select a good manipulation action example
- Display instructions for manual video download

### Step 2: Manual Video Download
After running Step 1, follow the instructions to manually download the video from the EPIC-Kitchens OneDrive link.
Save it to `data/videos/[video_id].MP4`

### Step 3: Extract Test Clip
```bash
python scripts/02_extract_single_clip.py
```
This script will:
- Extract a 2-4 second clip from the downloaded video
- Resize frames to 480x480
- Downsample to 15 FPS
- Save as `data/test_clip.mp4` and `data/test_clip_frames.npy`

### Step 4: Compute Features (Main Validation)
```bash
python scripts/03_compute_features.py
```
This script computes:
- Visual quality (blur, exposure)
- Motion metrics (variance, jerk)
- Hand detection using MediaPipe
- CLIP embeddings (512-D)
- Composite quality score

Output: `outputs/test_clip_features.json`

### Step 5: Visualize Results (Optional)
```bash
python scripts/04_visualize_clip.py
```
Creates annotated video with overlaid feature metrics.

Output: `outputs/test_clip_annotated.mp4`

## Success Criteria
After running all scripts, you should have:
- ✅ `data/test_clip.mp4` - Clean 2-4 second clip at 480x480, 15fps
- ✅ `outputs/test_clip_features.json` - All features computed successfully
- ✅ Verified that:
  - CLIP embeddings extract (512-D vector)
  - Hand detection works (ratio > 0)
  - Motion/quality metrics are reasonable
  - No errors in the pipeline

## Features Extracted

1. **Visual Quality**
   - Blur score (Laplacian variance): Higher = sharper
   - Exposure score (histogram entropy): Higher = better lighting

2. **Motion Features**
   - Motion variance: Standard deviation of optical flow magnitudes
   - Jerk score: Second derivative of motion (smoothness)

3. **Hand Detection**
   - Hand visible ratio: % of frames with detected hands

4. **CLIP Embeddings**
   - 512-dimensional semantic embedding
   - Averaged across 8 uniformly sampled frames

5. **Composite Score**
   - Weighted average of normalized features (0-1 scale)
   - Used as initial quality heuristic

## Next Steps
1. Review `outputs/test_clip_features.json`
2. Visualize the test clip: `data/test_clip.mp4`
3. Once Ego4D access granted, scale to 5000 clips
4. Then proceed to human annotation and reward model training

## Dependencies
All dependencies are installed via:
```bash
pip install opencv-python numpy pandas torch torchvision transformers pillow mediapipe tqdm scikit-learn
```
