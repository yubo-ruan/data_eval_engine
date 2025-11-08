"""
Compute features for all extracted clips (annotated + random).

This script:
1. Loads all clips from data/mixed_clips/
2. Computes features for each clip using the same pipeline as 03_compute_features.py
3. Compiles all features into a single CSV/JSON file

Features computed:
- Visual quality (blur, exposure)
- Motion metrics (variance, jerk)
- Hand detection (visibility ratio)
- CLIP embeddings (512-D)
- Composite quality score
"""

import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import mediapipe as mp
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("Loading models...")
# CLIP for semantic embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

print(f"Using device: {device}")

# Load clip metadata
metadata_df = pd.read_csv('data/mixed_clips/clip_metadata.csv')
print(f"\nLoaded metadata for {len(metadata_df)} clips")

# ============================================
# Feature Extraction Functions
# ============================================
def compute_blur_score(frame):
    """Variance of Laplacian - measures sharpness"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_exposure_score(frame):
    """Histogram entropy - measures lighting richness"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_optical_flow(frame1, frame2):
    """Compute dense optical flow between consecutive frames"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def extract_features(video_path):
    """Extract all features from a single clip"""

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames)

    # ============================================
    # VISUAL QUALITY FEATURES
    # ============================================
    blur_scores = [compute_blur_score(f) for f in frames]
    exposure_scores = [compute_exposure_score(f) for f in frames]

    blur_score = np.mean(blur_scores)
    exposure_score = np.mean(exposure_scores)

    # ============================================
    # MOTION FEATURES
    # ============================================
    flow_magnitudes = []

    for i in range(len(frames) - 1):
        flow = compute_optical_flow(frames[i], frames[i+1])
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(magnitude.mean())

    motion_var = np.std(flow_magnitudes) if flow_magnitudes else 0.0

    # Jerk (second derivative of motion)
    if len(flow_magnitudes) >= 3:
        jerks = [flow_magnitudes[i+1] - 2*flow_magnitudes[i] + flow_magnitudes[i-1]
                 for i in range(1, len(flow_magnitudes)-1)]
        jerk_score = np.mean(np.abs(jerks))
    else:
        jerk_score = 0.0

    # ============================================
    # HAND DETECTION
    # ============================================
    hand_detections = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_detections.append(results.multi_hand_landmarks is not None)

    hand_visible_ratio = sum(hand_detections) / len(hand_detections)

    # ============================================
    # CLIP EMBEDDINGS
    # ============================================
    # Sample 8 frames uniformly for CLIP
    indices = np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)
    sampled_frames = [frames[i] for i in indices]

    # Convert to PIL
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                  for f in sampled_frames]

    # Process with CLIP
    inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Average across frames
    clip_embedding = image_features.mean(dim=0).cpu().numpy()

    # ============================================
    # COMPOSITE SCORE
    # ============================================
    # Normalize each feature to 0-1 range (using reasonable ranges)
    normalized_blur = min(blur_score / 1000, 1.0)  # Higher = sharper
    normalized_exposure = min(exposure_score / 8, 1.0)  # Higher = better lighting
    normalized_motion = min(motion_var / 10, 1.0)  # Higher = more dynamic
    normalized_jerk = max(1.0 - jerk_score / 5, 0.0)  # Lower jerk = smoother
    normalized_hands = hand_visible_ratio  # Already 0-1

    # Weighted composite
    composite_score = (
        0.2 * normalized_blur +
        0.2 * normalized_exposure +
        0.2 * normalized_motion +
        0.2 * normalized_hands +
        0.2 * normalized_jerk
    )

    return {
        'blur_score': float(blur_score),
        'exposure_score': float(exposure_score),
        'motion_var': float(motion_var),
        'jerk_score': float(jerk_score),
        'hand_visible_ratio': float(hand_visible_ratio),
        'clip_embedding_mean': float(clip_embedding.mean()),
        'clip_embedding_std': float(clip_embedding.std()),
        'composite_score': float(composite_score),
        'normalized_blur': float(normalized_blur),
        'normalized_exposure': float(normalized_exposure),
        'normalized_motion': float(normalized_motion),
        'normalized_jerk': float(normalized_jerk),
        'clip_embedding': clip_embedding.tolist()
    }

# ============================================
# Process All Clips
# ============================================
print("\n" + "="*60)
print("COMPUTING FEATURES FOR ALL CLIPS")
print("="*60)

all_features = []

for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing clips"):
    clip_id = row['clip_id']
    clip_type = row['type']
    filename = row['filename']

    # Construct video path
    if clip_type == 'annotated':
        video_path = f"data/mixed_clips/annotated/{filename}"
    else:
        video_path = f"data/mixed_clips/random/{filename}"

    # Extract features
    features = extract_features(video_path)

    if features is None:
        print(f"Warning: Could not extract features for {clip_id}")
        continue

    # Combine metadata and features
    clip_data = {
        'clip_id': clip_id,
        'filename': filename,
        'type': clip_type,
        **features
    }

    # Add narration info for annotated clips
    if clip_type == 'annotated':
        clip_data['narration'] = row['narration']
        clip_data['verb'] = row['verb']
        clip_data['noun'] = row['noun']

    all_features.append(clip_data)

print(f"\nSuccessfully extracted features for {len(all_features)} clips")

# ============================================
# Save Results
# ============================================
# Create features DataFrame (excluding embedding vectors for CSV)
features_for_csv = []
for clip in all_features:
    clip_copy = clip.copy()
    clip_copy.pop('clip_embedding', None)  # Remove embedding for CSV
    features_for_csv.append(clip_copy)

features_df = pd.DataFrame(features_for_csv)

# Save CSV (without embeddings)
features_df.to_csv('data/mixed_clips/all_features.csv', index=False)
print(f"Saved features to: data/mixed_clips/all_features.csv")

# Save JSON (with embeddings)
with open('data/mixed_clips/all_features.json', 'w') as f:
    json.dump(all_features, f, indent=2)
print(f"Saved features with embeddings to: data/mixed_clips/all_features.json")

# ============================================
# Summary Statistics
# ============================================
print("\n" + "="*60)
print("FEATURE EXTRACTION SUMMARY")
print("="*60)

annotated = features_df[features_df['type'] == 'annotated']
random = features_df[features_df['type'] == 'random']

print(f"\nTotal clips: {len(features_df)}")
print(f"  - Annotated: {len(annotated)}")
print(f"  - Random: {len(random)}")

print(f"\nComposite Quality Score:")
print(f"  - Annotated clips: {annotated['composite_score'].mean():.3f} ± {annotated['composite_score'].std():.3f}")
print(f"  - Random clips: {random['composite_score'].mean():.3f} ± {random['composite_score'].std():.3f}")

print(f"\nHand Visibility:")
print(f"  - Annotated clips: {annotated['hand_visible_ratio'].mean():.2%} ± {annotated['hand_visible_ratio'].std():.2%}")
print(f"  - Random clips: {random['hand_visible_ratio'].mean():.2%} ± {random['hand_visible_ratio'].std():.2%}")

print(f"\nMotion Variance:")
print(f"  - Annotated clips: {annotated['motion_var'].mean():.3f} ± {annotated['motion_var'].std():.3f}")
print(f"  - Random clips: {random['motion_var'].mean():.3f} ± {random['motion_var'].std():.3f}")

print(f"\nBlur Score (Sharpness):")
print(f"  - Annotated clips: {annotated['blur_score'].mean():.1f} ± {annotated['blur_score'].std():.1f}")
print(f"  - Random clips: {random['blur_score'].mean():.1f} ± {random['blur_score'].std():.1f}")

print("\n" + "="*60)
print("Next steps:")
print("1. Review data/mixed_clips/all_features.csv")
print("2. Analyze quality differences between annotated and random clips")
print("3. Build human annotation interface")
print("="*60)
