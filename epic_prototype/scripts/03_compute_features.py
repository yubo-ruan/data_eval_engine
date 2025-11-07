"""
Compute all features that will be used in the reward model:
- Visual quality (blur, exposure)
- Motion metrics (variance, jerk)
- Hand detection
- CLIP embeddings

This validates our feature extraction pipeline end-to-end.
"""

import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import mediapipe as mp
import json

print("Loading models...")
# CLIP for semantic embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

print(f"Using device: {device}")

# Load frames
frames = np.load('data/test_clip_frames.npy')
print(f"Loaded {len(frames)} frames")

# ============================================
# VISUAL QUALITY FEATURES
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

print("\n[1/5] Computing visual quality features...")
blur_scores = [compute_blur_score(f) for f in frames]
exposure_scores = [compute_exposure_score(f) for f in frames]

blur_score = np.mean(blur_scores)
exposure_score = np.mean(exposure_scores)

print(f"  Blur score (sharpness): {blur_score:.2f}")
print(f"  Exposure score (lighting): {exposure_score:.2f}")

# ============================================
# MOTION FEATURES
# ============================================
def compute_optical_flow(frame1, frame2):
    """Compute dense optical flow between consecutive frames"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

print("\n[2/5] Computing motion features...")
flow_magnitudes = []

for i in range(len(frames) - 1):
    flow = compute_optical_flow(frames[i], frames[i+1])
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flow_magnitudes.append(magnitude.mean())

motion_var = np.std(flow_magnitudes)

# Jerk (second derivative of motion)
if len(flow_magnitudes) >= 3:
    jerks = [flow_magnitudes[i+1] - 2*flow_magnitudes[i] + flow_magnitudes[i-1]
             for i in range(1, len(flow_magnitudes)-1)]
    jerk_score = np.mean(np.abs(jerks))
else:
    jerk_score = 0.0

print(f"  Motion variance: {motion_var:.4f}")
print(f"  Jerk score: {jerk_score:.4f}")

# ============================================
# HAND DETECTION
# ============================================
print("\n[3/5] Detecting hands...")
hand_detections = []

for frame in frames:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detections.append(results.multi_hand_landmarks is not None)

hand_visible_ratio = sum(hand_detections) / len(hand_detections)
print(f"  Hand visible in {hand_visible_ratio*100:.1f}% of frames")

# ============================================
# CLIP EMBEDDINGS
# ============================================
print("\n[4/5] Computing CLIP embeddings...")

# Sample 8 frames uniformly for CLIP
indices = np.linspace(0, len(frames)-1, 8, dtype=int)
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

print(f"  CLIP embedding shape: {clip_embedding.shape}")
print(f"  CLIP embedding mean: {clip_embedding.mean():.4f}")

# ============================================
# COMPOSITE SCORE (Simple heuristic)
# ============================================
print("\n[5/5] Computing composite score...")

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

print(f"  Composite quality score: {composite_score:.3f}")

# ============================================
# SAVE RESULTS
# ============================================
features = {
    'visual_quality': {
        'blur_score': float(blur_score),
        'exposure_score': float(exposure_score)
    },
    'motion': {
        'motion_var': float(motion_var),
        'jerk_score': float(jerk_score)
    },
    'hands': {
        'hand_visible_ratio': float(hand_visible_ratio)
    },
    'clip_embedding': {
        'shape': list(clip_embedding.shape),
        'mean': float(clip_embedding.mean()),
        'std': float(clip_embedding.std()),
        'embedding': clip_embedding.tolist()
    },
    'composite_score': float(composite_score)
}

with open('outputs/test_clip_features.json', 'w') as f:
    json.dump(features, f, indent=2)

print(f"\n✅ Features saved to: outputs/test_clip_features.json")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("FEATURE EXTRACTION VALIDATION COMPLETE")
print("="*60)
print(f"Clip quality: {composite_score:.3f} / 1.0")
print(f"")
print(f"Visual Quality:")
print(f"  - Sharpness: {'Good' if normalized_blur > 0.5 else 'Poor'} ({normalized_blur:.2f})")
print(f"  - Lighting: {'Good' if normalized_exposure > 0.5 else 'Poor'} ({normalized_exposure:.2f})")
print(f"")
print(f"Motion Characteristics:")
print(f"  - Dynamics: {'Good' if normalized_motion > 0.3 else 'Static'} ({normalized_motion:.2f})")
print(f"  - Smoothness: {'Good' if normalized_jerk > 0.5 else 'Jerky'} ({normalized_jerk:.2f})")
print(f"")
print(f"Hand Visibility: {'Good' if hand_visible_ratio > 0.3 else 'Poor'} ({hand_visible_ratio:.2%})")
print(f"")
print(f"CLIP Embedding: {clip_embedding.shape[0]} dimensions extracted")
print("="*60)
print("\nNext steps:")
print("1. Review outputs/test_clip_features.json")
print("2. Visualize the test clip: data/test_clip.mp4")
print("3. Once Ego4D access granted, scale to 5000 clips")
print("4. Then proceed to human annotation (Step 3)")
