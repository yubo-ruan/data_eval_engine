"""
Extract a dataset of clips with varying quality for human annotation.

Strategy:
1. Annotated Clips (50%): From labeled manipulation segments (high quality)
2. Random Temporal Windows (50%): From anywhere in videos (variable quality)

This creates meaningful quality variance for training the reward model.
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Configuration
VIDEO_ID = 'P01_102'
VIDEO_PATH = f'data/videos/{VIDEO_ID}.mp4'
ANNOTATION_CSV = 'data/annotations/EPIC_100_train.csv'
OUTPUT_DIR = 'data/mixed_clips'
RANDOM_SEED = 42

# Clip parameters
TARGET_CLIP_DURATION = 3.0  # seconds
TARGET_FPS = 15
CLIP_SIZE = (480, 480)

# How many clips to extract
NUM_ANNOTATED_CLIPS = 10  # 50% from annotations
NUM_RANDOM_CLIPS = 10     # 50% random temporal windows

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f'{OUTPUT_DIR}/annotated').mkdir(parents=True, exist_ok=True)
Path(f'{OUTPUT_DIR}/random').mkdir(parents=True, exist_ok=True)

print("="*60)
print("MIXED CLIP EXTRACTION")
print("="*60)
print(f"Video: {VIDEO_ID}")
print(f"Target: {NUM_ANNOTATED_CLIPS} annotated + {NUM_RANDOM_CLIPS} random clips")
print(f"Clip duration: {TARGET_CLIP_DURATION}s at {TARGET_FPS} fps")
print(f"Random seed: {RANDOM_SEED}")
print()

# ============================================
# PART 1: Extract Annotated Clips
# ============================================
print("[1/2] Extracting annotated clips from EPIC-100 segments...")

# Load annotations
df = pd.read_csv(ANNOTATION_CSV)

# Filter for this video
video_segments = df[df['video_id'] == VIDEO_ID].copy()

print(f"Found {len(video_segments)} annotated segments in {VIDEO_ID}")

# Sample annotated clips
if len(video_segments) >= NUM_ANNOTATED_CLIPS:
    selected_segments = video_segments.sample(n=NUM_ANNOTATED_CLIPS, random_state=RANDOM_SEED)
else:
    print(f"Warning: Only {len(video_segments)} segments available, using all")
    selected_segments = video_segments

print(f"Selected {len(selected_segments)} annotated segments")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {VIDEO_PATH}")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = int(original_fps / TARGET_FPS)

print(f"Video FPS: {original_fps}, Total frames: {total_frames}")

# Extract annotated clips
annotated_metadata = []

for idx, (_, segment) in enumerate(selected_segments.iterrows()):
    start_frame = int(segment['start_frame'])
    stop_frame = int(segment['stop_frame'])

    # Calculate center of segment and extract fixed duration around it
    center_frame = (start_frame + stop_frame) // 2
    clip_frames = int(TARGET_CLIP_DURATION * original_fps)
    clip_start = max(0, center_frame - clip_frames // 2)
    clip_stop = min(total_frames, clip_start + clip_frames)

    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frames = []

    for i in range(clip_start, clip_stop):
        ret, frame = cap.read()
        if not ret:
            break

        if (i - clip_start) % frame_interval == 0:
            frame_resized = cv2.resize(frame, CLIP_SIZE)
            frames.append(frame_resized)

    # Save clip
    clip_filename = f'annotated_{idx:03d}_{segment["narration"].replace(" ", "_")[:30]}.mp4'
    clip_path = f'{OUTPUT_DIR}/annotated/{clip_filename}'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, TARGET_FPS, CLIP_SIZE)
    for frame in frames:
        out.write(frame)
    out.release()

    # Save metadata
    annotated_metadata.append({
        'clip_id': f'annotated_{idx:03d}',
        'filename': clip_filename,
        'type': 'annotated',
        'narration': segment['narration'],
        'verb': segment['verb'],
        'noun': segment['noun'],
        'start_frame': clip_start,
        'stop_frame': clip_stop,
        'duration': len(frames) / TARGET_FPS,
        'num_frames': len(frames)
    })

    print(f"  [{idx+1}/{len(selected_segments)}] {segment['narration']}: {len(frames)} frames")

print(f"Extracted {len(annotated_metadata)} annotated clips")

# ============================================
# PART 2: Extract Random Temporal Windows
# ============================================
print(f"\n[2/2] Extracting {NUM_RANDOM_CLIPS} random temporal windows...")

random_metadata = []
clip_duration_frames = int(TARGET_CLIP_DURATION * original_fps)

# Generate random start positions
random_starts = []
for i in range(NUM_RANDOM_CLIPS):
    max_start = total_frames - clip_duration_frames
    random_start = random.randint(0, max_start)
    random_starts.append(random_start)

# Extract random clips
for idx, start_frame in enumerate(random_starts):
    stop_frame = start_frame + clip_duration_frames

    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []

    for i in range(start_frame, stop_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if (i - start_frame) % frame_interval == 0:
            frame_resized = cv2.resize(frame, CLIP_SIZE)
            frames.append(frame_resized)

    # Save clip
    clip_filename = f'random_{idx:03d}.mp4'
    clip_path = f'{OUTPUT_DIR}/random/{clip_filename}'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, TARGET_FPS, CLIP_SIZE)
    for frame in frames:
        out.write(frame)
    out.release()

    # Save metadata
    timestamp = start_frame / original_fps
    random_metadata.append({
        'clip_id': f'random_{idx:03d}',
        'filename': clip_filename,
        'type': 'random',
        'start_frame': start_frame,
        'stop_frame': stop_frame,
        'timestamp': f'{int(timestamp//60):02d}:{int(timestamp%60):02d}',
        'duration': len(frames) / TARGET_FPS,
        'num_frames': len(frames)
    })

    print(f"  [{idx+1}/{NUM_RANDOM_CLIPS}] Random window at frame {start_frame}: {len(frames)} frames")

cap.release()

print(f"Extracted {len(random_metadata)} random clips")

# ============================================
# Save Combined Metadata
# ============================================
all_metadata = annotated_metadata + random_metadata

metadata_df = pd.DataFrame(all_metadata)
metadata_df.to_csv(f'{OUTPUT_DIR}/clip_metadata.csv', index=False)
metadata_df.to_json(f'{OUTPUT_DIR}/clip_metadata.json', orient='records', indent=2)

print(f"\n{'='*60}")
print("EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Total clips extracted: {len(all_metadata)}")
print(f"  - Annotated: {len(annotated_metadata)} ({len(annotated_metadata)/len(all_metadata)*100:.0f}%)")
print(f"  - Random: {len(random_metadata)} ({len(random_metadata)/len(all_metadata)*100:.0f}%)")
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"Metadata saved to: {OUTPUT_DIR}/clip_metadata.csv")
print(f"\nNext steps:")
print("1. Review extracted clips visually")
print("2. Build human annotation interface")
print("3. Collect pairwise comparisons")
