"""
Extract a 3-second clip from the downloaded video.
This validates our video processing pipeline.
"""

import cv2
import json
import numpy as np
from pathlib import Path

# Load selected segment
with open('data/selected_segment.json', 'r') as f:
    segment = json.load(f)

video_path = f"data/videos/{segment['video_id']}.MP4"
output_path = f"data/test_clip.mp4"

print(f"Processing video: {video_path}")

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {video_path}. Did you download it?")

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(segment['start_frame'])
stop_frame = int(segment['stop_frame'])

print(f"Original FPS: {fps}")
print(f"Extracting frames {start_frame} to {stop_frame}")

# Set to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Extract frames at 15 fps
target_fps = 15
frame_interval = int(fps / target_fps)
frames = []

for i in range(start_frame, stop_frame):
    ret, frame = cap.read()
    if not ret:
        break

    if (i - start_frame) % frame_interval == 0:
        # Resize to 480x480
        frame_resized = cv2.resize(frame, (480, 480))
        frames.append(frame_resized)

cap.release()

print(f"Extracted {len(frames)} frames")

# Save as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, target_fps, (480, 480))

for frame in frames:
    out.write(frame)

out.release()

print(f"Saved test clip to: {output_path}")
print(f"Duration: {len(frames)/target_fps:.2f} seconds")

# Save frames for feature extraction
np.save('data/test_clip_frames.npy', np.array(frames))
print(f"Saved frames array for feature extraction")
