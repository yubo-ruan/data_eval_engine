"""
Create a visualization showing the clip with feature annotations.
Helps validate that features make sense.
"""

import cv2
import json
import numpy as np

# Load features
with open('outputs/test_clip_features.json', 'r') as f:
    features = json.load(f)

# Load clip
cap = cv2.VideoCapture('data/test_clip.mp4')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Creating annotated visualization...")

# Create output video with feature overlay
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/test_clip_annotated.mp4', fourcc, 15, (480, 480))

for frame in frames:
    # Add text overlay
    cv2.putText(frame, f"Quality: {features['composite_score']:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hand Visible: {features['hands']['hand_visible_ratio']:.2%}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Motion: {features['motion']['motion_var']:.3f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)

out.release()
print(f"Saved annotated clip to: outputs/test_clip_annotated.mp4")
print("\n✅ Visualization complete! Review the annotated clip.")
