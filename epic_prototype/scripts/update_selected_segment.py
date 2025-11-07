"""
Update the selected segment to a different action.
Usage: Modify the filters below to select your desired clip.
"""

import pandas as pd
import json

# Load annotations
df = pd.read_csv('data/annotations/EPIC_100_train.csv')

# Get the wash knife action from P01_102
segment = df[
    (df['video_id'] == 'P01_102') &
    (df['verb'] == 'wash') &
    (df['noun'] == 'knife')
].iloc[0]

# Save to JSON
segment.to_json('data/selected_segment.json')

print('Updated selected segment:')
print(f'Video: {segment["video_id"]}')
print(f'Action: {segment["narration"]}')
print(f'Frames: {segment["start_frame"]} to {segment["stop_frame"]}')
print(f'Duration: {(segment["stop_frame"] - segment["start_frame"])/50:.2f} seconds (assuming 50fps)')
print(f'Timestamp: {segment["start_timestamp"]} to {segment["stop_timestamp"]}')
print(f'\nNow download: {segment["video_id"]}.MP4')
print(f'Save to: data/videos/{segment["video_id"]}.MP4')
