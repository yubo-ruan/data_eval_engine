"""
Download a single EPIC-Kitchens clip for testing.
We'll use P01_01 video segment with a manipulation action.
"""

import urllib.request
import pandas as pd
from pathlib import Path

# Create directories
Path('data/videos').mkdir(parents=True, exist_ok=True)
Path('data/annotations').mkdir(parents=True, exist_ok=True)

# Download annotation file to find a good segment
print("Downloading annotation file...")
annotation_url = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv"
urllib.request.urlretrieve(annotation_url, 'data/annotations/EPIC_100_train.csv')

# Load annotations and find a manipulation action
df = pd.read_csv('data/annotations/EPIC_100_train.csv')

# Filter for a good example: P01, verb='wash', duration 2-4s
example = df[
    (df['participant_id'] == 'P01') &
    (df['verb'] == 'wash') &
    (df['stop_frame'] - df['start_frame'] < 240)  # ~4 seconds at 60fps
].iloc[0]

print(f"\nSelected example clip:")
print(f"Video: {example['video_id']}")
print(f"Action: {example['narration']}")
print(f"Duration: {(example['stop_frame'] - example['start_frame'])/60:.2f} seconds")
print(f"\nYou need to manually download {example['video_id']}.MP4 from:")
print("https://uob-my.sharepoint.com/personal/wq23021_bristol_ac_uk/_layouts/15/onedrive.aspx")
print(f"Navigate to: Videos/P01/{example['video_id']}.MP4")
print(f"Save to: data/videos/{example['video_id']}.MP4")

# Save the selected segment info for next step
example.to_json('data/selected_segment.json')
