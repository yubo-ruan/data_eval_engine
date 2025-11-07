"""
Download the actual EPIC-Kitchens-100 P01_01 video that matches the annotations.
EPIC-Kitchens-100 videos are hosted on the University of Bristol servers.
"""

import urllib.request
import json
from pathlib import Path

# Load the selected segment to know which video we need
with open('data/selected_segment.json', 'r') as f:
    segment = json.load(f)

video_id = segment['video_id']
participant_id = segment['participant_id']

print(f"Need to download: {video_id}.MP4")
print(f"Participant: {participant_id}")

# EPIC-Kitchens-100 video download information
print("\n" + "="*60)
print("EPIC-KITCHENS-100 VIDEO DOWNLOAD")
print("="*60)
print("\nThe EPIC-Kitchens-100 videos are large files hosted on:")
print("https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d")
print("\nOR via the official download script:")
print("https://github.com/epic-kitchens/epic-kitchens-download-scripts")
print("\n" + "="*60)
print("\nOPTION 1: Direct Download (Recommended)")
print("="*60)
print("1. Visit: https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d")
print(f"2. Navigate to: RGB_frames_and_videos/videos/{participant_id}/")
print(f"3. Download: {video_id}.MP4")
print(f"4. Save to: data/videos/{video_id}.MP4")

print("\n" + "="*60)
print("OPTION 2: Use Official Download Script")
print("="*60)
print("Run these commands:")
print(f"""
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
cd epic-kitchens-download-scripts
python download_videos.py --participants {participant_id} --video_ids {video_id}
# Then move the downloaded video to data/videos/
""")

print("\n" + "="*60)
print("OPTION 3: Use wget/curl (if direct link available)")
print("="*60)
print("Note: Direct URLs may require authentication")
print(f"You may be able to use a direct link format like:")
print(f"https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/{participant_id}/{video_id}.MP4")

print("\n" + "="*60)
print("\nAfter downloading, verify the file:")
print(f"  ls -lh data/videos/{video_id}.MP4")
print("\nThen continue with:")
print("  python scripts/02_extract_single_clip.py")
print("="*60)
