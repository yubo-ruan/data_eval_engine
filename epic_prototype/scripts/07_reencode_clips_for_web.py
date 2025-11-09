"""
Re-encode video clips to H.264 codec for web browser compatibility.

The clips were originally encoded with mp4v codec which doesn't play well in browsers.
This script re-encodes them to H.264 which is universally supported.
"""

import subprocess
from pathlib import Path
from tqdm import tqdm

CLIPS_DIR = Path('data/mixed_clips')

# Find all mp4 files
annotated_clips = list((CLIPS_DIR / 'annotated').glob('*.mp4'))
random_clips = list((CLIPS_DIR / 'random').glob('*.mp4'))
all_clips = annotated_clips + random_clips

print(f"Found {len(all_clips)} clips to re-encode")
print(f"  - Annotated: {len(annotated_clips)}")
print(f"  - Random: {len(random_clips)}")
print()

for clip_path in tqdm(all_clips, desc="Re-encoding clips"):
    # Create temporary output path
    temp_path = clip_path.with_suffix('.temp.mp4')

    # Re-encode with H.264 codec
    cmd = [
        'ffmpeg',
        '-i', str(clip_path),
        '-c:v', 'libx264',      # H.264 codec
        '-preset', 'medium',     # Encoding speed/quality tradeoff
        '-crf', '23',            # Quality (lower = better, 23 is good default)
        '-pix_fmt', 'yuv420p',   # Pixel format for compatibility
        '-movflags', '+faststart', # Enable streaming
        '-y',                    # Overwrite output
        str(temp_path)
    ]

    # Run ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Replace original with re-encoded version
        temp_path.replace(clip_path)
    else:
        print(f"\nError re-encoding {clip_path.name}:")
        print(result.stderr)

print("\n✓ Re-encoding complete!")
print("Videos are now in H.264 format and should play in all browsers")
