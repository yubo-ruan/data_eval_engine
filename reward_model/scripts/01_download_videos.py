"""
Download videos from EPIC-Kitchens-100 dataset.

This script wraps the official EPIC-Kitchens download script to download
specific videos to our local data directory.

Prerequisites:
    Clone the official download scripts:
    git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git external/epic-kitchens-download-scripts

Usage:
    # Download specific video IDs
    python scripts/01_download_videos.py --video_ids P01_101,P01_102,P01_103

    # Download all videos from specific participants
    python scripts/01_download_videos.py --participants P01,P02

    # Download videos mentioned in annotation file
    python scripts/01_download_videos.py --from_annotations data/annotations/epic_100_train.csv

    # Specify custom output directory
    python scripts/01_download_videos.py --video_ids P01_101 --output_dir data/raw_videos/epic_kitchens/
"""

import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EPIC_DOWNLOADER = PROJECT_ROOT.parent / 'external' / 'epic-kitchens-download-scripts' / 'epic_downloader.py'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw_videos' / 'epic_kitchens'
MANIFEST_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'download_manifest.json'

def check_downloader_exists():
    """Check if EPIC-Kitchens download script is available"""
    if not EPIC_DOWNLOADER.exists():
        print("❌ ERROR: EPIC-Kitchens download script not found!")
        print(f"   Expected location: {EPIC_DOWNLOADER}")
        print()
        print("📥 Please clone the official download scripts:")
        print("   cd", PROJECT_ROOT.parent)
        print("   git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git external/epic-kitchens-download-scripts")
        print()
        sys.exit(1)
    print(f"✓ Found EPIC downloader at: {EPIC_DOWNLOADER}")

def get_video_ids_from_annotations(annotation_file):
    """Extract unique video IDs from EPIC-100 annotation CSV"""
    print(f"📄 Reading annotations from: {annotation_file}")

    if not Path(annotation_file).exists():
        print(f"❌ ERROR: Annotation file not found: {annotation_file}")
        sys.exit(1)

    df = pd.read_csv(annotation_file)

    # EPIC-100 format: video_id column contains values like "P01_101"
    if 'video_id' in df.columns:
        video_ids = df['video_id'].unique().tolist()
    elif 'participant_id' in df.columns and 'video_id' in df.columns:
        # Alternative format
        video_ids = df['video_id'].unique().tolist()
    else:
        print("❌ ERROR: Could not find 'video_id' column in annotation file")
        print(f"   Available columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"   Found {len(video_ids)} unique videos in annotations")
    return video_ids

def download_videos(video_ids=None, participant_ids=None, output_dir=None):
    """
    Download EPIC-Kitchens-100 videos using official download script.

    Args:
        video_ids: List of video IDs (e.g., ['P01_101', 'P01_102'])
        participant_ids: List of participant IDs (e.g., ['P01', 'P02'])
        output_dir: Where to save videos
    """

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        'python',
        str(EPIC_DOWNLOADER),
        '--videos',  # Download videos only (not frames)
        '--output-path', str(output_dir)
    ]

    # Add filters
    if participant_ids:
        # Convert to comma-separated string
        participants_str = ','.join(participant_ids)
        cmd.extend(['--participants', participants_str])
        print(f"📥 Downloading videos from participants: {participants_str}")

    if video_ids:
        # Their script expects comma-separated video IDs
        videos_str = ','.join(video_ids)
        cmd.extend(['--specific-videos', videos_str])
        print(f"📥 Downloading {len(video_ids)} specific videos...")
        if len(video_ids) <= 10:
            print(f"   Videos: {video_ids}")

    print(f"💾 Output directory: {output_dir}")
    print()
    print("🚀 Starting download (this may take a while)...")
    print("=" * 60)

    # Run download (must run from epic-kitchens-download-scripts directory)
    epic_downloader_dir = EPIC_DOWNLOADER.parent

    try:
        result = subprocess.run(cmd, check=True, text=True, cwd=str(epic_downloader_dir))
        print("=" * 60)
        print("✅ Download completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Download failed with error code {e.returncode}")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Working directory: {epic_downloader_dir}")
        return False

def verify_downloads(video_ids, output_dir):
    """Verify that downloaded videos exist"""
    print()
    print("🔍 Verifying downloads...")

    output_dir = Path(output_dir)
    found = []
    missing = []

    for video_id in video_ids:
        # EPIC downloader organizes as: P01/videos/P01_101.MP4
        participant = video_id.split('_')[0]
        video_path = output_dir / participant / 'videos' / f'{video_id}.MP4'

        if video_path.exists():
            found.append({
                'video_id': video_id,
                'path': str(video_path),
                'size_mb': video_path.stat().st_size / (1024 * 1024)
            })
        else:
            missing.append(video_id)

    print(f"   ✓ Found: {len(found)} videos")
    if missing:
        print(f"   ✗ Missing: {len(missing)} videos")
        print(f"     {missing}")

    return found, missing

def create_download_manifest(downloaded_videos, output_path=None):
    """Create manifest file tracking downloaded videos"""
    if output_path is None:
        output_path = MANIFEST_PATH

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if it exists
    if output_path.exists():
        with open(output_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {
            'created_at': datetime.now().isoformat(),
            'videos': {}
        }

    # Add new downloads
    manifest['last_updated'] = datetime.now().isoformat()

    for video_info in downloaded_videos:
        video_id = video_info['video_id']
        manifest['videos'][video_id] = {
            'path': video_info['path'],
            'size_mb': round(video_info['size_mb'], 2),
            'downloaded_at': datetime.now().isoformat()
        }

    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"📝 Download manifest saved to: {output_path}")
    print(f"   Total videos tracked: {len(manifest['videos'])}")

def main():
    parser = argparse.ArgumentParser(
        description='Download EPIC-Kitchens-100 videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video_ids',
        type=str,
        help='Comma-separated list of video IDs (e.g., P01_101,P01_102)'
    )
    input_group.add_argument(
        '--participants',
        type=str,
        help='Comma-separated list of participant IDs (e.g., P01,P02)'
    )
    input_group.add_argument(
        '--from_annotations',
        type=str,
        help='Path to EPIC-100 annotation CSV file'
    )

    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for videos (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--skip_verification',
        action='store_true',
        help='Skip verification step after download'
    )

    args = parser.parse_args()

    # Check if downloader exists
    check_downloader_exists()

    # Parse input
    video_ids = None
    participant_ids = None

    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(',')]
    elif args.participants:
        participant_ids = [p.strip() for p in args.participants.split(',')]
    elif args.from_annotations:
        video_ids = get_video_ids_from_annotations(args.from_annotations)

    # Download videos
    success = download_videos(
        video_ids=video_ids,
        participant_ids=participant_ids,
        output_dir=args.output_dir
    )

    if not success:
        sys.exit(1)

    # Verify downloads
    if not args.skip_verification and video_ids:
        found, missing = verify_downloads(video_ids, args.output_dir)

        if found:
            create_download_manifest(found)

        if missing:
            print()
            print("⚠️  WARNING: Some videos failed to download")
            print("   You may need to re-run the script or check your internet connection")
            sys.exit(1)

    print()
    print("🎉 All done!")
    print(f"   Videos saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
