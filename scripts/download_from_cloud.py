"""
Download clips from AWS S3 cloud storage.

This script allows team members to download the shared dataset of video clips
from S3 to their local machine.

Usage:
    # Download all clips
    python scripts/download_from_cloud.py --all

    # Download only annotated clips
    python scripts/download_from_cloud.py --type annotated

    # Download only random clips
    python scripts/download_from_cloud.py --type random

    # Dry run (see what would be downloaded without actually downloading)
    python scripts/download_from_cloud.py --all --dry-run

    # Force re-download (even if files exist locally)
    python scripts/download_from_cloud.py --all --force
"""

import boto3
import argparse
from pathlib import Path
from tqdm import tqdm
import hashlib

# Configuration
BUCKET_NAME = 'data-eval-engine-clips'
CLIPS_DIR = Path('epic_prototype/data/mixed_clips')
S3_PREFIX = 'clips/epic_kitchens/mixed/'

def compute_md5(file_path):
    """Compute MD5 checksum for file integrity verification"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_s3_clips(s3, clip_type='all'):
    """List all clips in S3 bucket"""
    clips = []

    try:
        # List all objects in the bucket under our prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                # Filter by type
                if clip_type == 'annotated' and '/annotated/' not in key:
                    continue
                if clip_type == 'random' and '/random/' not in key:
                    continue

                # Only include .mp4 files
                if key.endswith('.mp4'):
                    clips.append({
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })

    except Exception as e:
        print(f"Error listing S3 objects: {str(e)}")
        return []

    return clips

def download_clips(clip_type='all', dry_run=False, force=False):
    """Download clips from S3"""

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Create clips directory if it doesn't exist
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of clips from S3
    print(f"Fetching clip list from s3://{BUCKET_NAME}/{S3_PREFIX}...")
    clips = get_s3_clips(s3, clip_type)

    if not clips:
        print(f"No clips found in S3 bucket")
        return

    print(f"Found {len(clips)} clips in S3")

    if dry_run:
        print("\n[DRY RUN] - No files will be downloaded\n")

    # Track statistics
    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0

    # Download each clip
    for clip in tqdm(clips, desc="Downloading clips"):
        s3_key = clip['key']
        file_size = clip['size']
        total_size += file_size

        # Determine local path
        # S3 key format: clips/epic_kitchens/mixed/annotated/clip_001.mp4
        # Local path: epic_prototype/data/mixed_clips/annotated/clip_001.mp4
        relative_path = s3_key.replace(S3_PREFIX, '')
        local_path = CLIPS_DIR / relative_path

        # Create directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and has same checksum (unless force flag)
        should_download = True
        if not force and local_path.exists():
            try:
                # Get S3 object metadata
                response = s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                s3_checksum = response.get('Metadata', {}).get('md5')

                # Compute local file checksum
                local_checksum = compute_md5(local_path)

                if s3_checksum and s3_checksum == local_checksum:
                    should_download = False
                    skipped += 1
                    tqdm.write(f"  ⏭️  Skipped {local_path.name} (already exists with same checksum)")
                elif not s3_checksum:
                    # No checksum in metadata, compare file sizes
                    if local_path.stat().st_size == file_size:
                        should_download = False
                        skipped += 1
                        tqdm.write(f"  ⏭️  Skipped {local_path.name} (same size)")
            except Exception as e:
                # If we can't verify, download anyway
                tqdm.write(f"  ⚠️  Could not verify {local_path.name}, will re-download")

        if should_download and not dry_run:
            try:
                s3.download_file(BUCKET_NAME, s3_key, str(local_path))
                downloaded += 1
                tqdm.write(f"  ✅ Downloaded {local_path.name} ({file_size / 1024:.1f} KB)")
            except Exception as e:
                failed += 1
                tqdm.write(f"  ❌ Failed {local_path.name}: {str(e)}")
        elif dry_run:
            tqdm.write(f"  [DRY RUN] Would download {local_path.name} from s3://{BUCKET_NAME}/{s3_key}")

    # Download metadata files
    metadata_files = [
        'clip_metadata.csv',
        'clip_metadata.json',
        'all_features.csv',
        'all_features.json'
    ]

    print("\nDownloading metadata files...")
    for meta_file in metadata_files:
        s3_key = f'metadata/{meta_file}'
        local_path = CLIPS_DIR / meta_file

        if not dry_run:
            try:
                s3.download_file(BUCKET_NAME, s3_key, str(local_path))
                print(f"  ✅ Downloaded {meta_file}")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"  ⚠️  {meta_file} not found in S3 (skipping)")
                else:
                    print(f"  ❌ Failed {meta_file}: {str(e)}")
        else:
            print(f"  [DRY RUN] Would download {meta_file}")

    # Print summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    if not dry_run:
        print(f"✅ Downloaded: {downloaded} clips")
        print(f"⏭️  Skipped:   {skipped} clips (already exist)")
        print(f"❌ Failed:    {failed} clips")
        print(f"📦 Total size: {total_size / (1024*1024):.2f} MB")
        print(f"📁 Local path: {CLIPS_DIR.resolve()}")
    else:
        print(f"Would download {len(clips)} clips ({total_size / (1024*1024):.2f} MB)")
    print(f"🪣 Source: s3://{BUCKET_NAME}/{S3_PREFIX}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Download clips from S3')
    parser.add_argument('--all', action='store_true', help='Download all clips')
    parser.add_argument('--type', choices=['annotated', 'random'], help='Download specific clip type')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without downloading')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist locally')

    args = parser.parse_args()

    # Determine clip type
    if args.all:
        clip_type = 'all'
    elif args.type:
        clip_type = args.type
    else:
        print("Error: Must specify --all or --type")
        parser.print_help()
        return

    # Download clips
    download_clips(clip_type, dry_run=args.dry_run, force=args.force)

if __name__ == '__main__':
    main()
