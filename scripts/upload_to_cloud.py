"""
Upload processed clips to AWS S3 cloud storage.

This script allows team members to upload their processed video clips to a shared
S3 bucket so that everyone can access the full dataset.

Usage:
    # Upload all clips from epic_prototype/data/mixed_clips/
    python scripts/upload_to_cloud.py --all

    # Upload only annotated clips
    python scripts/upload_to_cloud.py --type annotated

    # Upload only random clips
    python scripts/upload_to_cloud.py --type random

    # Dry run (see what would be uploaded without actually uploading)
    python scripts/upload_to_cloud.py --all --dry-run
"""

import boto3
import json
import argparse
from pathlib import Path
from datetime import datetime
import hashlib
from tqdm import tqdm
import subprocess

# Configuration
BUCKET_NAME = 'data-eval-engine-clips'
CLIPS_DIR = Path('epic_prototype/data/mixed_clips')
METADATA_FILE = CLIPS_DIR / 'clip_metadata.csv'

def get_git_user():
    """Get current git user name for tracking who uploaded"""
    try:
        result = subprocess.run(
            ['git', 'config', 'user.name'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return 'unknown'

def compute_md5(file_path):
    """Compute MD5 checksum for file integrity verification"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_clips_to_upload(clip_type='all'):
    """Get list of clip files to upload"""
    clips = []

    if clip_type in ['all', 'annotated']:
        annotated_dir = CLIPS_DIR / 'annotated'
        if annotated_dir.exists():
            clips.extend(list(annotated_dir.glob('*.mp4')))

    if clip_type in ['all', 'random']:
        random_dir = CLIPS_DIR / 'random'
        if random_dir.exists():
            clips.extend(list(random_dir.glob('*.mp4')))

    return clips

def upload_clips(clip_type='all', dry_run=False, force=False):
    """Upload clips to S3"""

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Get clips to upload
    clips = get_clips_to_upload(clip_type)

    if not clips:
        print(f"No clips found to upload in {CLIPS_DIR}")
        return

    print(f"Found {len(clips)} clips to upload")
    print(f"Bucket: s3://{BUCKET_NAME}/")
    print(f"Uploader: {get_git_user()}")

    if dry_run:
        print("\n[DRY RUN] - No files will be uploaded\n")

    # Track statistics
    uploaded = 0
    skipped = 0
    failed = 0
    total_size = 0

    # Upload each clip
    for clip_path in tqdm(clips, desc="Uploading clips"):
        # Determine S3 key (path in bucket)
        relative_path = clip_path.relative_to(CLIPS_DIR)
        s3_key = f'clips/epic_kitchens/mixed/{relative_path}'

        # Get file info
        file_size = clip_path.stat().st_size
        total_size += file_size
        checksum = compute_md5(clip_path)

        # Check if file already exists in S3 (unless force flag is set)
        should_upload = True
        if not force and not dry_run:
            try:
                response = s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                existing_checksum = response.get('Metadata', {}).get('md5')

                if existing_checksum == checksum:
                    should_upload = False
                    skipped += 1
                    tqdm.write(f"  ⏭️  Skipped {clip_path.name} (already exists with same checksum)")
            except s3.exceptions.ClientError:
                # File doesn't exist, proceed with upload
                pass

        if should_upload and not dry_run:
            try:
                # Upload with metadata
                s3.upload_file(
                    str(clip_path),
                    BUCKET_NAME,
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'uploaded_by': get_git_user(),
                            'uploaded_at': datetime.now().isoformat(),
                            'md5': checksum,
                            'source': 'epic_kitchens',
                            'clip_type': relative_path.parent.name
                        }
                    }
                )
                uploaded += 1
                tqdm.write(f"  ✅ Uploaded {clip_path.name} ({file_size / 1024:.1f} KB)")
            except Exception as e:
                failed += 1
                tqdm.write(f"  ❌ Failed {clip_path.name}: {str(e)}")
        elif dry_run:
            tqdm.write(f"  [DRY RUN] Would upload {clip_path.name} to s3://{BUCKET_NAME}/{s3_key}")

    # Upload metadata files
    metadata_files = [
        CLIPS_DIR / 'clip_metadata.csv',
        CLIPS_DIR / 'clip_metadata.json',
        CLIPS_DIR / 'all_features.csv',
        CLIPS_DIR / 'all_features.json'
    ]

    print("\nUploading metadata files...")
    for meta_file in metadata_files:
        if meta_file.exists():
            s3_key = f'metadata/{meta_file.name}'
            if not dry_run:
                try:
                    s3.upload_file(str(meta_file), BUCKET_NAME, s3_key)
                    print(f"  ✅ Uploaded {meta_file.name}")
                except Exception as e:
                    print(f"  ❌ Failed {meta_file.name}: {str(e)}")
            else:
                print(f"  [DRY RUN] Would upload {meta_file.name}")

    # Print summary
    print("\n" + "="*60)
    print("Upload Summary")
    print("="*60)
    if not dry_run:
        print(f"✅ Uploaded: {uploaded} clips")
        print(f"⏭️  Skipped:  {skipped} clips (already exist)")
        print(f"❌ Failed:   {failed} clips")
        print(f"📦 Total size: {total_size / (1024*1024):.2f} MB")
    else:
        print(f"Would upload {len(clips)} clips ({total_size / (1024*1024):.2f} MB)")
    print(f"🪣 Bucket: s3://{BUCKET_NAME}/clips/epic_kitchens/mixed/")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Upload clips to S3')
    parser.add_argument('--all', action='store_true', help='Upload all clips')
    parser.add_argument('--type', choices=['annotated', 'random'], help='Upload specific clip type')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without uploading')
    parser.add_argument('--force', action='store_true', help='Force upload even if file exists in S3')

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

    # Validate clips directory exists
    if not CLIPS_DIR.exists():
        print(f"Error: Clips directory not found at {CLIPS_DIR}")
        print("Have you run the clip extraction scripts?")
        return

    # Upload clips
    upload_clips(clip_type, dry_run=args.dry_run, force=args.force)

if __name__ == '__main__':
    main()
