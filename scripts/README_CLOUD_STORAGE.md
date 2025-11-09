# Cloud Storage Setup - Team Workflow

This guide explains how to use AWS S3 for sharing video clips across the team.

## One-Time Setup (For Each Team Member)

### 1. Install AWS CLI

```bash
# macOS
brew install awscli

# Or download from: https://aws.amazon.com/cli/
```

### 2. Configure AWS Credentials

Ask the team lead for the AWS Access Key ID and Secret Access Key, then run:

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: `[provided by team lead]`
- **AWS Secret Access Key**: `[provided by team lead]`
- **Default region**: `us-east-2`
- **Default output format**: `json` (or just press Enter)

### 3. Verify Setup

```bash
aws s3 ls
```

You should see: `data-eval-engine-clips`

---

## Usage

### Upload Clips to Cloud

After processing video clips locally, upload them to the shared S3 bucket:

```bash
# Upload all clips (annotated + random)
python scripts/upload_to_cloud.py --all

# Upload only annotated clips
python scripts/upload_to_cloud.py --type annotated

# Upload only random clips
python scripts/upload_to_cloud.py --type random

# Dry run (see what would be uploaded without uploading)
python scripts/upload_to_cloud.py --all --dry-run

# Force re-upload (even if files exist)
python scripts/upload_to_cloud.py --all --force
```

**What gets uploaded:**
- All `.mp4` files from `epic_prototype/data/mixed_clips/annotated/` and `/random/`
- Metadata files: `clip_metadata.csv`, `all_features.csv`, etc.
- Each file includes metadata: uploader name, timestamp, MD5 checksum

**Features:**
- ✅ Checksums prevent duplicate uploads (saves bandwidth)
- ✅ Tracks who uploaded each file (via git username)
- ✅ Progress bars for large uploads
- ✅ Auto-retry on failure

---

### Download Clips from Cloud

When cloning the repo or wanting to sync with latest clips:

```bash
# Download all clips
python scripts/download_from_cloud.py --all

# Download only annotated clips
python scripts/download_from_cloud.py --type annotated

# Download only random clips
python scripts/download_from_cloud.py --type random

# Dry run (see what would be downloaded)
python scripts/download_from_cloud.py --all --dry-run

# Force re-download (even if files exist locally)
python scripts/download_from_cloud.py --all --force
```

**What gets downloaded:**
- All clips from S3 to `epic_prototype/data/mixed_clips/`
- Metadata files
- Only downloads files that don't exist locally or have changed

**Features:**
- ✅ Checksums skip unchanged files (saves bandwidth)
- ✅ Creates directory structure automatically
- ✅ Progress bars for large downloads
- ✅ Verifies file integrity after download

---

## Common Workflows

### New Team Member Joins

```bash
# 1. Clone the repo
git clone https://github.com/yubo-ruan/data_eval_engine.git
cd data_eval_engine

# 2. Configure AWS (one-time)
aws configure
# [enter credentials provided by team lead]

# 3. Download all existing clips
python scripts/download_from_cloud.py --all

# 4. Start working!
```

### Process New Videos and Share

```bash
# 1. Run your extraction scripts
python epic_prototype/scripts/05_extract_mixed_clips.py
python epic_prototype/scripts/06_compute_all_features.py

# 2. Upload new clips to cloud
python scripts/upload_to_cloud.py --all

# 3. Commit code changes (but not video files!)
git add .
git commit -m "Add new clips from video XYZ"
git push
```

### Sync with Latest Clips

```bash
# Pull latest code
git pull

# Download any new clips uploaded by teammates
python scripts/download_from_cloud.py --all
# (Will only download new/changed files)
```

---

## S3 Bucket Structure

```
s3://data-eval-engine-clips/
├── clips/
│   └── epic_kitchens/
│       └── mixed/
│           ├── annotated/
│           │   ├── annotated_000_put_lid_onto_container.mp4
│           │   ├── annotated_001_turn_off_tap.mp4
│           │   └── ...
│           └── random/
│               ├── random_000.mp4
│               ├── random_001.mp4
│               └── ...
└── metadata/
    ├── clip_metadata.csv
    ├── clip_metadata.json
    ├── all_features.csv
    └── all_features.json
```

---

## Troubleshooting

### "SignatureDoesNotMatch" error
- **Cause**: Incorrect AWS credentials
- **Fix**:
  1. Go to AWS IAM Console → Users → your user → Security credentials
  2. Delete old access key
  3. Create new access key
  4. Run `aws configure` again with new keys

### "Access Denied" error
- **Cause**: IAM user doesn't have S3 permissions
- **Fix**: Team lead needs to attach `AmazonS3FullAccess` policy to your IAM user

### Files uploading slowly
- This is normal for large video files over slower connections
- Consider uploading only the clip type you worked on (`--type annotated`)

### Want to see what's in S3 without downloading
```bash
aws s3 ls s3://data-eval-engine-clips/clips/epic_kitchens/mixed/ --recursive --human-readable
```

---

## Cost Information

**Current storage cost:**
- 20 clips @ ~8.9 MB total
- Cost: ~$0.20/month (negligible)

**For 1000 clips @ 500 MB total:**
- Cost: ~$1-2/month

**Data transfer:**
- Upload: Free
- Download: First 100 GB/month free

**Bottom line:** Very cheap for our use case!

---

## Security Notes

- ✅ S3 bucket is **private** - only accessible with valid AWS credentials
- ✅ Credentials should be shared securely (1Password, LastPass, etc.)
- ✅ Each file includes metadata tracking who uploaded it
- ⚠️ Do NOT commit AWS credentials to git
- ⚠️ Do NOT make the S3 bucket public

---

## Future Improvements

See [FUTURE_IMPROVEMENTS.md](../FUTURE_IMPROVEMENTS.md) for planned enhancements:
- Support for multiple video sources (Ego4D, GoPro, etc.)
- Unified metadata across sources
- Cloud-based video processing pipeline
- Automated clip extraction on upload
