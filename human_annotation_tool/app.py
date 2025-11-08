"""
Flask web application for human annotation of video clip pairs.

Annotators compare two clips side-by-side and select which one is more useful
for training robot manipulation policies.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import random
from pathlib import Path
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Configuration
CLIPS_DIR = Path('../epic_prototype/data/mixed_clips')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Load clip metadata
metadata_df = pd.read_csv(CLIPS_DIR / 'clip_metadata.csv')
all_clips = metadata_df['clip_id'].tolist()

print(f"Loaded {len(all_clips)} clips for annotation")

# Store annotations in memory (will save to file periodically)
annotations = []

def load_existing_annotations():
    """Load any existing annotations from file"""
    global annotations
    annotations_file = RESULTS_DIR / 'annotations.json'
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")

def save_annotations():
    """Save annotations to file"""
    annotations_file = RESULTS_DIR / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    # Also save as CSV
    if annotations:
        df = pd.DataFrame(annotations)
        df.to_csv(RESULTS_DIR / 'annotations.csv', index=False)

def generate_pair():
    """Generate a random pair of clips to compare"""
    pair = random.sample(all_clips, 2)
    return {
        'clip_a_id': pair[0],
        'clip_b_id': pair[1],
        'pair_id': f"{pair[0]}_vs_{pair[1]}"
    }

def get_clip_info(clip_id):
    """Get metadata for a clip"""
    clip_row = metadata_df[metadata_df['clip_id'] == clip_id].iloc[0]
    clip_type = clip_row['type']

    if clip_type == 'annotated':
        video_path = f"annotated/{clip_row['filename']}"
        label = clip_row['narration']
    else:
        video_path = f"random/{clip_row['filename']}"
        label = f"Random clip at {clip_row.get('timestamp', 'unknown')}"

    return {
        'id': clip_id,
        'path': video_path,
        'label': label,
        'type': clip_type
    }

@app.route('/')
def index():
    """Main annotation interface"""
    return render_template('index.html')

@app.route('/api/get_pair')
def get_pair():
    """API endpoint to get a new clip pair"""
    pair = generate_pair()
    clip_a = get_clip_info(pair['clip_a_id'])
    clip_b = get_clip_info(pair['clip_b_id'])

    return jsonify({
        'pair_id': pair['pair_id'],
        'clip_a': clip_a,
        'clip_b': clip_b,
        'total_annotations': len(annotations)
    })

@app.route('/api/submit_annotation', methods=['POST'])
def submit_annotation():
    """API endpoint to submit an annotation"""
    data = request.json

    annotation = {
        'pair_id': data['pair_id'],
        'clip_a_id': data['clip_a_id'],
        'clip_b_id': data['clip_b_id'],
        'preference': data['preference'],  # 'A', 'B', or 'tie'
        'timestamp': datetime.now().isoformat(),
        'annotator_id': data.get('annotator_id', 'anonymous')
    }

    annotations.append(annotation)
    save_annotations()

    return jsonify({
        'success': True,
        'total_annotations': len(annotations)
    })

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    # Serve from epic_prototype/data/mixed_clips
    return send_from_directory(CLIPS_DIR, filename)

@app.route('/stats')
def stats():
    """Show annotation statistics"""
    if not annotations:
        return render_template('stats.html', stats={})

    df = pd.DataFrame(annotations)

    stats = {
        'total_annotations': len(annotations),
        'preference_distribution': df['preference'].value_counts().to_dict(),
        'annotations_per_annotator': df['annotator_id'].value_counts().to_dict(),
        'unique_pairs': df['pair_id'].nunique(),
        'coverage': f"{df['pair_id'].nunique()} / {len(all_clips) * (len(all_clips) - 1) / 2:.0f} possible pairs"
    }

    return render_template('stats.html', stats=stats)

if __name__ == '__main__':
    load_existing_annotations()
    print("\nStarting annotation server...")
    print("Access the tool at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
