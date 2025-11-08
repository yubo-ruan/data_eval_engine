# Human Annotation Tool

A web-based interface for collecting pairwise comparisons of video clips to train a reward model for robot manipulation.

## Purpose

Collect human judgments on which video clips are more useful for training robot manipulation policies. Annotators compare pairs of clips and select which one would be more valuable for learning manipulation tasks.

## Directory Structure

```
human_annotation_tool/
├── static/           # CSS, JS, and client-side assets
├── templates/        # HTML templates for the web interface
├── data/             # Symlink or copy of clips from epic_prototype/data/mixed_clips
├── results/          # Annotation results and analysis
├── app.py            # Flask web server
└── README.md
```

## Features to Implement

- [ ] Flask web server to serve clips and collect responses
- [ ] Side-by-side video comparison interface
- [ ] Randomized clip pair generation
- [ ] Progress tracking
- [ ] Export annotations to CSV/JSON
- [ ] Basic statistics and agreement metrics

## Annotation Format

Each annotation will contain:
- `pair_id`: Unique identifier for the comparison
- `clip_a_id`: ID of first clip
- `clip_b_id`: ID of second clip
- `preference`: Which clip was chosen (A, B, or tie)
- `timestamp`: When annotation was made
- `annotator_id`: Anonymous identifier for annotator

## Next Steps

1. Set up Flask app with basic routing
2. Create video comparison UI
3. Implement pair generation logic
4. Add annotation storage
5. Build results dashboard
