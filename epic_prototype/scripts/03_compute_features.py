"""
Compute all features that will be used in the reward model:
- Visual quality (blur, exposure)
- Motion metrics (variance, jerk)
- Hand detection
- CLIP embeddings
- Robot-usefulness features (causal coupling, interaction, etc.)

This validates our feature extraction pipeline end-to-end and can also
batch-process directories of clips for downstream analysis.
"""

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

try:
    import mediapipe as mp
except ImportError:
    mp = None
    warnings.warn(
        "MediaPipe is not available. Hand metrics will default to 0 and "
        "has_hands will be False."
    )

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    warnings.warn(
        "Ultralytics YOLOv8 is not available. Object metrics will default to 0 "
        "and has_object will be False."
    )


ROBOT_SUMMARY_COLUMNS = [
    "causal_hand_object_coupling",
    "state_change_score",
    "interaction_ratio",
    "mean_jerk",
    "smoothness",
    "trajectory_length_norm",
    "affordance_match_score",
    "has_hands",
    "has_object",
]

AFFORDANCE_PROMPTS = [
    "graspable region",
    "handle",
    "hinge",
    "spout",
    "cutting edge",
]


def load_models():
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Pre-compute affordance text features once.
    with torch.no_grad():
        text_inputs = processor(
            text=AFFORDANCE_PROMPTS, return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = clip.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    # MediaPipe hands
    if mp is not None:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
    else:
        hands_detector = None

    # YOLO object detector
    if YOLO is not None:
        try:
            yolo_model = YOLO("yolov8n.pt")
        except Exception as exc:
            print(f"Warning: Could not load YOLOv8n weights ({exc}).")
            yolo_model = None
    else:
        yolo_model = None

    print(f"Using device: {device}")
    return {
        "device": device,
        "clip_model": clip,
        "clip_processor": processor,
        "text_features": text_features,
        "hands_detector": hands_detector,
        "yolo_model": yolo_model,
    }


MODELS = load_models()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute clip features or batch process a directory."
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        help="Directory containing .mp4 clips to process.",
    )
    return parser.parse_args()


def compute_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_exposure_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / max(hist.sum(), 1e-8)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return flow


def compute_visual_quality(frames, verbose=False):
    if verbose:
        print("\n[1/5] Computing visual quality features...")
    blur_scores = [compute_blur_score(f) for f in frames]
    exposure_scores = [compute_exposure_score(f) for f in frames]
    blur_score = float(np.mean(blur_scores))
    exposure_score = float(np.mean(exposure_scores))
    if verbose:
        print(f"  Blur score (sharpness): {blur_score:.2f}")
        print(f"  Exposure score (lighting): {exposure_score:.2f}")
    return blur_score, exposure_score


def compute_motion_features(frames, verbose=False):
    if verbose:
        print("\n[2/5] Computing motion features...")
    if len(frames) < 2:
        return 0.0, 0.0
    flow_magnitudes = []
    for i in range(len(frames) - 1):
        flow = compute_optical_flow(frames[i], frames[i + 1])
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(magnitude.mean())
    motion_var = float(np.std(flow_magnitudes)) if flow_magnitudes else 0.0
    if len(flow_magnitudes) >= 3:
        jerks = [
            flow_magnitudes[i + 1]
            - 2 * flow_magnitudes[i]
            + flow_magnitudes[i - 1]
            for i in range(1, len(flow_magnitudes) - 1)
        ]
        jerk_score = float(np.mean(np.abs(jerks)))
    else:
        jerk_score = 0.0
    if verbose:
        print(f"  Motion variance: {motion_var:.4f}")
        print(f"  Jerk score: {jerk_score:.4f}")
    return motion_var, jerk_score


def detect_hands_info(frames, verbose=False):
    detector = MODELS["hands_detector"]
    hand_info = []
    if detector is None:
        if verbose:
            print("\n[3/5] Detecting hands... (MediaPipe unavailable)")
        return [
            {"visible": False, "bbox": None, "centroid": None}
            for _ in frames
        ], 0.0

    if verbose:
        print("\n[3/5] Detecting hands...")
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            coords = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    coords.append((lm.x * w, lm.y * h))
            coords = np.array(coords)
            x1 = float(np.clip(coords[:, 0].min(), 0, w))
            y1 = float(np.clip(coords[:, 1].min(), 0, h))
            x2 = float(np.clip(coords[:, 0].max(), 0, w))
            y2 = float(np.clip(coords[:, 1].max(), 0, h))
            centroid = tuple(coords.mean(axis=0))
            hand_info.append(
                {
                    "visible": True,
                    "bbox": (x1, y1, x2, y2),
                    "centroid": centroid,
                }
            )
        else:
            hand_info.append({"visible": False, "bbox": None, "centroid": None})
    ratio = sum(1 for info in hand_info if info["visible"]) / max(len(frames), 1)
    if verbose:
        print(f"  Hand visible in {ratio * 100:.1f}% of frames")
    return hand_info, float(ratio)


def detect_object_info(frames):
    model = MODELS["yolo_model"]
    if model is None:
        return [
            {"bbox": None, "centroid": None, "confidence": 0.0}
            for _ in frames
        ], False

    object_info = []
    has_object = False
    for frame in frames:
        try:
            results = model.predict(frame, verbose=False)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Warning: YOLO inference failed ({exc}).")
            model = None
            MODELS["yolo_model"] = None
            return [
                {"bbox": None, "centroid": None, "confidence": 0.0}
                for _ in frames
            ], False

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.detach().cpu().numpy()
            idx = int(np.argmax(confs))
            xyxy = result.boxes.xyxy[idx].detach().cpu().numpy()
            bbox = tuple(float(v) for v in xyxy)
            centroid = (
                float((xyxy[0] + xyxy[2]) / 2.0),
                float((xyxy[1] + xyxy[3]) / 2.0),
            )
            confidence = float(confs[idx])
            object_info.append(
                {"bbox": bbox, "centroid": centroid, "confidence": confidence}
            )
            has_object = True
        else:
            object_info.append({"bbox": None, "centroid": None, "confidence": 0.0})
    return object_info, has_object


def sample_frames(frames, num_samples=8):
    if len(frames) == 0:
        return []
    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    sampled = [frames[i] for i in indices]
    return sampled


def encode_images(pil_images, normalize=False):
    if not pil_images:
        return None
    inputs = MODELS["clip_processor"](images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(MODELS["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        feats = MODELS["clip_model"].get_image_features(**inputs)
    if normalize:
        feats = F.normalize(feats, dim=-1)
    return feats


def compute_clip_embedding(frames, verbose=False):
    if verbose:
        print("\n[4/5] Computing CLIP embeddings...")
    sampled_frames = sample_frames(frames, num_samples=8)
    pil_images = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in sampled_frames
    ]
    feats = encode_images(pil_images, normalize=False)
    if feats is None:
        embedding = np.zeros(512, dtype=float)
    else:
        embedding = feats.mean(dim=0).cpu().numpy()
    if verbose:
        print(f"  CLIP embedding shape: {embedding.shape}")
        print(f"  CLIP embedding mean: {embedding.mean():.4f}")
    return embedding


def compute_composite_score(
    blur_score, exposure_score, motion_var, jerk_score, hand_visible_ratio, verbose=False
):
    if verbose:
        print("\n[5/5] Computing composite score...")
    normalized_blur = min(blur_score / 1000.0, 1.0)
    normalized_exposure = min(exposure_score / 8.0, 1.0)
    normalized_motion = min(motion_var / 10.0, 1.0)
    normalized_jerk = max(1.0 - jerk_score / 5.0, 0.0)
    normalized_hands = hand_visible_ratio
    composite_score = (
        0.2 * normalized_blur
        + 0.2 * normalized_exposure
        + 0.2 * normalized_motion
        + 0.2 * normalized_hands
        + 0.2 * normalized_jerk
    )
    if verbose:
        print(f"  Composite quality score: {composite_score:.3f}")
    return (
        float(composite_score),
        float(normalized_blur),
        float(normalized_exposure),
        float(normalized_motion),
        float(normalized_jerk),
    )


def interpolate_points(points):
    n = len(points)
    arr = np.full((n, 2), np.nan, dtype=float)
    valid_mask = np.zeros(n, dtype=bool)
    for i, pt in enumerate(points):
        if pt is not None:
            arr[i] = pt
            valid_mask[i] = True
    if not valid_mask.any():
        return arr, False

    idx = np.arange(n)
    for dim in range(2):
        values = arr[:, dim]
        valid_idx = idx[valid_mask]
        valid_vals = values[valid_mask]
        if len(valid_idx) == 1:
            values[:] = valid_vals[0]
        else:
            values[: valid_idx[0]] = valid_vals[0]
            values[valid_idx[-1] + 1 :] = valid_vals[-1]
            missing = ~valid_mask
            values[missing] = np.interp(idx[missing], valid_idx, valid_vals)
        arr[:, dim] = values
    return arr, True


def compute_speed_series(points):
    if len(points) < 2:
        return np.array([])
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return distances


def safe_corr(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    length = min(len(a), len(b))
    a = np.asarray(a[:length])
    b = np.asarray(b[:length])
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    corr_matrix = np.corrcoef(a, b)
    if np.isnan(corr_matrix).any():
        return 0.0
    return float(corr_matrix[0, 1])


def bbox_iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def extract_crop(frame, bbox):
    h, w = frame.shape[:2]
    if bbox is None:
        return frame
    x1 = int(max(0, min(w, math.floor(bbox[0]))))
    y1 = int(max(0, min(h, math.floor(bbox[1]))))
    x2 = int(max(x1 + 1, min(w, math.ceil(bbox[2]))))
    y2 = int(max(y1 + 1, min(h, math.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def compute_state_change_score(frames, object_info):
    if frames is None or len(frames) == 0:
        return 0.0
    start_idx = next(
        (i for i, info in enumerate(object_info) if info["bbox"] is not None),
        None,
    )
    end_idx = next(
        (
            len(object_info) - 1 - i
            for i, info in enumerate(reversed(object_info))
            if info["bbox"] is not None
        ),
        None,
    )
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(frames) - 1

    start_crop = extract_crop(frames[start_idx], object_info[start_idx]["bbox"])
    end_crop = extract_crop(frames[end_idx], object_info[end_idx]["bbox"])
    pil_images = [
        Image.fromarray(cv2.cvtColor(start_crop, cv2.COLOR_BGR2RGB)),
        Image.fromarray(cv2.cvtColor(end_crop, cv2.COLOR_BGR2RGB)),
    ]
    feats = encode_images(pil_images, normalize=True)
    if feats is None or feats.shape[0] < 2:
        return 0.0
    cos_sim = float(torch.sum(feats[0] * feats[1]).item())
    distance = max(0.0, 1.0 - cos_sim)
    return distance


def compute_affordance_match_score(frames, hand_info, object_info):
    if frames is None or len(frames) == 0:
        return 0.0
    # Find frame with maximum IoU.
    best_idx = None
    best_iou = 0.0
    for idx, (h_info, o_info) in enumerate(zip(hand_info, object_info)):
        current_iou = bbox_iou(h_info["bbox"], o_info["bbox"])
        if current_iou > best_iou:
            best_iou = current_iou
            best_idx = idx
    if best_idx is not None and best_iou > 0:
        frame = frames[best_idx]
        bbox = object_info[best_idx]["bbox"]
    else:
        # Fallback to last frame with object, else last frame full image.
        fallback_idx = next(
            (
                len(object_info) - 1 - i
                for i, info in enumerate(reversed(object_info))
                if info["bbox"] is not None
            ),
            None,
        )
        if fallback_idx is None:
            frame = frames[-1]
            bbox = None
        else:
            frame = frames[fallback_idx]
            bbox = object_info[fallback_idx]["bbox"]
    crop = extract_crop(frame, bbox)
    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    feats = encode_images([pil_image], normalize=True)
    if feats is None:
        return 0.0
    text_feats = MODELS["text_features"]
    scores = torch.matmul(feats, text_feats.T)
    return float(scores.max().item())


def compute_robot_usefulness(frames, hand_info, object_info):
    if frames is None or len(frames) == 0:
        return {
            "causal_hand_object_coupling": 0.0,
            "state_change_score": 0.0,
            "interaction_ratio": 0.0,
            "mean_jerk": 0.0,
            "smoothness": 0.0,
            "trajectory_length_norm": 0.0,
            "affordance_match_score": 0.0,
            "has_hands": False,
            "has_object": False,
        }
    has_hands = any(info["visible"] for info in hand_info)
    has_object = any(info["bbox"] is not None for info in object_info)

    hand_points = [info["centroid"] for info in hand_info]
    object_points = [info["centroid"] for info in object_info]
    interp_hand, hand_valid = interpolate_points(hand_points)
    interp_object, object_valid = interpolate_points(object_points)

    hand_speed = compute_speed_series(interp_hand) if hand_valid else np.array([])
    object_speed = (
        compute_speed_series(interp_object) if object_valid else np.array([])
    )

    if len(hand_speed) >= 3:
        mean_jerk = float(
            np.mean(np.abs(np.diff(hand_speed, n=2)))
        )
    else:
        mean_jerk = 0.0

    smoothness = (
        float(1.0 / (1.0 + np.var(hand_speed))) if len(hand_speed) >= 2 else 0.0
    )

    h, w = frames[0].shape[:2]
    frame_diag = math.hypot(w, h) or 1.0
    if hand_valid and len(interp_hand) >= 2:
        path_length = float(np.sum(hand_speed))
        trajectory_length_norm = float(path_length / frame_diag)
    else:
        trajectory_length_norm = 0.0

    io_count = 0
    for h_info, o_info in zip(hand_info, object_info):
        if bbox_iou(h_info["bbox"], o_info["bbox"]) > 0.05:
            io_count += 1
    interaction_ratio = float(io_count / max(len(frames), 1))

    coupling = safe_corr(hand_speed[:-1], object_speed[1:]) - safe_corr(
        object_speed[:-1], hand_speed[1:]
    )

    state_change_score = compute_state_change_score(frames, object_info)
    affordance_match_score = compute_affordance_match_score(
        frames, hand_info, object_info
    )

    return {
        "causal_hand_object_coupling": float(coupling),
        "state_change_score": float(state_change_score),
        "interaction_ratio": float(interaction_ratio),
        "mean_jerk": float(mean_jerk),
        "smoothness": float(smoothness),
        "trajectory_length_norm": float(trajectory_length_norm),
        "affordance_match_score": float(affordance_match_score),
        "has_hands": bool(has_hands),
        "has_object": bool(has_object),
    }


def assemble_feature_dict(
    frames,
    blur_score,
    exposure_score,
    motion_var,
    jerk_score,
    hand_visible_ratio,
    clip_embedding,
    composite_score,
    robot_block,
):
    feature_dict = {
        "visual_quality": {
            "blur_score": float(blur_score),
            "exposure_score": float(exposure_score),
        },
        "motion": {
            "motion_var": float(motion_var),
            "jerk_score": float(jerk_score),
        },
        "hands": {
            "hand_visible_ratio": float(hand_visible_ratio),
        },
        "clip_embedding": {
            "shape": list(clip_embedding.shape),
            "mean": float(clip_embedding.mean()),
            "std": float(clip_embedding.std()),
            "embedding": clip_embedding.tolist(),
        },
        "composite_score": float(composite_score),
    }
    feature_dict["robot_usefulness"] = robot_block
    return feature_dict


def compute_all_features(frames, verbose=False):
    blur_score, exposure_score = compute_visual_quality(frames, verbose=verbose)
    motion_var, jerk_score = compute_motion_features(frames, verbose=verbose)
    hand_info, hand_visible_ratio = detect_hands_info(frames, verbose=verbose)
    object_info, _ = detect_object_info(frames)
    clip_embedding = compute_clip_embedding(frames, verbose=verbose)
    (
        composite_score,
        normalized_blur,
        normalized_exposure,
        normalized_motion,
        normalized_jerk,
    ) = compute_composite_score(
        blur_score,
        exposure_score,
        motion_var,
        jerk_score,
        hand_visible_ratio,
        verbose=verbose,
    )
    robot_block = compute_robot_usefulness(frames, hand_info, object_info)
    features = assemble_feature_dict(
        frames,
        blur_score,
        exposure_score,
        motion_var,
        jerk_score,
        hand_visible_ratio,
        clip_embedding,
        composite_score,
        robot_block,
    )
    summary = {
        "composite_score": composite_score,
        "normalized_blur": normalized_blur,
        "normalized_exposure": normalized_exposure,
        "normalized_motion": normalized_motion,
        "normalized_jerk": normalized_jerk,
        "hand_visible_ratio": hand_visible_ratio,
        "clip_embedding_dim": clip_embedding.shape[0],
        "robot_usefulness": robot_block,
    }
    return features, summary


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames


def process_single_clip():
    frames = np.load("data/test_clip_frames.npy")
    print(f"Loaded {len(frames)} frames")
    features, summary = compute_all_features(frames, verbose=True)

    output_path = Path("outputs/test_clip_features.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(features, f, indent=2)
    print(f"\n✅ Features saved to: {output_path}")

    normalized_blur = summary["normalized_blur"]
    normalized_exposure = summary["normalized_exposure"]
    normalized_motion = summary["normalized_motion"]
    normalized_jerk = summary["normalized_jerk"]
    hand_visible_ratio = summary["hand_visible_ratio"]
    composite_score = summary["composite_score"]
    clip_dim = summary["clip_embedding_dim"]
    robot_block = summary["robot_usefulness"]

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Clip quality: {composite_score:.3f} / 1.0")
    print("")
    print("Visual Quality:")
    print(
        f"  - Sharpness: {'Good' if normalized_blur > 0.5 else 'Poor'} ({normalized_blur:.2f})"
    )
    print(
        f"  - Lighting: {'Good' if normalized_exposure > 0.5 else 'Poor'} ({normalized_exposure:.2f})"
    )
    print("")
    print("Motion Characteristics:")
    print(
        f"  - Dynamics: {'Good' if normalized_motion > 0.3 else 'Static'} ({normalized_motion:.2f})"
    )
    print(
        f"  - Smoothness: {'Good' if normalized_jerk > 0.5 else 'Jerky'} ({normalized_jerk:.2f})"
    )
    print("")
    print(
        f"Hand Visibility: {'Good' if hand_visible_ratio > 0.3 else 'Poor'} ({hand_visible_ratio:.2%})"
    )
    print("")
    print(f"CLIP Embedding: {clip_dim} dimensions extracted")
    print("")
    print(
        "Robot Usefulness:"
        f" interaction_ratio={robot_block['interaction_ratio']:.2f}, "
        f"affordance_match_score={robot_block['affordance_match_score']:.2f}"
    )
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review outputs/test_clip_features.json")
    print("2. Visualize the test clip: data/test_clip.mp4")
    print("3. Once Ego4D access granted, scale to 5000 clips")
    print("4. Then proceed to human annotation (Step 3)")


def process_directory(clips_dir):
    clips_path = Path(clips_dir)
    if not clips_path.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    output_dir = Path("outputs/robot_usefulness")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mp4_files = sorted(clips_path.glob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 clips found in {clips_path}")
        return

    for video_path in mp4_files:
        frames = load_video_frames(video_path)
        if frames is None or len(frames) == 0:
            print(f"Skipping {video_path} (no frames)")
            continue
        features, summary = compute_all_features(frames, verbose=False)
        clip_output = output_dir / f"{video_path.stem}_features.json"
        with clip_output.open("w") as f:
            json.dump(features, f, indent=2)
        robot_block = summary["robot_usefulness"]
        row = {"clip_path": str(video_path)}
        for column in ROBOT_SUMMARY_COLUMNS:
            row[column] = robot_block[column]
        rows.append(row)
        print(f"Processed {video_path.name} -> {clip_output.name}")

    if rows:
        summary_path = output_dir / "summary.csv"
        with summary_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["clip_path"] + ROBOT_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary saved to {summary_path}")


def main():
    args = parse_args()
    if args.clips_dir:
        process_directory(args.clips_dir)
    else:
        process_single_clip()


if __name__ == "__main__":
    main()
