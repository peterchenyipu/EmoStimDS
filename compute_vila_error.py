import json
import re
import numpy as np
from collections import defaultdict
import argparse

EMOTION_KEYS = [
    "Interested / Concentrated / Alert",
    "Fearful / Scared / Afraid",
    "Anxious / Tense / Nervous",
    "Moved",
    "Angry / Irritated / Mad",
    "Ashamed / Embarrassed",
    "Warm-hearted / Gleeful / Elated",
    "Joyful / Amused / Happy",
    "Sad / Downhearted / Blue",
    "Satisfied / Pleased",
    "Surprised / Amazed / Astonished",
    "Loving / Affectionate / Friendly",
    "Guilty / Remorseful",
    "Disgusted / Turned off / Repulsed",
    "Disdainful / Scornful / Contemptuous",
    "Calm / Serene / Relaxed"
]

def normalize_video_name(name):
    return name.split("/")[-1]

def parse_emotion_string(text):
    emotion_dict = {}
    parts = text.split(",")
    for part in parts:
        match = re.match(r"([^:]+?):\s*([0-9.]+)", part.strip())
        if match:
            label, score = match.groups()
            label = label.strip()
            if label in EMOTION_KEYS:
                emotion_dict[label] = float(score)
    return emotion_dict

def compute_error(pred, gt, metric):
    if metric == "mse":
        return (pred - gt) ** 2
    elif metric == "mae":
        return abs(pred - gt)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def main(pred_path, gt_path, out_path, metric):
    with open(pred_path, "r") as f:
        predictions_data = json.load(f)

    with open(gt_path, "r") as f:
        ground_truth_data = json.load(f)

    # Extract ground truth
    ground_truth = {}
    for item in ground_truth_data:
        video = normalize_video_name(item["video"])
        scores = []
        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                parsed = parse_emotion_string(conv["value"])
                if len(parsed) == 16:
                    scores.append(parsed)
        if scores:
            avg = {k: np.mean([s[k] for s in scores]) for k in EMOTION_KEYS}
            ground_truth[video] = avg

    # Compute errors
    error_per_emotion = defaultdict(list)
    error_overall = defaultdict(list)

    for model, videos in predictions_data.items():
        for video, predictions in videos.items():
            video_name = normalize_video_name(video)
            if video_name not in ground_truth:
                continue

            pred_scores = []
            for entry in predictions:
                parsed = parse_emotion_string(entry["prediction"])
                if len(parsed) == 16:
                    pred_scores.append(parsed)

            if not pred_scores:
                continue

            avg_pred = {k: np.mean([p[k] for p in pred_scores]) for k in EMOTION_KEYS}
            gt = ground_truth[video_name]

            for emotion in EMOTION_KEYS:
                err = compute_error(avg_pred[emotion], gt[emotion], metric)
                error_per_emotion[(model, emotion)].append(err)

            overall = np.mean([compute_error(avg_pred[k], gt[k], metric) for k in EMOTION_KEYS])
            error_overall[model].append(overall)

    # try a baseline guess of 3
    baseline_model = "baseline"
    for video in ground_truth:
        pred_scores = {emotion: 3.0 for emotion in EMOTION_KEYS}
        gt = ground_truth[video]
        overall = np.mean([compute_error(pred_scores[k], gt[k], metric) for k in EMOTION_KEYS])
        error_overall[baseline_model].append(overall)

        for emotion in EMOTION_KEYS:
            err = compute_error(pred_scores[emotion], gt[emotion], metric)
            error_per_emotion[(baseline_model, emotion)].append(err)

    # try a random guess
    baseline_model2 = 'random_baseline'
    for video in ground_truth:
        pred_scores = {emotion: np.random.uniform(1.0, 5.0) for emotion in EMOTION_KEYS}
        gt = ground_truth[video]
        overall = np.mean([compute_error(pred_scores[k], gt[k], metric) for k in EMOTION_KEYS])
        error_overall[baseline_model2].append(overall)

        for emotion in EMOTION_KEYS:
            err = compute_error(pred_scores[emotion], gt[emotion], metric)
            error_per_emotion[(baseline_model2, emotion)].append(err)


    # Aggregate
    final_results = {
        f"{metric}_overall": {
            model: float(np.mean(errors)) for model, errors in error_overall.items()
        },
        f"{metric}_per_emotion": {
            model: {
                emotion: float(np.mean(error_per_emotion[(model, emotion)]))
                for emotion in EMOTION_KEYS
            }
            for model in error_overall.keys()
        }
    }

    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"{metric.upper()} results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--gt", type=str, required=True, help="Path to test set JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to save result JSON")
    parser.add_argument("--metric", type=str, choices=["mse", "mae"], default="mse", help="Error metric to compute")

    args = parser.parse_args()
    main(args.pred, args.gt, args.out, args.metric)
