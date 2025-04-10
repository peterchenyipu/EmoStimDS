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


def main(pred_path, out_path):
    with open(pred_path, "r") as f:
        predictions_data = json.load(f)

    videos_results = defaultdict(dict)
    videos = predictions_data['nvila-15b-sft2ep']
    for video, predictions in videos.items():
        video_name = normalize_video_name(video)
        video_name = video_name.split(".")[0]
        pred_scores = []
        for entry in predictions:
            parsed = parse_emotion_string(entry["prediction"])
            if len(parsed) == 16:
                pred_scores.append(parsed)

        avg_pred = {k: np.mean([p[k] for p in pred_scores]) for k in EMOTION_KEYS}

        for emotion in EMOTION_KEYS:
            videos_results[video_name][emotion] = avg_pred[emotion]

    print('total videos:', len(videos_results))

    with open(out_path, "w") as f:
        json.dump(videos_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to save result JSON")

    args = parser.parse_args()
    main(args.pred, args.out)
