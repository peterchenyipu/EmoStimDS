import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load JSON ===
with open("vila_mae_results.json", "r") as f:
    data = json.load(f)

# === Rename and filter models ===
exclude_model = "NVILA-15B"
rename_map = {
    "nvila-15b-sft": "NVILA-Lite-15B-Video (1 epoch)",
    "nvila-15b-sft2ep": "NVILA-Lite-15B-Video (2 epochs)",
    "nvila-15b-sft3ep": "NVILA-Lite-15B-Video (3 epochs)",
    "baseline": "Baseline (always 3)",
    "random_baseline": "Random Baseline (1–5)"
}
compact_label_map = {
    "NVILA-Lite-15B-Video (1 epoch)": "SFT-1ep",
    "NVILA-Lite-15B-Video (2 epochs)": "SFT-2ep",
    "NVILA-Lite-15B-Video (3 epochs)": "SFT-3ep",
    "NVILA-Lite-15B-Video": "Pretrained",
    "Baseline (always 3)": "Baseline",
    "Random Baseline (1–5)": "Random",
}

# === Overall MAE ===
mae_overall = data["mae_overall"]
df_overall = pd.DataFrame(list(mae_overall.items()), columns=["Model", "MAE"])
df_overall = df_overall[df_overall["Model"] != exclude_model]
df_overall["Model"] = df_overall["Model"].replace(rename_map).replace(compact_label_map)

# === Per-emotion MAE ===
mae_per_emotion = data["mae_per_emotion"]
records = []
for model_name, emotions in mae_per_emotion.items():
    if model_name == exclude_model:
        continue
    for emotion, mae in emotions.items():
        records.append({
            "Model": rename_map.get(model_name, model_name),
            "Emotion": emotion,
            "MAE": mae
        })
df_emotion = pd.DataFrame(records)
df_emotion["Model"] = df_emotion["Model"].replace(compact_label_map)

# === Shorten emotion labels ===
full_emotion_label_map = {
    'Interest': 'Interested / Concentrated / Alert',
    'Fear': 'Fearful / Scared / Afraid',
    'Anxious': 'Anxious / Tense / Nervous',
    'Moved': 'Moved',
    'Anger': 'Angry / Irritated / Mad',
    'Ashamed': 'Ashamed / Embarrassed',
    'Warm-hearted': 'Warm-hearted / Gleeful / Elated',
    'Joy': 'Joyful / Amused / Happy',
    'Sad': 'Sad / Downhearted / Blue',
    'Satisfied': 'Satisfied / Pleased',
    'Surprise': 'Surprised / Amazed / Astonished',
    'Love': 'Loving / Affectionate / Friendly',
    'Guilt': 'Guilty / Remorseful',
    'Disgust': 'Disgusted / Turned off / Repulsed',
    'Disdainful': 'Disdainful / Scornful / Contemptuous',
    'Calm': 'Calm / Serene / Relaxed',
}
reverse_emotion_map = {v: k for k, v in full_emotion_label_map.items()}
df_emotion["Emotion"] = df_emotion["Emotion"].replace(reverse_emotion_map)

# === Plot Setup ===
model_order = df_overall["Model"].tolist()
emotion_order = df_emotion.groupby("Emotion")["MAE"].mean().sort_values().index.tolist()

sns.set(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("tab10", n_colors=len(model_order))
color_dict = dict(zip(model_order, palette))

# === 1. Overall MAE (vertical) ===
plt.figure(figsize=(8, 5))
sns.barplot(data=df_overall, x="Model", y="MAE", palette=color_dict, order=model_order)
plt.xticks(rotation=30, ha="right")
plt.title("Overall MAE by Model", fontsize=14)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
plt.xlabel("")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("overall_mae_by_model_compact.pdf", format="pdf")
# also save as svg
plt.savefig("overall_mae_by_model_compact.svg", format="svg", transparent=True, facecolor='none')
plt.close()

# === 2. Per-Emotion MAE Boxplot (horizontal) ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_emotion, y="Model", x="MAE", palette=color_dict, order=model_order)
plt.title("Per-Emotion MAE Distribution", fontsize=14)
plt.xlabel("Mean Absolute Error (MAE)", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("boxplot_per_emotion_mae_horizontal.pdf", format="pdf")
# also save as svg
plt.savefig("boxplot_per_emotion_mae_horizontal.svg", format="svg", transparent=True, facecolor='none')
plt.close()

# === 3. Per-Emotion MAE per Emotion Category (horizontal barplot with legend) ===
plt.figure(figsize=(12, 10))
sns.barplot(
    data=df_emotion,
    x="MAE",
    y="Emotion",
    hue="Model",
    palette=color_dict,
    order=emotion_order,
    hue_order=model_order
)
plt.title("Per-Emotion MAE by Model", fontsize=14)
plt.xlabel("Mean Absolute Error (MAE)", fontsize=12)
plt.ylabel("Emotion", fontsize=12)
plt.legend(title="Model", loc='upper right', fontsize=10)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("per_emotion_mae_compact_legend.pdf", format="pdf")
# also save as svg
plt.savefig("per_emotion_mae_compact_legend.svg", format="svg", transparent=True, facecolor='none')
plt.close()
