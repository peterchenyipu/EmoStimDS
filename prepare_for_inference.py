import json
import os

TEST_JSON = "inference.json"
INFERENCE_VIDEOS_DIR = 'top5trailers'

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

prompt_templates = [
    "How might someone feel after watching this movie scene? Rate each emotion from 1 (very low) to 5 (very high):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",
    
    "How would a typical viewer emotionally respond to this video? Please rate each of the following emotions from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Predict the emotional reaction a person might have while watching this clip. Use a 1-5 scale to rate each emotion:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Estimate how strongly each emotion could be felt by someone viewing this clip. Rate each one from 1 (low) to 5 (high):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "How intense might these emotions be for a viewer watching the video? Score from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Imagine someone watching this video. What emotional response would they likely experience? Rate from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Rate how emotionally moving this video might be for an average viewer, using 1 (not at all) to 5 (very strong):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "From a viewer's perspective, how strongly might each of these emotions be felt during this clip? Rate each from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Based on the scene, what might a person feel emotionally? Use the 1-5 scale to score the following:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "How emotionally intense would these feelings be for someone watching the clip? Rate 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "After watching the video, how did you feel? Rate the intensity of each emotion from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Reflecting on the video, how did it make you feel? Use 1 (low) to 5 (high) to rate each emotion:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Please describe your emotional response after watching the clip. Score each emotion from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Imagine experiencing this scene. How would each of these emotions be felt? Rate each from 1 (low) to 5 (high):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Having just seen the video, how strongly do you feel each emotion listed below? Rate from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "How would you personally rate the emotional impact of this video? Score 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "If you were emotionally affected by this scene, how would you score the following emotions from 1 to 5?\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "What emotional impression did the video leave on you? Use a scale from 1 (minimal) to 5 (strong):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Based on your experience of the video, how would you rate each of the following emotions? Rate from 1 to 5:\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}",

    "Considering your own reaction, rate each emotion from 1 (not felt) to 5 (strongly felt):\n\nPlease respond using this format: Emotion Label: Score (1 to 5), separated by commas, all on one line.\nExample response: Joyful / Amused / Happy: 4, Sad / Downhearted / Blue: 2, Fearful / Scared / Afraid: 1\nYour response should include all of the following 16 emotions.\n\n{}"
]

emotion_list_text = "\n".join([f"- {desc}" for desc in full_emotion_label_map.values()])
video_names = os.listdir(INFERENCE_VIDEOS_DIR)

entries = []
for name in video_names:
    video_path = f"{INFERENCE_VIDEOS_DIR}/{name}"    
    for template in prompt_templates:
        prompt = template.format(emotion_list_text)
        response = ""
        entries.append({
            "video": video_path,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": response}
            ]
        })

with open(TEST_JSON, "w") as f:
    json.dump(entries, f, indent=2)

print(f"Saved {len(entries)} test entries to {TEST_JSON}")

