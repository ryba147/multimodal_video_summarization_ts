import os
import whisper

model = whisper.load_model("base")

video_folder = "../tvsum50_ver_1_1/ydata-tvsum50-v1_1/video"
transcripts_folder = "../transcripts/TVSum"
os.makedirs(transcripts_folder, exist_ok=True)

# video_folder = os.path.join(config.summe_base_path, "video/")
videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]


for video in videos:
    video_path = os.path.join(video_folder, video)
    video_name = os.path.splitext(video)[0]

    result = model.transcribe(video_path, language="en", verbose=True)
    transcription = result["text"].strip()

    output_file_path = os.path.join(transcripts_folder, f"{video_name}.txt")
    with open(output_file_path, "w") as f:
        f.write(transcription)

    print("Transcription saved for", video_name)

print("Transcriptions saved in", transcripts_folder)
