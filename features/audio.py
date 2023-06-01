import torch
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

vggish = torch.hub.load("harritaylor/torchvggish", "vggish")
# vggish.eval()


def get_audio_embeddings(video_path: str, model=None) -> list:
    embeddings_list = []

    video = VideoFileClip(video_path)
    audio = video.audio

    audio_bytes = audio.to_soundarray(fps=22000).astype(np.int16)
    audio_segment = AudioSegment(
        audio_bytes.tobytes(),
        frame_rate=22000,
        sample_width=2,  # 16 bit audio has a sample width of 2 bytes
        channels=1,
    )
    audio_array = np.array(audio_segment.get_array_of_samples())

    # Split audio array into 1 second segments with 0.5 second overlap
    step_size = int(22000 * 0.5)
    window_size = int(22000 * 1)
    segments = []
    for start in range(0, len(audio_array) - window_size, step_size):
        segments.append(audio_array[start : start + window_size])

    for segment in segments:
        input_data = model.forward(segment)
        embeddings = model.postprocess(input_data)
        embeddings_list.append(embeddings)

    return embeddings_list


if __name__ == "__main__":
    video_path = "../tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/sample_video.mp4"

    embeddings = get_audio_embeddings(video_path, vggish)
    print(embeddings)
