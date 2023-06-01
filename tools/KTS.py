import cv2
import numpy as np


# cpd_auto.py
def compute_msd(frame1, frame2):
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)
    squared_diff = np.square(diff)
    msd = np.mean(squared_diff)
    return msd


def detect_shot_boundaries(video_path: str, kernel_size: int, threshold: int):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame and convert it to grayscale
    _, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    boundaries = []

    for i in range(1, frame_count):
        _, curr_frame = cap.read()
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if i % kernel_size == 0:
            # Compute the mean squared difference (MSD) between frames within the sliding window
            msd = compute_msd(prev_frame, curr_frame)

            if msd > threshold:
                # Shot boundary detected at the end of the sliding window
                boundaries.append(i)

        # Slide the window by updating the previous frame
        prev_frame = curr_frame

    cap.release()

    return boundaries


if __name__ == "__main__":
    video_path = "../tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/sTEELN-vY30.mp4"
    kernel_size = 5
    threshold = 1000

    boundaries = detect_shot_boundaries(video_path, kernel_size, threshold)
    print("Shot boundaries detected at frames:", boundaries)
