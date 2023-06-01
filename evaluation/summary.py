import numpy as np
from tools.knapsack import knapsack

# from evaluation.metrics import evaluate_summary


def generate_summary(
    shots_bounds, scores_within_shots, num_frames_list, positions, fps: int = 30
):
    all_summaries = []
    for video_index in range(len(scores_within_shots)):
        shot_bound = shots_bounds[video_index]
        frame_init_scores = scores_within_shots[video_index]
        n_frames = num_frames_list[video_index]
        positions = positions[video_index]

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        shot_importance_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_importance_scores.append(np.mean(frame_scores[shot[0] : shot[1] + 1]))

        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapsack(
            final_max_length,
            shot_lengths,
            shot_importance_scores,
        )

        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        # the selected frames are set to 1 and the rest are set to 0
        for shot in selected:
            summary[shot_bound[shot][0] : shot_bound[shot][1] + 1] = 1

        # Generate timestamps for selected keyshots
        # It calculates the start time and end time of the keyshot using the frame rate.
        # The start time is obtained by dividing the start frame index by the frame rate,
        # and the end time is obtained by dividing the end frame index by the frame rate.
        timestamps = []
        for shot in selected:
            start_frame = shot_bound[shot][0]
            end_frame = shot_bound[shot][1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            timestamps.append({"start_time": start_time, "end_time": end_time})

        all_summaries.append(summary)

    return all_summaries, timestamps
