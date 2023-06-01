import numpy as np


def evaluate_summary(generated_summary, ground_truth_summary, mode: str):
    max_len = max(len(generated_summary), ground_truth_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[: len(generated_summary)] = generated_summary

    f_scores = []
    for user in range(ground_truth_summary.shape[0]):
        G[: ground_truth_summary.shape[1]] = ground_truth_summary[user]
        overlapped = S & G

        precision = np.sum(overlapped) / np.sum(S)
        print(precision)
        recall = np.sum(overlapped) / np.sum(G)
        print(recall)
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append((2 * precision * recall * 100) / (precision + recall))

    if mode == "max":  # for SumMe
        # print(np.max(f_scores))
        return np.max(f_scores)
    else:
        # print((np.sum(f_scores) / len(f_scores)))
        return np.sum(f_scores) / len(f_scores)
