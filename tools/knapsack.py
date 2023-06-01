def knapsack(weights, values, max_duration):
    n = len(weights)
    K = [[0 for _ in range(max_duration + 1)] for _ in range(n + 1)]

    # Build K[][] in a bottom-up manner
    for i in range(n + 1):
        for w in range(max_duration + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i - 1] <= w:
                K[i][w] = max(values[i - 1] + K[i - 1][w - weights[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # Store the result of knapsack
    res = K[n][max_duration]
    print("Total importance score:", res)

    selected_shots = []
    w = max_duration
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # either the result comes from the top (K[i-1][w]) or from (values[i-1] + K[i-1][w-weights[i-1]]) as in Knapsack table.
        # If it comes from the latter one, it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included
            selected_shots.append(i - 1)

            # Since this weight is included, its value is deducted
            res = res - values[i - 1]
            w = w - weights[i - 1]

    print("Selected shots:", selected_shots)


if __name__ == "__main__":
    # keyshot durations (weights)
    durations = [10, 20, 30, 40, 50]

    # keyshot importance scores (values)
    importance_scores = [60, 100, 120, 200, 300]

    # maximum duration for the summary
    # in our case it is 15% from duration of video
    max_duration = 100
    knapsack(durations, importance_scores, max_duration)
