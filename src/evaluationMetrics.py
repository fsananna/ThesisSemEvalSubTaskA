def topKAcc(actualResult, predictedResult, K=1):
    # Calculate Top-K accuracy: checks if the top K predictions exactly match the actual order
    # actualResult: list of correct items in order (e.g., ground truth order)
    # predictedResult: list of predicted items in order (e.g., ranked images)
    # K: number of top predictions to check (default is 1)
    if not actualResult or not predictedResult:  # Check if either list is empty
        return 0.0  # Return 0 if no data
    correct = 0  # Counter for correct predictions
    for i in range(min(K, len(predictedResult), len(actualResult))):  # Loop through top K or min length
        if predictedResult[i] == actualResult[i]:  # Check for exact match at position i
            correct += 1  # Increment counter if correct
    return correct / min(K, len(actualResult))  # Return accuracy as a fraction

def NDCG(actualResult, predictedResult):
    # Calculate Normalized Discounted Cumulative Gain (NDCG) to evaluate ranking quality
    # actualResult: list of ground truth relevance scores or rankings
    # predictedResult: list of predicted relevance scores or rankings
    def DCG(scores):  # Helper function to calculate Discounted Cumulative Gain
        from math import log  # Import log for calculations
        return sum((2 ** score - 1) / (log(i + 2) / log(2)) for i, score in enumerate(scores) if score > 0)
    if not actualResult or not predictedResult or len(actualResult) != len(predictedResult):
        return 0.0  # Return 0 if lists are empty or mismatched
    ideal_dcg = DCG(sorted(actualResult, reverse=True))  # Calculate DCG for ideal (sorted) ranking
    if ideal_dcg == 0:  # Avoid division by zero
        return 0.0
    pred_dcg = DCG(predictedResult)  # Calculate DCG for predicted ranking
    return pred_dcg / ideal_dcg  # Return NDCG as the ratio of predicted to ideal DCG