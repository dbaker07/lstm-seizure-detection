import scipy.stats as st
import numpy as np

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return (0, 0)

    p_hat = successes / total
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + (z**2 / total)
    adjusted_p = (p_hat + (z**2 / (2 * total))) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))) / denominator

    lower_bound = max(0, adjusted_p - margin)
    upper_bound = min(1, adjusted_p + margin)

    return lower_bound, upper_bound

if __name__ == "__main__":
    correct_predictions = int(input("Enter correct predictions: "))
    total_predictions = int(input("Enter total predictions: "))
    confidence_level = 0.95

    lower, upper = wilson_score_interval(correct_predictions, total_predictions, confidence_level)
    print(f"Wilson Score {confidence_level*100:.0f}% Confidence Interval: ({lower:.4f}, {upper:.4f})")
