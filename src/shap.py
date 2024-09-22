from typing import Callable

from shap import Explanation
import numpy as np


def calculate_explanation_power_shap(scoring_function: Callable[[str], float], text: str,
                                     explanation: Explanation) -> np.ndarray:
    thresholds = [np.percentile(abs(explanation.values), q) for q in range(10, 100, 10)]
    base_score = scoring_function(text)
    scores = []
    for threshold in thresholds:
        filtered_text = " ".join(
            [d for (d, v) in zip(explanation.feature_names, explanation.values) if abs(v) >= threshold])
        score_shortened = scoring_function(filtered_text)
        scores.append(score_shortened)
    scores_diff = abs(np.array(scores) - base_score)
    return scores_diff
