from typing import Dict, Callable
import numpy as np

from lime.explanation import Explanation


def calculate_explanation_power_lime(scoring_function: Callable[[str], float], text: str, explanation: Explanation) -> \
        np.ndarray:
    explanation_values = np.array(list(list(zip(*explanation.local_exp[1]))[1]))
    position_importance_dict = get_position_to_importance_dict(explanation)
    thresholds = [np.percentile(abs(explanation_values), q) for q in range(10, 100, 10)]
    base_score = scoring_function(text)
    scores = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        filtered_text = filter_text(explanation, position_importance_dict, threshold)
        score = scoring_function(filtered_text)
        scores[i] = score
    scores_diff = abs(scores - base_score)
    return scores_diff


def get_position_to_importance_dict(explanation: Explanation) -> Dict:
    explanation_values = list(zip(*explanation.local_exp[1]))[1]
    return dict(zip(explanation.domain_mapper.indexed_string.inverse_vocab, explanation_values))


def filter_text(explanation: Explanation, importance_dict: Dict, threshold: float) -> str:
    result = []
    for word in explanation.domain_mapper.indexed_string.as_list:
        if abs(importance_dict.get(word, 0)) > threshold:
            result.append(word)
    return ' '.join(result)
