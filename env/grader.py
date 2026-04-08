def normalize_score(score: float) -> float:
    """
    Ensure score is strictly within (0, 1)
    """
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score


def grade_easy(predicted_source, correct_source):
    score = 1.0 if predicted_source == correct_source else 0.0
    return normalize_score(score)


def grade_medium(predicted_answer, correct_answer):
    predicted_answer = predicted_answer.lower()
    correct_answer = correct_answer.lower()

    if correct_answer in predicted_answer:
        score = 1.0
    else:
        score = 0.0

    return normalize_score(score)


def grade_hard(predicted_answer, predicted_source, correct_answer, correct_source):
    score = 0.0

    predicted_answer = predicted_answer.lower()
    correct_answer = correct_answer.lower()

    # Answer correctness (partial reward)
    if correct_answer in predicted_answer:
        score += 0.5

    # Source correctness (partial reward)
    if predicted_source == correct_source:
        score += 0.5

    return normalize_score(score)