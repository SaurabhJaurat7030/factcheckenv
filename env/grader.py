def normalize_score(score: float) -> float:
    """
    Ensure score is strictly within (0, 1)
    """
    # Add epsilon safety
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99

    # Also protect against rounding issues
    if score == 0.0:
        return 0.01
    if score == 1.0:
        return 0.99

    return score


def grade_easy(predicted_source, correct_source):
    if predicted_source == correct_source:
        score = 0.99   # instead of 1.0
    else:
        score = 0.01   # instead of 0.0

    return normalize_score(score)


def grade_medium(predicted_answer, correct_answer):
    predicted_answer = (predicted_answer or "").lower()
    correct_answer = (correct_answer or "").lower()

    if correct_answer in predicted_answer:
        score = 0.99
    else:
        score = 0.01

    return normalize_score(score)


def grade_hard(predicted_answer, predicted_source, correct_answer, correct_source):
    score = 0.0

    predicted_answer = (predicted_answer or "").lower()
    correct_answer = (correct_answer or "").lower()

    # Partial scoring
    if correct_answer in predicted_answer:
        score += 0.5

    if predicted_source == correct_source:
        score += 0.5

    # Convert 0.0 → 0.01 and 1.0 → 0.99 safely
    return normalize_score(score)