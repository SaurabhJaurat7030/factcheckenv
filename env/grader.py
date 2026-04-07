def grade_easy(predicted_source, correct_source):
    return 1.0 if predicted_source == correct_source else 0.0


def grade_medium(predicted_answer, correct_answer):
    predicted_answer = predicted_answer.lower()
    correct_answer = correct_answer.lower()

    if correct_answer in predicted_answer:
        return 1.0
    return 0.0


def grade_hard(predicted_answer, predicted_source, correct_answer, correct_source):
    score = 0.0

    predicted_answer = predicted_answer.lower()
    correct_answer = correct_answer.lower()

    # Answer correctness
    if correct_answer in predicted_answer:
        score += 0.5

    # Source correctness
    if predicted_source == correct_source:
        score += 0.5

    return score