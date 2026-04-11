from env.models import Observation, Action, Reward
from env.tasks import TaskLoader
from env.grader import grade_easy, grade_medium, grade_hard


class FactCheckEnv:
    def __init__(self):
        self.loader = TaskLoader()
        self.current_task = None
        self.done = False
        self.history = []
        self.current_difficulty = None

    def reset(self, difficulty="easy"):
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "easy"

        self.current_difficulty = difficulty
        self.current_task = self.loader.get_task(difficulty)

        if not self.current_difficulty:
            self.current_difficulty = self.current_task.get("task", "easy")

        self.done = False
        self.history = []

        return Observation(
            documents=self.current_task["documents"],
            question=self.current_task["question"],
            history=self.history
        )

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode already finished. Call reset().")

        if not self.current_difficulty:
            self.current_difficulty = self.current_task.get("task", "easy")

        correct_answer = (self.current_task.get("answer") or "").lower()
        correct_source = self.current_task.get("source")

        predicted_answer = (action.answer or "").lower()
        predicted_source = action.source

        feedback = []

 
        if correct_source is None:
            if "not enough information" in predicted_answer:
                score = 0.99
                feedback.append("Correctly identified missing information")
            else:
                score = 0.01
                feedback.append("Should have said no information available")

        else:

            if self.current_difficulty == "easy":
                score = grade_easy(predicted_source, correct_source)

            elif self.current_difficulty == "medium":
                score = grade_medium(predicted_answer, correct_answer)

            elif self.current_difficulty == "hard":
                score = grade_hard(
                    predicted_answer,
                    predicted_source,
                    correct_answer,
                    correct_source
                )

            else:
                score = 0.5

        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99

        self.done = True

        observation = Observation(
            documents=self.current_task["documents"],
            question=self.current_task["question"],
            history=self.history + [action.answer]
        )

        reward = Reward(
            score=score,
            feedback=", ".join(feedback) if feedback else "Evaluation completed"
        )

        return observation, reward, self.done, {}

    def state(self):
        return {
            "documents": self.current_task["documents"],
            "question": self.current_task["question"],
            "history": self.history
        }