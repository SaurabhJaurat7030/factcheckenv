from env.models import Observation, Action, Reward
from env.tasks import TaskLoader


class KnowledgeQAEnv:
    def __init__(self):
        self.loader = TaskLoader()
        self.current_task = None
        self.done = False
        self.history = []

    def reset(self, difficulty="easy"):
        self.current_task = self.loader.get_task(difficulty)
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

        correct_answer = self.current_task["answer"].lower()
        correct_source = self.current_task["source"]

        predicted_answer = action.answer.lower()
        predicted_source = action.source

        score = 0.0
        feedback = []

        # ✅ Answer check
        if correct_answer in predicted_answer:
            score += 0.5
            feedback.append("Correct answer")
        else:
            feedback.append("Incorrect answer")

        # ✅ Source check
        if predicted_source == correct_source:
            score += 0.3
            feedback.append("Correct source")
        else:
            feedback.append("Incorrect source")

        # ✅ Grounding check (answer present in source doc)
        source_doc = next(
            (doc for doc in self.current_task["documents"] if doc["id"] == correct_source),
            None
        )

        if source_doc and correct_answer in source_doc["text"].lower():
            score += 0.2
            feedback.append("Answer grounded in document")

        # ❌ Hallucination penalty
        if correct_answer not in predicted_answer:
            score -= 0.3
            feedback.append("Possible hallucination")

        # mark done
        self.done = True

        observation = Observation(
            documents=self.current_task["documents"],
            question=self.current_task["question"],
            history=self.history + [action.answer]
        )

        reward = Reward(
            score=max(0.0, min(score, 1.0)),
            feedback=", ".join(feedback)
        )

        return observation, reward, self.done, {}
    
    def state(self):
        return {
            "documents": self.current_task["documents"],
            "question": self.current_task["question"],
            "history": self.history
        }