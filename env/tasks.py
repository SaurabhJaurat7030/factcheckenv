import json
import random


class TaskLoader:
    def __init__(self, data_path="data/dataset.json"):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def get_task(self, difficulty="easy"):
        # filter tasks by difficulty
        filtered = [item for item in self.data if item["task"] == difficulty]

        if not filtered:
            raise ValueError(f"No tasks found for difficulty: {difficulty}")

        # pick random task
        return random.choice(filtered)