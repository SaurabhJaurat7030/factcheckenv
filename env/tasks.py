import json


class TaskLoader:
    def __init__(self, data_path="data/dataset.json"):
        with open(data_path, "r") as f:
            self.data = json.load(f)

        # 🔥 group tasks by difficulty (ONLY required ones)
        self.grouped = {
            "easy": [],
            "medium": [],
            "hard": []
        }

        for item in self.data:
            task_type = item.get("task")

            if task_type in self.grouped:
                self.grouped[task_type].append(item)

        # 🔥 deterministic index tracking
        self.indices = {
            "easy": 0,
            "medium": 0,
            "hard": 0
        }

    def get_task(self, difficulty="easy"):
        tasks = self.grouped.get(difficulty, [])

        if not tasks:
            raise ValueError(f"No tasks found for difficulty: {difficulty}")

        # 🔥 deterministic cycling (NO randomness)
        index = self.indices[difficulty] % len(tasks)
        self.indices[difficulty] += 1

        return tasks[index]