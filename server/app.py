from fastapi import FastAPI, Request
from env.environment import FactCheckEnv
from env.models import Action
import uvicorn


def create_app():
    app = FastAPI(title="FactCheckEnv")

    TASKS = [
        {
            "id": "easy",
            "name": "easy",
            "description": "Retrieve the correct document for a given question",
            "grader": "grade_easy",
            "difficulty": "easy"
        },
        {
            "id": "medium",
            "name": "medium",
            "description": "Answer a question using provided documents",
            "grader": "grade_medium",
            "difficulty": "medium"
        },
        {
            "id": "hard",
            "name": "hard",
            "description": "Answer a question, cite source, resolve conflicts",
            "grader": "grade_hard",
            "difficulty": "hard"
        }
    ]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def home():
        return {"status": "running", "tasks": len(TASKS)}

    @app.get("/tasks")
    @app.post("/tasks")
    def get_tasks():
        return {"tasks": TASKS}

    @app.get("/reset")
    @app.post("/reset")
    def reset(task_id: str = "easy"):
        env = FactCheckEnv()
        obs = env.reset(task_id)
        return {
            "task_id": task_id,
            "documents": obs.documents,
            "question": obs.question,
            "history": obs.history
        }

    @app.post("/step")
    async def step(request: Request):
        data = await request.json()
        env = FactCheckEnv()
        task_id = data.get("task_id", "easy")
        env.reset(task_id)
        action = Action(
            answer=data.get("answer", ""),
            source=data.get("source", None)
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": {
                "documents": obs.documents,
                "question": obs.question,
                "history": obs.history
            },
            "reward": reward.score,
            "feedback": reward.feedback,
            "done": done,
            "info": info
        }

    @app.get("/state")
    @app.post("/state")
    def state():
        return {"state": "ready", "tasks": [t["id"] for t in TASKS]}

    return app


app = create_app()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()