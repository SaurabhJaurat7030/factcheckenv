from fastapi import FastAPI, Request
from env.environment import FactCheckEnv
from env.models import Action
import uvicorn


def create_app():
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def home():
        return {"status": "running"}

    @app.get("/tasks")
    def get_tasks():
        return {
            "tasks": [
                {
                    "id": "easy",
                    "name": "easy",
                    "description": "Retrieve the correct document for a given question",
                    "grader": "env.grader:grade_easy"
                },
                {
                    "id": "medium",
                    "name": "medium",
                    "description": "Answer a question using provided documents",
                    "grader": "env.grader:grade_medium"
                },
                {
                    "id": "hard",
                    "name": "hard",
                    "description": "Answer a question and cite the correct source with conflict resolution",
                    "grader": "env.grader:grade_hard"
                }
            ]
        }

    @app.get("/reset")
    @app.post("/reset")
    def reset(task_id: str = "easy"):
        env = FactCheckEnv()
        obs = env.reset(task_id)
        return {
            "documents": obs.documents,
            "question": obs.question,
            "history": obs.history
        }

    @app.post("/step")
    async def step(request: Request):
        data = await request.json()
        env = FactCheckEnv()
        env.reset(data.get("task_id", "easy"))
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
            "done": done,
            "info": info
        }

    @app.get("/state")
    def state():
        return {"state": "no active session"}

    return app


app = create_app()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()