from fastapi import FastAPI
from env.environment import FactCheckEnv
import uvicorn


def create_app():
    app = FastAPI()

    @app.get("/")
    def home():
        return {"status": "running"}

    @app.get("/reset")
    @app.post("/reset")
    def reset():
        env = FactCheckEnv()
        obs = env.reset("easy")

        return {
            "documents": obs.documents,
            "question": obs.question,
            "history": obs.history
        }

    return app


app = create_app()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()