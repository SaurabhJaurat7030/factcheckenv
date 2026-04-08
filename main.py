from fastapi import FastAPI
from env.environment import FactCheckEnv

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