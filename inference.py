import os
from openai import OpenAI

from env.environment import FactCheckEnv
from env.models import Action

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")
TASK_NAME = "factcheck"
BENCHMARK = "factcheckenv"


def parse_response(text):
    answer = ""
    source = None

    for line in text.split("\n"):
        line = line.strip()

        if line.lower().startswith("answer"):
            answer = line.split(":", 1)[-1].strip()

        if line.lower().startswith("source"):
            val = line.split(":", 1)[-1].strip()
            if val.lower() in ["none", "null"]:
                source = None
            else:
                try:
                    source = int(val)
                except:
                    source = None

    if not answer:
        answer = text.strip()

    return Action(answer=answer, source=source)


def main():
    env = FactCheckEnv()

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    step = 1
    success = False

    try:
        obs = env.reset("hard")  # evaluate hardest task

        prompt = f"""
You are an AI assistant performing grounded question answering.

STRICT INSTRUCTIONS:
- Answer ONLY using the provided documents
- Do NOT use outside knowledge
- If multiple documents conflict, choose the most recent
- If answer not present, say "Not enough information"

Documents:
{obs.documents}

Question:
{obs.question}

Output format:
Answer: ...
Source: ...
"""

        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            temperature=0
        )

        text = response.output[0].content[0].text
        action = parse_response(text)

        obs, reward, done, _ = env.step(action)

        rewards.append(reward.score)

        print(
            f"[STEP] step={step} action={action.answer} reward={reward.score:.2f} done={str(done).lower()} error=null"
        )

        score = max(0.0, min(sum(rewards), 1.0))
        success = score >= 0.5

    except Exception as e:
        print(
            f"[STEP] step=1 action=error reward=0.00 done=true error={str(e)}"
        )
        score = 0.0
        success = False

    print(
        f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}"
    )


if __name__ == "__main__":
    main()