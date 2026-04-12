import os
from openai import OpenAI
from env.environment import FactCheckEnv
from env.models import Action

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
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


def run_single_task(env, difficulty, step):
    obs = env.reset(difficulty)

    prompt = f"""You are an AI assistant performing grounded question answering.

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
Source: <document id number>
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content
    except Exception as e:
        text = "Answer: Not enough information\nSource: null"

    action = parse_response(text)
    obs, reward, done, _ = env.step(action)

    print(
        f"[STEP] step={step} task={difficulty} grader=grade_{difficulty} action={action.answer} reward={reward.score:.2f} done=true error=null"
    )

    return reward.score


def main():
    env = FactCheckEnv()

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    step = 1

    try:
        for difficulty in ["easy", "medium", "hard"]:
            score = run_single_task(env, difficulty, step)
            rewards.append(score)
            step += 1

        score = sum(rewards) / len(rewards)

        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99

        success = score >= 0.5

    except Exception as e:
        print(
            f"[STEP] step=1 task=error grader=none action=error reward=0.01 done=true error={str(e)}"
        )
        score = 0.01
        success = False
        rewards = [0.01]

    print(
        f"[END] success={str(success).lower()} steps={step-1} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}"
    )


if __name__ == "__main__":
    main()