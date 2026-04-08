import os
from openai import OpenAI

from env.environment import FactCheckEnv
from env.models import Action

# Baseline script for evaluating LLM performance on FactCheckEnv
# Uses OpenAI API and produces reproducible scores

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_response(text):
    answer = ""
    source = None

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if line.lower().startswith("answer"):
            answer = line.split(":", 1)[-1].strip()

        if line.lower().startswith("source"):
            value = line.split(":", 1)[-1].strip()

            if value.lower() in ["none", "null"]:
                source = None
            else:
                try:
                    source = int(value)
                except:
                    source = None

    # fallback if model outputs everything in one line
    if not answer:
        answer = text.strip()

    return Action(answer=answer, source=source)


def run_task(difficulty):
    env = FactCheckEnv()
    obs = env.reset(difficulty)

    prompt = f"""
        You are an AI assistant performing grounded question answering.

        STRICT INSTRUCTIONS:
        - Answer ONLY using the provided documents
        - Do NOT use outside knowledge
        - If multiple documents conflict, ALWAYS choose the most recent or updated information
        - Ignore outdated or irrelevant documents
        - Your answer MUST match exactly what is written in the document
        - Always select the correct source document ID
        - If answer is not present in any document, respond with: "Not enough information"

        Documents:
        {obs.documents}

        Question:
        {obs.question}

        Output format:
        Answer: <exact answer from document>
        Source: <document id>
        """

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0
        )

        text = response.output[0].content[0].text

        print("\n🔍 Model Output:")
        print(text)

    except Exception as e:
        print("Error during API call:", e)
        return 0.0

    action = parse_response(text)

    obs, reward, done, _ = env.step(action)

    return reward.score


def run_all():
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        print(f"\n🚀 Running task: {task.upper()}")
        score = run_task(task)
        results[task] = score
        print(f"{task.upper()} SCORE: {score}")

    avg_score = sum(results.values()) / len(results)

    print("\n📊 FINAL RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")

    print(f"\nAVERAGE SCORE: {avg_score}")


if __name__ == "__main__":
    run_all()