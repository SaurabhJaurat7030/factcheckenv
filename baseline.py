import os
from openai import OpenAI

from env.environment import KnowledgeQAEnv
from env.models import Action

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_response(text):
    """
    Expected format:
    Answer: ...
    Source: <id>
    """
    answer = ""
    source = None

    lines = text.split("\n")
    for line in lines:
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        if line.lower().startswith("source:"):
            try:
                source = int(line.split(":", 1)[1].strip())
            except:
                source = None

    return Action(answer=answer, source=source)


def run_task(difficulty):
    env = KnowledgeQAEnv()
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

                Documents:
                {obs.documents}

                Question:
                {obs.question}

                Output format:
                Answer: <exact answer from document>
                Source: <document id>
            """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output[0].content[0].text

    action = parse_response(text)

    obs, reward, done, _ = env.step(action)

    return reward.score


def run_all():
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        score = run_task(task)
        results[task] = score
        print(f"{task.upper()} SCORE: {score}")

    avg_score = sum(results.values()) / len(results)
    print(f"\nAVERAGE SCORE: {avg_score}")


if __name__ == "__main__":
    run_all()