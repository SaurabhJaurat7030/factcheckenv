from env.environment import FactCheckEnv
from env.models import Action


def run_demo():
    env = FactCheckEnv()
    tasks = [
        ("easy", "Refunds are allowed within 7 days", 1),
        ("medium", "Support is available 24/7", 1),
        ("hard", "Premium users get delivery in 2 days", 2),
        # No-answer case (hard task with no info)
        ("hard", "Not enough information", None)
    ]

    for idx, (difficulty, answer, source) in enumerate(tasks):
        print(f"\n=== Task {idx+1}: {difficulty.upper()} ===")
        obs = env.reset(difficulty)
        print("\n📄 Documents:")
        for doc in obs.documents:
            print(f"{doc['id']}: {doc['text']}")
        print("\n❓ Question:")
        print(obs.question)

        action = Action(answer=answer, source=source)
        obs, reward, done, _ = env.step(action)
        print("\n🎯 Result:")
        print(f"Score: {reward.score}")
        print(f"Feedback: {reward.feedback}")


if __name__ == "__main__":
    run_demo()