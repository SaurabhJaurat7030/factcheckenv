from env.environment import KnowledgeQAEnv
from env.models import Action


def run_demo():
    env = KnowledgeQAEnv()

    obs = env.reset("easy")

    print("\n📄 Documents:")
    for doc in obs.documents:
        print(f"{doc['id']}: {doc['text']}")

    print("\n❓ Question:")
    print(obs.question)

    # ✅ Auto agent (no input required)
    action = Action(
        answer="Refunds are allowed within 7 days",
        source=1
    )

    obs, reward, done, _ = env.step(action)

    print("\n🎯 Result:")
    print(f"Score: {reward.score}")
    print(f"Feedback: {reward.feedback}")


if __name__ == "__main__":
    run_demo()