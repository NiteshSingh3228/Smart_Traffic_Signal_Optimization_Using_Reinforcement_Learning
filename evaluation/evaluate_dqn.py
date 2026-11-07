
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment
from evaluation.metrics import TrafficMetrics


def evaluate_dqn(seed=42):

    print("\n========== DQN EVALUATION ==========")

    # Create agent and load trained model
    agent = DQNAgent()

    # Prioritize triple-source integrated model, then ambulance-updated, then baseline
    loaded = False
    for model_name in [
        "dqn_integrated_model.pth",
        "dqn_ambulance_model_updated.pth",
        "dqn_ambulance_model.pth"
    ]:
        try:
            agent.load_model(model_name)
            print(f"Loaded model: {model_name}")
            loaded = True
            break
        except (FileNotFoundError, KeyError, ValueError):
            continue

    if not loaded:
        print("WARNING: No valid DQN model found. Using untrained agent.")

    # Disable exploration
    agent.epsilon = 0

    # Create SUMO environment (defaults to corridor_2)
    env = TrafficEnvironment(use_gui=False)

    # Create performance metrics with incoming edges from the active environment
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)

    try:

        # Start simulation
        state = env.reset(seed=seed)

        done = False
        total_reward = 0
        step_count = 0

        while not done:

            # Select action using trained DQN
            action = agent.choose_action(state)

            # Execute action in SUMO
            next_state, reward, done = env.step(
                action,
                metrics=metrics
            )

            state = next_state

            total_reward += reward
            step_count += 1

        print("\nDQN simulation completed!")
        print("Total steps:", step_count)
        print("Total reward:", total_reward)

        results = metrics.get_results()

        print("\n========== DQN PERFORMANCE ==========")

        for metric, value in results.items():
            print(f"{metric}: {value}")

        return results

    finally:

        env.close()


if __name__ == "__main__":
    evaluate_dqn()
