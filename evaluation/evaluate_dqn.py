
from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment
from evaluation.metrics import TrafficMetrics


def evaluate_dqn():

    print("\n========== DQN EVALUATION ==========")

    # Create agent and load trained model
    agent = DQNAgent()

    # Try the ambulance-updated model first; fall back to baseline
    try:
        agent.load_model("dqn_ambulance_model_updated.pth")
    except (FileNotFoundError, KeyError, ValueError):
        try:
            agent.load_model("dqn_ambulance_model.pth")
        except (FileNotFoundError, KeyError, ValueError):
            print("WARNING: No valid DQN model found. Using untrained agent.")

    # Disable exploration
    agent.epsilon = 0

    # Create SUMO environment
    env = TrafficEnvironment(use_gui=False)

    # Create performance metrics
    metrics = TrafficMetrics()

    try:

        # Start simulation
        state = env.reset(seed=42)

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
