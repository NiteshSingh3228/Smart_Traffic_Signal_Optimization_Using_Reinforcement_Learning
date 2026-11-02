
import traci

from environment.traffic_environment import TrafficEnvironment

env = TrafficEnvironment(use_gui=False)

try:
    state = env.reset()

    print("Initial state:", state)
    print("Initial simulation time:", traci.simulation.getTime())

    step_count = 0
    done = False

    while not done:

        # Alternate between East-West and North-South
        action = step_count % 2

        state, reward, done = env.step(action)

        step_count += 1

        print(
            f"Step: {step_count} | "
            f"Time: {traci.simulation.getTime()} | "
            f"Action: {action} | "
            f"Reward: {reward} | "
            f"Done: {done}"
        )

    print("\nEpisode finished!")
    print("Total steps:", step_count)
    print("Final simulation time:", traci.simulation.getTime())

finally:
    env.close()
