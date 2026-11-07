
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

for possible_path in [
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.28.0/EclipseSUMO/share/sumo/tools",
    os.path.abspath(".venv/lib/python3.10/site-packages"),
]:
    if os.path.exists(possible_path) and possible_path not in sys.path:
        sys.path.append(possible_path)

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
