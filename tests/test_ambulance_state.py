
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment.traffic_environment import TrafficEnvironment

env = TrafficEnvironment(use_gui=False)

try:
    state = env.reset(seed=42)

    print("Initial state:", state)
    print("Initial state length:", len(state))

    for i in range(15):

        env.advance_simulation(1)

        state = env.get_state()

        print("\nSimulation time:", i + 1)
        print("State:", state)
        print("State length:", len(state))

finally:
    env.close()
