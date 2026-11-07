import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment.traffic_environment import TrafficEnvironment

env = TrafficEnvironment(use_gui=False)

try:
    env.reset(seed=42)

    for i in range(15):

        info = env.get_ambulance_info()

        print("Simulation time:", i)
        print("Ambulance information:", info)

        env.advance_simulation(1)

finally:
    env.close()