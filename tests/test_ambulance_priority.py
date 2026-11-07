
import sys
import os

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

from environment.traffic_environment import (
    TrafficEnvironment,
    EAST_WEST_GREEN
)


env = TrafficEnvironment(use_gui=False, network_mode="single")

try:

    print("Starting ambulance priority test...")

    # Start SUMO
    state = env.reset(seed=42)

    # Set East-West green initially
    traci.trafficlight.setPhase(
        "center",
        EAST_WEST_GREEN
    )

    env.current_phase = EAST_WEST_GREEN

    traci.trafficlight.setPhaseDuration(
        "center",
        100
    )

    print("Initial signal: East-West GREEN")

    # First 30-second RL step
    next_state, reward, done = env.step(action=1)

    # Continue with East-West green so the ambulance keeps waiting
    if not done:
        next_state, reward, done = env.step(action=1)

    print("\nSimulation time:", traci.simulation.getTime())
    print("Current signal phase:", env.current_phase)
    print("State length:", len(next_state))
    print("Reward:", reward)

    if env.current_phase == 0:
        print("\nSUCCESS: North-South green was activated!")
    else:
        print("\nPriority not activated. Check ambulance timing.")

finally:

    env.close()
