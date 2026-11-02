
import traci

from environment.traffic_environment import (
    TrafficEnvironment,
    EAST_WEST_GREEN
)


env = TrafficEnvironment(use_gui=False)

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
