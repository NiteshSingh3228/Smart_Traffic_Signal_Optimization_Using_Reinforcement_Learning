
import traci

from environment.traffic_environment import (
    TrafficEnvironment,
    EAST_WEST_GREEN,
    NORTH_SOUTH_GREEN
)

env = TrafficEnvironment(use_gui=False)

try:

    print("Starting ambulance crossing test...")

    env.reset(seed=42)

    # Initially give East-West green
    traci.trafficlight.setPhase("center", EAST_WEST_GREEN)
    traci.trafficlight.setPhaseDuration("center", 100)

    env.current_phase = EAST_WEST_GREEN

    print("Initial signal: East-West GREEN")

    # Continue choosing East-West.
    # Ambulance priority should override this after 15 seconds.
    for step_number in range(10):

        if traci.simulation.getMinExpectedNumber() == 0:
            break

        next_state, reward, done = env.step(action=1)

        print(
            "Step:", step_number + 1,
            "| Time:", traci.simulation.getTime(),
            "| Phase:", env.current_phase
        )

        if done:
            break

    # Check whether ambulance has exited
    remaining_ambulances = [
        vehicle_id
        for vehicle_id in traci.vehicle.getIDList()
        if traci.vehicle.getTypeID(vehicle_id) == "ambulance"
    ]

    if len(remaining_ambulances) == 0:
        print("\nSUCCESS: Ambulance has exited the network.")
    else:
        print("\nAmbulance is still in simulation:", remaining_ambulances)

    # Test whether normal RL action can control the signal again
    if not done:

        next_state, reward, done = env.step(action=1)

        print("\nNormal RL action selected: East-West")
        print("Current signal phase:", env.current_phase)

        if env.current_phase == EAST_WEST_GREEN:
            print("SUCCESS: Normal RL control has resumed.")
        else:
            print("Signal is not yet East-West green.")

finally:

    env.close()
