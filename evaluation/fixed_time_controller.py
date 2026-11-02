
import traci

from evaluation.metrics import TrafficMetrics

TRAFFIC_LIGHT_ID = "center"

NORTH_SOUTH_GREEN = 0
NORTH_SOUTH_YELLOW = 1

EAST_WEST_GREEN = 2
EAST_WEST_YELLOW = 3

GREEN_DURATION = 30
YELLOW_DURATION = 3
MAX_SIMULATION_TIME = 600


def run_fixed_time_controller():

    metrics = TrafficMetrics()

    current_phase = NORTH_SOUTH_GREEN

    traci.trafficlight.setPhase(
        TRAFFIC_LIGHT_ID,
        current_phase
    )

    while traci.simulation.getTime() < MAX_SIMULATION_TIME:

        # Run the current green phase
        traci.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            GREEN_DURATION
        )

        for _ in range(GREEN_DURATION):

            if traci.simulation.getTime() >= MAX_SIMULATION_TIME:
                break

            traci.simulationStep()
            metrics.collect()

        if traci.simulation.getTime() >= MAX_SIMULATION_TIME:
            break

        # Select the correct yellow and next green phases
        if current_phase == NORTH_SOUTH_GREEN:
            yellow_phase = NORTH_SOUTH_YELLOW
            next_phase = EAST_WEST_GREEN
        else:
            yellow_phase = EAST_WEST_YELLOW
            next_phase = NORTH_SOUTH_GREEN

        # Run yellow phase
        traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            yellow_phase
        )

        for _ in range(YELLOW_DURATION):

            if traci.simulation.getTime() >= MAX_SIMULATION_TIME:
                break

            traci.simulationStep()
            metrics.collect()

        if traci.simulation.getTime() >= MAX_SIMULATION_TIME:
            break

        # IMPORTANT: Actually switch to the next green phase
        traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            next_phase
        )

        current_phase = next_phase

    print("\nFixed-time simulation completed!")

    results = metrics.get_results()

    print("\n========== TRAFFIC PERFORMANCE ==========")

    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results
