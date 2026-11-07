
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

from evaluation.metrics import TrafficMetrics


TRAFFIC_LIGHT_ID = "center"

def _resolve_tls(preferred=None):
    try:
        tls = traci.trafficlight.getIDList()
        if preferred and preferred in tls:
            return [preferred]
        if tls:
            return list(tls)
        return ["center"]
    except Exception:
        return ["center"]

INCOMING_EDGES = ["N_in", "S_in", "E_in", "W_in"]

NORTH_SOUTH_GREEN = 0
NORTH_SOUTH_YELLOW = 1

EAST_WEST_GREEN = 2
EAST_WEST_YELLOW = 3

GREEN_DURATION = 30
YELLOW_DURATION = 3

# FIX 1.2: Standardize to 600s to match MAX_EPISODE_TIME in traffic_environment.py
MAX_SIMULATION_TIME = 600
AMBULANCE_WAIT_THRESHOLD = 15


def get_ambulance_info():

    ambulance_info = []

    for vehicle_id in traci.vehicle.getIDList():

        if traci.vehicle.getTypeID(vehicle_id) != "ambulance":
            continue

        route = traci.vehicle.getRoute(vehicle_id)

        if len(route) < 2:
            continue

        direction = route[0]
        outgoing_edge = route[1]

        if direction not in INCOMING_EDGES:
            continue

        road_id = traci.vehicle.getRoadID(vehicle_id)

        # FIX: Skip ambulances that have already crossed
        if road_id == outgoing_edge:
            continue

        if road_id.startswith(":"):
            continue

        ambulance_info.append({
            "id": vehicle_id,
            "direction": direction,
            "outgoing_edge": outgoing_edge,
            "road_id": road_id,
            "waiting_time": traci.vehicle.getWaitingTime(vehicle_id)
        })

    return ambulance_info


def update_ambulance_priority(priority_ambulances):

    ambulance_info = get_ambulance_info()

    current_ambulances = {
        ambulance["id"]: ambulance
        for ambulance in ambulance_info
    }

    # Release priority if ambulance has cleared the intersection
    for ambulance_id in list(priority_ambulances):

        if ambulance_id not in current_ambulances:

            del priority_ambulances[ambulance_id]
            continue

    # Activate priority after threshold seconds of waiting
    for ambulance in ambulance_info:

        if ambulance["waiting_time"] >= AMBULANCE_WAIT_THRESHOLD:

            ambulance_id = ambulance["id"]

            if ambulance_id not in priority_ambulances:

                priority_ambulances[ambulance_id] = ambulance["direction"]

    if priority_ambulances:

        ambulance_id = next(iter(priority_ambulances))

        return {
            "id": ambulance_id,
            "direction": priority_ambulances[ambulance_id]
        }

    return None


def get_green_phase(direction):

    if direction in ["N_in", "S_in"]:
        return NORTH_SOUTH_GREEN

    return EAST_WEST_GREEN


def switch_to_green(target_green, current_phase, metrics):

    if current_phase == target_green:
        return current_phase

    if current_phase == NORTH_SOUTH_GREEN:
        yellow_phase = NORTH_SOUTH_YELLOW
    else:
        yellow_phase = EAST_WEST_YELLOW

    # Safely switch through yellow
    for tl in _resolve_tls():
        try:
            traci.trafficlight.setPhase(tl, yellow_phase)
        except Exception:
            pass

    for _ in range(YELLOW_DURATION):

        if traci.simulation.getTime() >= MAX_SIMULATION_TIME:
            return current_phase

        traci.simulationStep()
        metrics.collect()

    # Activate the required green
    for tl in _resolve_tls():
        try:
            traci.trafficlight.setPhase(tl, target_green)
            traci.trafficlight.setPhaseDuration(tl, GREEN_DURATION)
        except Exception:
            pass

    return target_green


def run_fixed_time_controller(ambulance_priority=False, max_simulation_time=MAX_SIMULATION_TIME):

    metrics = TrafficMetrics()
    limit = max_simulation_time if max_simulation_time is not None else float("inf")

    current_phase = NORTH_SOUTH_GREEN

    priority_ambulances = {}

    green_elapsed = 0

    for tl in _resolve_tls():
        try:
            traci.trafficlight.setPhase(tl, current_phase)
            traci.trafficlight.setPhaseDuration(tl, GREEN_DURATION)
        except Exception:
            pass

    while traci.simulation.getTime() < limit:

        # Advance simulation by one second
        traci.simulationStep()
        metrics.collect()

        # Check ambulance priority
        ambulance = None

        if ambulance_priority:

            ambulance = update_ambulance_priority(
                priority_ambulances
            )

        # Handle active ambulance priority
        if ambulance is not None:

            emergency_green = get_green_phase(
                ambulance["direction"]
            )

            if current_phase != emergency_green:

                current_phase = switch_to_green(
                    emergency_green,
                    current_phase,
                    metrics
                )

            green_elapsed = 0

            # Keep emergency green until ambulance clears
            while traci.simulation.getTime() < limit:

                traci.simulationStep()
                metrics.collect()

                ambulance = update_ambulance_priority(
                    priority_ambulances
                )

                if ambulance is None:
                    break

                emergency_green = get_green_phase(
                    ambulance["direction"]
                )

                if current_phase != emergency_green:

                    current_phase = switch_to_green(
                        emergency_green,
                        current_phase,
                        metrics
                    )

            # Resume normal fixed-time cycle by switching to opposite phase
            # to prevent starving cross-traffic that waited during emergency
            next_green = (
                EAST_WEST_GREEN
                if current_phase == NORTH_SOUTH_GREEN
                else NORTH_SOUTH_GREEN
            )
            current_phase = switch_to_green(
                next_green,
                current_phase,
                metrics
            )
            green_elapsed = 0
            continue

        # Normal fixed-time green duration
        green_elapsed += 1

        if green_elapsed >= GREEN_DURATION:

            if current_phase == NORTH_SOUTH_GREEN:
                next_green = EAST_WEST_GREEN
            else:
                next_green = NORTH_SOUTH_GREEN

            current_phase = switch_to_green(
                next_green,
                current_phase,
                metrics
            )

            green_elapsed = 0

    results = metrics.get_results()

    return results
