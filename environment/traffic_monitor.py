
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

# Incoming roads
INCOMING_EDGES = [
    "N_in",
    "S_in",
    "E_in",
    "W_in"
]


def get_traffic_data():

    traffic_data = {}

    for edge in INCOMING_EDGES:

        # Count vehicles on this road
        vehicle_count = traci.edge.getLastStepVehicleNumber(edge)

        # Count stopped vehicles
        queue_length = traci.edge.getLastStepHaltingNumber(edge)

        # Get total waiting time of vehicles
        waiting_time = traci.edge.getWaitingTime(edge)

        traffic_data[edge] = {
            "vehicle_count": vehicle_count,
            "queue_length": queue_length,
            "waiting_time": waiting_time
        }

    return traffic_data

