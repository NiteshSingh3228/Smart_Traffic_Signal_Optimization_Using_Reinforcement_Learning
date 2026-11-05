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

INCOMING_EDGES = [
    "N_in",
    "S_in",
    "E_in",
    "W_in"
]


class TrafficMetrics:

    def __init__(self, on_update=None):

        self.vehicle_waiting_times = {}
        self.ambulance_waiting_times = {}

        self.total_queue_length = 0
        self.total_samples = 0
        self.completed_vehicles = 0

        # Actual vehicle travel-time tracking
        self.departure_times = {}
        self.vehicle_travel_times = []

        # Optional callback for live dashboard updates
        self.on_update = on_update

    def collect(self):

        current_time = traci.simulation.getTime()

        # 1. Record newly departed vehicles
        departed_ids = traci.simulation.getDepartedIDList()

        for vehicle_id in departed_ids:
            self.departure_times[vehicle_id] = current_time

        # 2. Measure waiting time of active vehicles
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            if vehicle_id not in self.vehicle_waiting_times:
                self.vehicle_waiting_times[vehicle_id] = 0

            speed = traci.vehicle.getSpeed(vehicle_id)

            if speed < 0.1:
                self.vehicle_waiting_times[vehicle_id] += 1

            if traci.vehicle.getTypeID(vehicle_id) == "ambulance":

                if vehicle_id not in self.ambulance_waiting_times:
                    self.ambulance_waiting_times[vehicle_id] = 0

                if speed < 0.1:
                    self.ambulance_waiting_times[vehicle_id] += 1

        # 3. Record completed vehicle travel times
        arrived_ids = traci.simulation.getArrivedIDList()

        for vehicle_id in arrived_ids:

            if vehicle_id in self.departure_times:

                travel_time = (
                    current_time - self.departure_times[vehicle_id]
                )

                self.vehicle_travel_times.append(travel_time)
                del self.departure_times[vehicle_id]

        self.completed_vehicles += len(arrived_ids)

        # 4. Calculate queue length
        current_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in INCOMING_EDGES
        )

        self.total_queue_length += current_queue
        self.total_samples += 1

        # 5. Send updated metrics to dashboard
        if self.on_update is not None:
            self.on_update(self.get_results())

    def get_results(self):

        # Average waiting time
        total_waiting_time = sum(
            self.vehicle_waiting_times.values()
        )

        observed_vehicles = len(self.vehicle_waiting_times)

        average_waiting_time = (
            total_waiting_time / observed_vehicles
            if observed_vehicles > 0 else 0
        )

        # Average queue length
        average_queue_length = (
            self.total_queue_length / self.total_samples
            if self.total_samples > 0 else 0
        )

        # Average actual travel time of completed vehicles
        average_travel_time = (
            sum(self.vehicle_travel_times) / len(self.vehicle_travel_times)
            if self.vehicle_travel_times else 0
        )

        # Ambulance metrics
        ambulance_count = len(self.ambulance_waiting_times)

        total_ambulance_waiting = sum(
            self.ambulance_waiting_times.values()
        )

        average_ambulance_waiting = (
            total_ambulance_waiting / ambulance_count
            if ambulance_count > 0 else 0
        )

        maximum_ambulance_waiting = (
            max(self.ambulance_waiting_times.values())
            if ambulance_count > 0 else 0
        )

        return {
            "average_waiting_time": round(average_waiting_time, 2),
            "average_travel_time": round(average_travel_time, 2),
            "average_queue_length": round(average_queue_length, 2),
            "completed_vehicles": self.completed_vehicles,
            "observed_vehicles": observed_vehicles,
            "average_ambulance_waiting_time": round(
                average_ambulance_waiting, 2
            ),
            "maximum_ambulance_waiting_time": round(
                maximum_ambulance_waiting, 2
            ),
            "ambulances_observed": ambulance_count,
            "simulation_time": traci.simulation.getTime()
        }
