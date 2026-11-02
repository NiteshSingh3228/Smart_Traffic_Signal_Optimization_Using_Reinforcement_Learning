
import traci

INCOMING_EDGES = [
    "N_in",
    "S_in",
    "E_in",
    "W_in"
]


class TrafficMetrics:

    def __init__(self):

        # Total waiting seconds accumulated by all observed vehicles
        self.vehicle_waiting_times = {}

        # Total queue length across all incoming roads
        self.total_queue_length = 0

        # Number of simulation steps measured
        self.total_samples = 0

        # Total vehicles that completed their journeys
        self.completed_vehicles = 0

    def collect(self):

        # 1. Measure waiting time for every active vehicle
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            if vehicle_id not in self.vehicle_waiting_times:
                self.vehicle_waiting_times[vehicle_id] = 0

            speed = traci.vehicle.getSpeed(vehicle_id)

            # Consider a vehicle waiting if its speed is below 0.1 m/s
            if speed < 0.1:
                self.vehicle_waiting_times[vehicle_id] += 1

        # 2. Calculate total queue length across all incoming roads
        current_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in INCOMING_EDGES
        )

        self.total_queue_length += current_queue
        self.total_samples += 1

        # 3. Count vehicles that arrived during this simulation step
        self.completed_vehicles += traci.simulation.getArrivedNumber()

    def get_results(self):

        # Average waiting time per observed vehicle
        total_waiting_time = sum(
            self.vehicle_waiting_times.values()
        )

        observed_vehicles = len(self.vehicle_waiting_times)

        average_waiting_time = (
            total_waiting_time / observed_vehicles
            if observed_vehicles > 0 else 0
        )

        # Average total queue length per simulation step
        average_queue_length = (
            self.total_queue_length / self.total_samples
            if self.total_samples > 0 else 0
        )

        return {
            "average_waiting_time": round(average_waiting_time, 2),
            "average_queue_length": round(average_queue_length, 2),
            "completed_vehicles": self.completed_vehicles,
            "observed_vehicles": observed_vehicles,
            "simulation_time": traci.simulation.getTime()
        }
