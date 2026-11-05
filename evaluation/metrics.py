
import traci

INCOMING_EDGES = [
    "N_in",
    "S_in",
    "E_in",
    "W_in"
]


class TrafficMetrics:

    def __init__(self):

        # Waiting seconds accumulated by normal and emergency vehicles
        self.vehicle_waiting_times = {}

        # Waiting seconds accumulated by ambulances only
        self.ambulance_waiting_times = {}

        # Total queue length across incoming roads
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

            # Consider a vehicle waiting if speed is below 0.1 m/s
            if speed < 0.1:
                self.vehicle_waiting_times[vehicle_id] += 1

            # 2. Measure ambulance-specific waiting time
            if traci.vehicle.getTypeID(vehicle_id) == "ambulance":

                if vehicle_id not in self.ambulance_waiting_times:
                    self.ambulance_waiting_times[vehicle_id] = 0

                if speed < 0.1:
                    self.ambulance_waiting_times[vehicle_id] += 1

        # 3. Calculate total queue length
        current_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in INCOMING_EDGES
        )

        self.total_queue_length += current_queue
        self.total_samples += 1

        # 4. Count vehicles that completed their journeys
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

        # Average queue length per simulation step
        average_queue_length = (
            self.total_queue_length / self.total_samples
            if self.total_samples > 0 else 0
        )

        # Ambulance-specific results
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
