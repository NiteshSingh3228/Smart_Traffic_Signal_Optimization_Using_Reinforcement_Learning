
import traci

TRAFFIC_LIGHT_ID = "center"

INCOMING_EDGES = [
    "N_in", "S_in", "E_in", "W_in"
]

NORTH_SOUTH_GREEN = 0
NORTH_SOUTH_YELLOW = 1
EAST_WEST_GREEN = 2
EAST_WEST_YELLOW = 3

YELLOW_DURATION = 3
GREEN_DURATION = 30
MAX_EPISODE_TIME = 600

AMBULANCE_WAIT_THRESHOLD = 15


class TrafficEnvironment:

    def __init__(self, use_gui=False):

        self.current_phase = NORTH_SOUTH_GREEN
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.is_running = False

        # Remember ambulances that have received priority
        self.priority_ambulances = {}

        # Track whether ambulance priority affected the current RL step
        self.emergency_override_during_step = False

    # =====================================
    # 1. RESET ENVIRONMENT
    # =====================================

    def reset(self, seed=42):

        sumo_cmd = [
            self.sumo_binary,
            "-n", "sumo/network/intersection.net.xml",
            "-r", "sumo/routes/routes.rou.xml",
            "--no-step-log", "true",
            "--seed", str(seed)
        ]

        traci.start(sumo_cmd)

        self.is_running = True
        self.current_phase = NORTH_SOUTH_GREEN

        # Clear priority information for a new episode
        self.priority_ambulances = {}

        # Reset the override flag
        self.emergency_override_during_step = False

        traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            NORTH_SOUTH_GREEN
        )

        traci.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            GREEN_DURATION
        )

        return self.get_state()

    # =====================================
    # 2. GET TRAFFIC STATE
    # =====================================

    def get_state(self):

        state = []

        # Normal traffic information
        for edge in INCOMING_EDGES:

            vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
            queue_length = traci.edge.getLastStepHaltingNumber(edge)

            state.append(vehicle_count)
            state.append(queue_length)

        ambulance_info = self.get_ambulance_info()

        # Ambulance count in each direction
        for edge in INCOMING_EDGES:

            ambulance_count = sum(
                1 for ambulance in ambulance_info
                if ambulance["direction"] == edge
            )

            state.append(ambulance_count)

        # Ambulance waiting time in each direction
        for edge in INCOMING_EDGES:

            waiting_time = max(
                (
                    ambulance["waiting_time"]
                    for ambulance in ambulance_info
                    if ambulance["direction"] == edge
                ),
                default=0
            )

            state.append(waiting_time)

        # Current traffic signal phase
        state.append(self.current_phase)

        return tuple(state)

    # =====================================
    # 3. DETECT AMBULANCES
    # =====================================

    def get_ambulance_info(self):

        ambulance_info = []

        for vehicle_id in traci.vehicle.getIDList():

            if traci.vehicle.getTypeID(vehicle_id) != "ambulance":
                continue

            # Get complete route
            route = traci.vehicle.getRoute(vehicle_id)

            if len(route) < 2:
                continue

            # Original incoming direction
            direction = route[0]

            # Destination edge after the intersection
            outgoing_edge = route[1]

            if direction not in INCOMING_EDGES:
                continue

            road_id = traci.vehicle.getRoadID(vehicle_id)

            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)

            ambulance_info.append({

                "id": vehicle_id,
                "direction": direction,
                "outgoing_edge": outgoing_edge,
                "road_id": road_id,
                "waiting_time": waiting_time

            })

        return ambulance_info

    # =====================================
    # 4. AMBULANCE PRIORITY LOCK
    # =====================================

    def get_priority_ambulance(self):

        ambulance_info = self.get_ambulance_info()

        current_ambulances = {
            ambulance["id"]: ambulance
            for ambulance in ambulance_info
        }

        # Check previously prioritized ambulances
        for ambulance_id in list(self.priority_ambulances):

            # Ambulance has left the simulation
            if ambulance_id not in current_ambulances:

                del self.priority_ambulances[ambulance_id]

                print(
                    "Ambulance priority released:",
                    ambulance_id
                )

                continue

            ambulance = current_ambulances[ambulance_id]

            # Ambulance has reached its outgoing road
            if ambulance["road_id"] == ambulance["outgoing_edge"]:

                del self.priority_ambulances[ambulance_id]

                print(
                    "Ambulance cleared intersection:",
                    ambulance_id
                )

        # Activate priority after 15 seconds of waiting
        for ambulance in ambulance_info:

            if ambulance["waiting_time"] >= AMBULANCE_WAIT_THRESHOLD:

                if ambulance["id"] not in self.priority_ambulances:

                    self.priority_ambulances[ambulance["id"]] = (
                        ambulance["direction"]
                    )

                    print(
                        "Ambulance priority activated:",
                        ambulance["id"],
                        "Direction:",
                        ambulance["direction"]
                    )

        # Return an ambulance that still has priority
        if self.priority_ambulances:

            ambulance_id = next(iter(self.priority_ambulances))

            return {
                "id": ambulance_id,
                "direction": self.priority_ambulances[ambulance_id]
            }

        return None

    # =====================================
    # 5. ADVANCE SIMULATION
    # =====================================

    def advance_simulation(self, duration, metrics=None):

        for _ in range(duration):

            if traci.simulation.getTime() >= MAX_EPISODE_TIME:
                break

            if traci.simulation.getMinExpectedNumber() == 0:
                break

            traci.simulationStep()

            if metrics is not None:
                metrics.collect()

    # =====================================
    # 6. SWITCH SIGNAL SAFELY
    # =====================================

    def switch_to_green(self, target_green, metrics=None):

        if self.current_phase == target_green:
            return True

        # NS green -> EW green requires NS yellow
        if self.current_phase == NORTH_SOUTH_GREEN:

            yellow_phase = NORTH_SOUTH_YELLOW

        # EW green -> NS green requires EW yellow
        else:

            yellow_phase = EAST_WEST_YELLOW

        # First activate yellow
        traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            yellow_phase
        )

        traci.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            YELLOW_DURATION
        )

        # Complete yellow phase
        self.advance_simulation(
            YELLOW_DURATION,
            metrics
        )

        if (
            traci.simulation.getTime() >= MAX_EPISODE_TIME
            or traci.simulation.getMinExpectedNumber() == 0
        ):
            return False

        # Activate required green
        traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            target_green
        )

        traci.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            GREEN_DURATION
        )

        self.current_phase = target_green

        return True

    # =====================================
    # 7. MONITOR TRAFFIC AND AMBULANCE
    # =====================================

    def advance_green_with_priority(self, duration, metrics=None):

        elapsed = 0

        while elapsed < duration:

            if traci.simulation.getTime() >= MAX_EPISODE_TIME:
                break

            if traci.simulation.getMinExpectedNumber() == 0:
                break

            # Advance simulation by one second
            traci.simulationStep()

            elapsed += 1

            if metrics is not None:
                metrics.collect()

            # Check ambulance priority every second
            ambulance = self.get_priority_ambulance()

            if ambulance is not None:

                # Mark this RL step as affected by ambulance priority
                self.emergency_override_during_step = True

                direction = ambulance["direction"]

                # North/South ambulance
                if direction in ["N_in", "S_in"]:

                    emergency_green = NORTH_SOUTH_GREEN

                # East/West ambulance
                else:

                    emergency_green = EAST_WEST_GREEN

                # Override RL if necessary
                if self.current_phase != emergency_green:

                    print(
                        "AMBULANCE PRIORITY:",
                        ambulance["id"],
                        "Direction:",
                        direction
                    )

                    switched = self.switch_to_green(
                        emergency_green,
                        metrics
                    )

                    if not switched:
                        break

    # =====================================
    # 8. RL STEP
    # =====================================

    def step(self, action, metrics=None):

        if action not in [0, 1]:

            raise ValueError("Action must be 0 or 1")

        if not self.is_running:

            raise RuntimeError(
                "Environment is not running. Call reset() first."
            )

        if traci.simulation.getTime() >= MAX_EPISODE_TIME:

            raise RuntimeError(
                "Episode has ended. Call reset() to start again."
            )

        # Reset override flag for this RL step
        self.emergency_override_during_step = False

        # RL-selected signal
        if action == 0:

            target_green = NORTH_SOUTH_GREEN

        else:

            target_green = EAST_WEST_GREEN

        # Check active ambulance priority
        ambulance = self.get_priority_ambulance()

        if ambulance is not None:

            # RL action is overridden by ambulance priority
            self.emergency_override_during_step = True

            direction = ambulance["direction"]

            if direction in ["N_in", "S_in"]:

                target_green = NORTH_SOUTH_GREEN

            else:

                target_green = EAST_WEST_GREEN

            print(
                "AMBULANCE PRIORITY ACTIVE:",
                ambulance["id"],
                "Direction:",
                direction
            )

        # Switch through yellow if necessary
        switched = self.switch_to_green(
            target_green,
            metrics
        )

        if not switched:

            next_state = self.get_state()
            reward = self.calculate_reward()

            return next_state, reward, True

        # Keep selected green active
        traci.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            GREEN_DURATION
        )

        # Monitor traffic and ambulance every second
        self.advance_green_with_priority(
            GREEN_DURATION,
            metrics
        )

        next_state = self.get_state()
        reward = self.calculate_reward()

        done = (
            traci.simulation.getTime() >= MAX_EPISODE_TIME
            or traci.simulation.getMinExpectedNumber() == 0
        )

        return next_state, reward, done

    # =====================================
    # 9. CALCULATE REWARD
    # =====================================

    def calculate_reward(self):

        total_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in INCOMING_EDGES
        )

        return -total_queue

    # =====================================
    # 10. CLOSE ENVIRONMENT
    # =====================================

    def close(self):

        if self.is_running:

            traci.close()
            self.is_running = False
