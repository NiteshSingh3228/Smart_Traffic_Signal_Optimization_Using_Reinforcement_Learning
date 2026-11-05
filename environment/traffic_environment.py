
import sys
import os
import shutil

# Ensure traci can be found even if python is invoked without activating .venv
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

# FIX 1.2: Standardize to 600s to match fixed_time_controller.py
MAX_EPISODE_TIME = 600

AMBULANCE_WAIT_THRESHOLD = 15


def _resolve_sumo_binary(gui):
    """
    FIX 3.1: Dynamically resolve the SUMO binary.
    Prefers the macOS framework path if it exists,
    otherwise falls back to system PATH.
    """
    name = "sumo-gui" if gui else "sumo"
    framework = (
        "/Library/Frameworks/EclipseSUMO.framework"
        "/Versions/1.28.0/EclipseSUMO/bin/" + name
    )
    if os.path.exists(framework):
        return framework
    resolved = shutil.which(name)
    return resolved if resolved else name


class TrafficEnvironment:

    def __init__(self, use_gui=False):

        self.current_phase = NORTH_SOUTH_GREEN
        self.sumo_binary = _resolve_sumo_binary(use_gui)
        self.use_gui = use_gui
        self.is_running = False

        # Remember ambulances that have received priority
        self.priority_ambulances = {}

        # Track whether ambulance priority affected the current RL step
        self.emergency_override_during_step = False

    # =====================================
    # 1. RESET ENVIRONMENT
    # =====================================

    def reset(self, seed=42):

        # Close any existing connection before starting a new one
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            self.is_running = False

        sumo_cmd = [
            self.sumo_binary,
            "-n", "sumo/network/intersection.net.xml",
            "-r", "sumo/routes/routes.rou.xml",
            "--no-step-log", "true",
            "--seed", str(seed)
        ]

        if self.use_gui:
            sumo_cmd.extend(["--delay", "200"])

        traci.start(sumo_cmd)

        self.is_running = True
        self.current_phase = NORTH_SOUTH_GREEN
        self.priority_ambulances = {}
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
    # Only report ambulances still on incoming edges (not yet crossed)
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

            # FIX: Skip ambulances that have already crossed the intersection
            # (they are on the outgoing edge or an internal edge)
            if road_id == outgoing_edge:
                continue

            # Also skip ambulances that are on internal junction edges
            # (they are actively passing through)
            if road_id.startswith(":"):
                continue

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

            # Ambulance has left the simulation or already crossed
            if ambulance_id not in current_ambulances:

                del self.priority_ambulances[ambulance_id]
                continue

        # Activate priority after threshold seconds of waiting
        for ambulance in ambulance_info:

            if ambulance["waiting_time"] >= AMBULANCE_WAIT_THRESHOLD:

                if ambulance["id"] not in self.priority_ambulances:

                    self.priority_ambulances[ambulance["id"]] = (
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

        if traci.simulation.getTime() >= MAX_EPISODE_TIME:
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

        done = traci.simulation.getTime() >= MAX_EPISODE_TIME

        return next_state, reward, done

    # =====================================
    # 9. CALCULATE REWARD
    # FIX 2.1: Include ambulance waiting penalty
    # =====================================

    def calculate_reward(self):

        total_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in INCOMING_EDGES
        )

        # Additional penalty for ambulances waiting
        # (weight 5.0 so ambulance waiting is strongly discouraged)
        ambulance_penalty = sum(
            ambulance["waiting_time"] * 5.0
            for ambulance in self.get_ambulance_info()
        )

        return -(total_queue + ambulance_penalty)

    # =====================================
    # 10. CLOSE ENVIRONMENT
    # =====================================

    def close(self):

        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_running = False
