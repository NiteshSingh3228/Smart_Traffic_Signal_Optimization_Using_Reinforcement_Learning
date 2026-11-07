
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

NORTH_SOUTH_GREEN = 0
NORTH_SOUTH_YELLOW = 1
EAST_WEST_GREEN = 2
EAST_WEST_YELLOW = 3

YELLOW_DURATION = 3
GREEN_DURATION = 30
MAX_EPISODE_TIME = 600
AMBULANCE_WAIT_THRESHOLD = 15

# Default 1-intersection fallback definitions
TRAFFIC_LIGHT_ID = "center"
INCOMING_EDGES = ["N_in", "S_in", "E_in", "W_in"]


def _resolve_sumo_binary(gui):
    """
    Dynamically resolve the SUMO binary.
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
    """
    Reinforcement Learning Traffic Environment with support for:
    - 2-Intersection Arterial Corridor ('corridor_2'): J1 (West), J2 (East)
    - Single Isolated Intersection ('single'): center
    """

    def __init__(self, use_gui=False, max_episode_time=MAX_EPISODE_TIME, network_mode="corridor_2"):

        self.network_mode = network_mode
        self._configure_network(network_mode)

        self.current_phase = NORTH_SOUTH_GREEN
        self.tl_phases = {tl: NORTH_SOUTH_GREEN for tl in self.traffic_light_ids}
        self.sumo_binary = _resolve_sumo_binary(use_gui)
        self.use_gui = use_gui
        self.max_episode_time = (
            max_episode_time if max_episode_time is not None else float("inf")
        )
        self.is_running = False

        # Remember ambulances that have received priority
        self.priority_ambulances = {}

        # Track whether ambulance priority affected the current RL step
        self.emergency_override_during_step = False

    def _configure_network(self, network_mode):
        """Sets network files, traffic light IDs, and edge groups."""
        if network_mode == "corridor_2":
            self.net_file = "sumo/network/corridor_2.net.xml"
            self.route_file = "sumo/routes/corridor_2.rou.xml"
            self.traffic_light_ids = ["J1", "J2"]
            self.direction_edges = {
                "north": ["N1_in", "N2_in"],
                "south": ["S1_in", "S2_in"],
                "east": ["E_in", "J2_J1"],
                "west": ["W_in", "J1_J2"]
            }
            self.monitored_incoming_edges = [
                "N1_in", "S1_in", "W_in", "J2_J1",
                "N2_in", "S2_in", "E_in", "J1_J2"
            ]
            self.edge_to_junction = {
                "N1_in": "J1", "S1_in": "J1", "W_in": "J1", "J2_J1": "J1",
                "N2_in": "J2", "S2_in": "J2", "E_in": "J2", "J1_J2": "J2"
            }
        elif network_mode == "corridor_3":
            self.net_file = "sumo/network/corridor_3.net.xml"
            self.route_file = "sumo/routes/corridor_3.rou.xml"
            self.traffic_light_ids = ["J1", "J2", "J3"]
            self.direction_edges = {
                "north": ["N1_in", "N2_in", "N3_in"],
                "south": ["S1_in", "S2_in", "S3_in"],
                "east": ["E_in", "J3_J2", "J2_J1"],
                "west": ["W_in", "J1_J2", "J2_J3"]
            }
            self.monitored_incoming_edges = [
                "N1_in", "N2_in", "N3_in",
                "S1_in", "S2_in", "S3_in",
                "E_in", "W_in",
                "J1_J2", "J2_J1", "J2_J3", "J3_J2"
            ]
            self.edge_to_junction = {
                "N1_in": "J1", "S1_in": "J1", "W_in": "J1", "J2_J1": "J1",
                "N2_in": "J2", "S2_in": "J2", "J1_J2": "J2", "J3_J2": "J2",
                "N3_in": "J3", "S3_in": "J3", "E_in": "J3", "J2_J3": "J3"
            }
        else:
            self.net_file = "sumo/network/intersection.net.xml"
            self.route_file = "sumo/routes/routes.rou.xml"
            self.traffic_light_ids = ["center"]
            self.direction_edges = {
                "north": ["N_in"],
                "south": ["S_in"],
                "east": ["E_in"],
                "west": ["W_in"]
            }
            self.monitored_incoming_edges = ["N_in", "S_in", "E_in", "W_in"]
            self.edge_to_junction = {
                "N_in": "center", "S_in": "center", "E_in": "center", "W_in": "center"
            }

    def _get_cardinal_direction(self, edge):
        for card, edges in self.direction_edges.items():
            if edge in edges:
                return card
        return None

    def _set_traffic_lights_phase(self, phase, duration=None, target_tls=None):
        tls = target_tls if target_tls is not None else self.traffic_light_ids
        for tl in tls:
            try:
                traci.trafficlight.setPhase(tl, phase)
                if duration is not None:
                    traci.trafficlight.setPhaseDuration(tl, duration)
                self.tl_phases[tl] = phase
            except Exception:
                pass

    # =====================================
    # 1. RESET ENVIRONMENT
    # =====================================

    def reset(self, seed=42):

        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            self.is_running = False

        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--seed", str(seed)
        ]

        if self.use_gui:
            sumo_cmd.extend(["--delay", "100", "--start"])

        traci.start(sumo_cmd)

        self.is_running = True
        self.current_phase = NORTH_SOUTH_GREEN
        self.tl_phases = {tl: NORTH_SOUTH_GREEN for tl in self.traffic_light_ids}
        self.priority_ambulances = {}
        self.emergency_override_during_step = False

        self._set_traffic_lights_phase(NORTH_SOUTH_GREEN, GREEN_DURATION)

        return self.get_state()

    # =====================================
    # 2. GET TRAFFIC STATE (17-Dimensional)
    # =====================================

    def get_state(self):

        state = []

        # 1-8: Normal vehicle count and queue per cardinal direction
        for card in ["north", "south", "east", "west"]:
            edges = self.direction_edges[card]
            vehicle_count = sum(traci.edge.getLastStepVehicleNumber(e) for e in edges)
            queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in edges)
            state.append(vehicle_count)
            state.append(queue_length)

        ambulance_info = self.get_ambulance_info()

        # 9-12: Ambulance count per cardinal direction
        for card in ["north", "south", "east", "west"]:
            amb_count = sum(1 for amb in ambulance_info if amb["cardinal"] == card)
            state.append(amb_count)

        # 13-16: Ambulance waiting time per cardinal direction
        for card in ["north", "south", "east", "west"]:
            amb_wait = max(
                (amb["waiting_time"] for amb in ambulance_info if amb["cardinal"] == card),
                default=0
            )
            state.append(amb_wait)

        # 17: Current traffic signal phase
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

            route = traci.vehicle.getRoute(vehicle_id)
            if len(route) < 2:
                continue

            road_id = traci.vehicle.getRoadID(vehicle_id)

            if road_id.startswith(":"):
                continue

            # If ambulance reached the final destination edge
            if road_id == route[-1]:
                continue

            cardinal = self._get_cardinal_direction(road_id)
            if cardinal is None:
                cardinal = self._get_cardinal_direction(route[0])
            if cardinal is None:
                continue

            target_tl = self.edge_to_junction.get(road_id, self.traffic_light_ids[0])
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)

            ambulance_info.append({
                "id": vehicle_id,
                "direction": route[0],
                "cardinal": cardinal,
                "road_id": road_id,
                "target_tl": target_tl,
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
            if ambulance_id not in current_ambulances:
                del self.priority_ambulances[ambulance_id]
                continue

        # Activate priority after threshold seconds of waiting
        for ambulance in ambulance_info:
            if ambulance["waiting_time"] >= AMBULANCE_WAIT_THRESHOLD:
                if ambulance["id"] not in self.priority_ambulances:
                    self.priority_ambulances[ambulance["id"]] = ambulance
                    return ambulance

        if self.priority_ambulances:
            return next(iter(self.priority_ambulances.values()))

        return None

    # =====================================
    # 5. ADVANCE SIMULATION
    # =====================================

    def _step_simulation_and_record(self, metrics=None):
        """Advance simulation by 1 second and record instantaneous queue and ambulance wait."""
        traci.simulationStep()

        if metrics is not None:
            metrics.collect()

        current_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in self.monitored_incoming_edges
        )

        ambulance_wait = sum(
            ambulance["waiting_time"]
            for ambulance in self.get_ambulance_info()
        )

        arrived = len(traci.simulation.getArrivedIDList())

        return current_queue, ambulance_wait, arrived

    def advance_simulation(self, duration, metrics=None, step_stats=None):

        for _ in range(duration):

            if traci.simulation.getTime() >= self.max_episode_time:
                break

            queue, amb_wait, arrived = self._step_simulation_and_record(metrics)

            if step_stats is not None:
                step_stats["queue"] += queue
                step_stats["amb"] += amb_wait
                step_stats["completed"] += arrived
                step_stats["seconds"] += 1

    # =====================================
    # 6. SWITCH SIGNAL SAFELY
    # =====================================

    def switch_to_green(self, target_green, metrics=None, step_stats=None, target_tls=None):

        tls = target_tls if target_tls is not None else self.traffic_light_ids

        # Check if all targeted lights are already at target_green
        all_match = all(self.tl_phases.get(tl, self.current_phase) == target_green for tl in tls)
        if all_match:
            return True

        # Activate yellow for transition
        if self.current_phase == NORTH_SOUTH_GREEN:
            yellow_phase = NORTH_SOUTH_YELLOW
        else:
            yellow_phase = EAST_WEST_YELLOW

        self._set_traffic_lights_phase(yellow_phase, YELLOW_DURATION, target_tls=tls)

        self.advance_simulation(
            YELLOW_DURATION,
            metrics,
            step_stats
        )

        if traci.simulation.getTime() >= self.max_episode_time:
            return False

        self._set_traffic_lights_phase(target_green, GREEN_DURATION, target_tls=tls)
        self.current_phase = target_green

        return True

    # =====================================
    # 7. MONITOR TRAFFIC AND AMBULANCE
    # =====================================

    def advance_green_with_priority(self, duration, metrics=None, step_stats=None):

        elapsed = 0

        while elapsed < duration:

            if traci.simulation.getTime() >= self.max_episode_time:
                break

            priority_ambulance = self.get_priority_ambulance()

            if priority_ambulance:

                cardinal = priority_ambulance.get("cardinal")
                target_tl = priority_ambulance.get("target_tl", self.traffic_light_ids[0])

                if cardinal in ["north", "south"]:
                    required_phase = NORTH_SOUTH_GREEN
                else:
                    required_phase = EAST_WEST_GREEN

                # Only pre-empt the junction where the ambulance is waiting
                current_tl_phase = self.tl_phases.get(target_tl, self.current_phase)

                if current_tl_phase != required_phase:

                    self.emergency_override_during_step = True

                    switched = self.switch_to_green(
                        required_phase,
                        metrics,
                        step_stats,
                        target_tls=[target_tl]
                    )

                    if not switched:
                        break

                    break

            queue, amb_wait, arrived = self._step_simulation_and_record(metrics)

            if step_stats is not None:
                step_stats["queue"] += queue
                step_stats["amb"] += amb_wait
                step_stats["completed"] += arrived
                step_stats["seconds"] += 1

            elapsed += 1

    # =====================================
    # 8. STEP ENVIRONMENT
    # =====================================

    def step(self, action, metrics=None):

        step_stats = {
            "queue": 0.0,
            "amb": 0.0,
            "completed": 0,
            "seconds": 0
        }

        self.emergency_override_during_step = False

        if action == 0:
            target_green = NORTH_SOUTH_GREEN
        else:
            target_green = EAST_WEST_GREEN

        switched = self.switch_to_green(
            target_green,
            metrics,
            step_stats
        )

        if not switched:

            next_state = self.get_state()
            reward = self.calculate_reward(
                step_stats["queue"],
                step_stats["amb"],
                step_stats["completed"],
                step_stats["seconds"]
            )

            return next_state, reward, True

        self._set_traffic_lights_phase(self.current_phase, GREEN_DURATION)

        self.advance_green_with_priority(
            GREEN_DURATION,
            metrics,
            step_stats
        )

        next_state = self.get_state()
        reward = self.calculate_reward(
            step_stats["queue"],
            step_stats["amb"],
            step_stats["completed"],
            step_stats["seconds"]
        )

        done = traci.simulation.getTime() >= self.max_episode_time

        return next_state, reward, done

    # =====================================
    # 9. CALCULATE REWARD (Combined -ve and +ve)
    # =====================================

    def calculate_reward(self, step_queue_total=None, step_amb_total=None, step_completed=None, step_seconds=None):
        """
        Combined (-ve and +ve) Reward:
        Reward = + (Throughput Bonus) - (Average Queue Length) - (Ambulance Delay Penalty)
        - When traffic clears smoothly: Reward is POSITIVE (> 0).
        - When traffic is congested or emergency vehicle blocked: Reward is NEGATIVE (< 0).
        """
        if step_seconds and step_seconds > 0:

            avg_queue = step_queue_total / step_seconds
            avg_amb_wait = (step_amb_total / step_seconds) if step_amb_total else 0.0
            ambulance_penalty = min(avg_amb_wait * 1.5, 15.0)

            completed = step_completed if step_completed is not None else 0
            throughput_bonus = float(completed) * 2.5

            return round(throughput_bonus - (avg_queue + ambulance_penalty), 2)

        total_queue = sum(
            traci.edge.getLastStepHaltingNumber(edge)
            for edge in self.monitored_incoming_edges
        )

        ambulance_penalty = sum(
            min(ambulance["waiting_time"] * 1.5, 15.0)
            for ambulance in self.get_ambulance_info()
        )

        arrived = len(traci.simulation.getArrivedIDList())
        return round((arrived * 2.5) - (total_queue + ambulance_penalty), 2)

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
