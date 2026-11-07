
import sys
import csv
import os
import shutil

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

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

from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment
from evaluation.metrics import TrafficMetrics
from evaluation.fixed_time_controller import run_fixed_time_controller


def _get_sumo_binary():
    """Resolve sumo binary path dynamically."""
    framework = (
        "/Library/Frameworks/EclipseSUMO.framework"
        "/Versions/1.28.0/EclipseSUMO/bin/sumo"
    )
    if os.path.exists(framework):
        return framework
    resolved = shutil.which("sumo")
    return resolved if resolved else "sumo"


# =====================================
# FIXED-TIME EVALUATION
# =====================================

def evaluate_fixed_time(ambulance_priority=False, seed=42, network_mode="corridor_2"):

    if ambulance_priority:
        print("\n========== FIXED-TIME + AMBULANCE PRIORITY EVALUATION ==========")
    else:
        print("\n========== FIXED-TIME EVALUATION ==========")

    env = TrafficEnvironment(use_gui=False, network_mode=network_mode)
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)

    try:
        env.reset(seed=seed)
        done = False
        step_cycle = 0

        while not done:
            # Fixed 30s phase alternation (action 0 = NS Green, action 1 = EW Green)
            action = 0 if (step_cycle < 1) else 1
            step_cycle = (step_cycle + 1) % 2

            next_state, reward, done = env.step(action, metrics=metrics)

        results = metrics.get_results()

    finally:
        env.close()

    return results


# =====================================
# Q-LEARNING EVALUATION
# =====================================

def evaluate_q_learning(seed=42):

    print("\n========== Q-LEARNING EVALUATION ==========")

    agent = QLearningAgent()
    agent.load_model("q_learning_ambulance_model_updated.pkl")
    agent.epsilon = 0

    env = TrafficEnvironment(use_gui=False)
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)

    try:

        state = env.reset(seed=seed)
        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(
                action,
                metrics=metrics
            )

            state = next_state

        results = metrics.get_results()

    finally:
        env.close()

    return results


# =====================================
# DQN EVALUATION
# =====================================

def evaluate_dqn(seed=42):

    print("\n========== DQN EVALUATION ==========")

    agent = DQNAgent()
    agent.load_model("dqn_ambulance_model_updated.pth")
    agent.epsilon = 0

    env = TrafficEnvironment(use_gui=False)
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)

    try:

        state = env.reset(seed=seed)
        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(
                action,
                metrics=metrics
            )

            state = next_state

        results = metrics.get_results()

    finally:
        env.close()

    return results


# =====================================
# SAVE COMPARISON RESULTS
# =====================================

def save_comparison(results):

    os.makedirs("results/metrics", exist_ok=True)

    filepath = "results/metrics/controller_comparison.csv"

    with open(filepath, "w", newline="") as file:

        fieldnames = [
            "Controller",
            "Average Waiting Time",
            "Average Travel Time",
            "Average Queue Length",
            "Completed Vehicles",
            "Observed Vehicles",
            "Average Ambulance Waiting Time",
            "Maximum Ambulance Waiting Time",
            "Ambulances Observed",
            "Simulation Time"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for controller, metrics in results.items():

            writer.writerow({
                "Controller": controller,
                "Average Waiting Time": metrics["average_waiting_time"],
                "Average Travel Time": metrics.get("average_travel_time", 0),
                "Average Queue Length": metrics["average_queue_length"],
                "Completed Vehicles": metrics["completed_vehicles"],
                "Observed Vehicles": metrics["observed_vehicles"],
                "Average Ambulance Waiting Time": metrics["average_ambulance_waiting_time"],
                "Maximum Ambulance Waiting Time": metrics["maximum_ambulance_waiting_time"],
                "Ambulances Observed": metrics["ambulances_observed"],
                "Simulation Time": metrics["simulation_time"]
            })

    print("\nComparison CSV saved:", filepath)


# =====================================
# MAIN EVALUATION
# =====================================

if __name__ == "__main__":

    results = {}

    results["Fixed-Time"] = evaluate_fixed_time(
        ambulance_priority=False
    )

    results["Fixed-Time + Ambulance Priority"] = evaluate_fixed_time(
        ambulance_priority=True
    )

    results["Q-Learning + Ambulance Priority"] = evaluate_q_learning()

    results["DQN + Ambulance Priority"] = evaluate_dqn()

    print("\n========== FINAL COMPARISON ==========")

    for controller, metrics in results.items():

        print(f"\n{controller}")

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    save_comparison(results)

    print("\n========== ALL EVALUATIONS COMPLETED ==========")
