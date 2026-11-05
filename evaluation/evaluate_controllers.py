
import csv
import os
import traci

from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment
from evaluation.metrics import TrafficMetrics
from evaluation.fixed_time_controller import run_fixed_time_controller


# =====================================
# FIXED-TIME EVALUATION
# =====================================

def evaluate_fixed_time(ambulance_priority=False):

    if ambulance_priority:
        print("\n========== FIXED-TIME + AMBULANCE PRIORITY EVALUATION ==========")
    else:
        print("\n========== FIXED-TIME EVALUATION ==========")

    sumo_cmd = [
        "sumo",
        "-n", "sumo/network/intersection.net.xml",
        "-r", "sumo/routes/routes.rou.xml",
        "--no-step-log", "true",
        "--seed", "42"
    ]

    traci.start(sumo_cmd)

    try:
        results = run_fixed_time_controller(
            ambulance_priority=ambulance_priority
        )

    finally:
        traci.close()

    return results


# =====================================
# Q-LEARNING EVALUATION
# =====================================

def evaluate_q_learning():

    print("\n========== Q-LEARNING EVALUATION ==========")

    agent = QLearningAgent()
    agent.load_model("q_learning_ambulance_model.pkl")
    agent.epsilon = 0

    env = TrafficEnvironment(use_gui=False)
    metrics = TrafficMetrics()

    try:

        state = env.reset(seed=42)
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

def evaluate_dqn():

    print("\n========== DQN EVALUATION ==========")

    agent = DQNAgent()
    agent.load_model("dqn_ambulance_model.pth")
    agent.epsilon = 0

    env = TrafficEnvironment(use_gui=False)
    metrics = TrafficMetrics()

    try:

        state = env.reset(seed=42)
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
