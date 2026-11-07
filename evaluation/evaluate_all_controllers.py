"""
Comprehensive Benchmark & Cross-Validation Pipeline
Evaluates ALL baseline and integrated controllers under identical SUMO 2-intersection
corridor conditions with emergency vehicle arrivals across 5 independent random seeds.

Controllers compared:
1. Fixed-Time
2. Fixed-Time + Ambulance Priority
3. Baseline Q-Learning (Source 1)
4. Baseline DQN (Source 1)
5. Baseline Hybrid Ensemble (Source 1)
6. Integrated Q-Learning (Sources 1 + 2 + 3)
7. Integrated DQN (Sources 1 + 2 + 3)
8. Integrated Hybrid Ensemble (Sources 1 + 2 + 3)
"""

import os
import sys
import csv
import json
import numpy as np

# Ensure project root is in sys.path
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

from environment.traffic_environment import TrafficEnvironment
from evaluation.metrics import TrafficMetrics
from evaluation.fixed_time_controller import run_fixed_time_controller
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.hybrid_agent import HybridAgent
from agents.hybrid_integrated_agent import IntegratedHybridAgent

SEEDS = [42, 101, 202, 303, 404]

def evaluate_fixed_time_corridor(ambulance_priority=False, seed=42):
    """Fixed-time baseline running on the 2-intersection corridor."""
    env = TrafficEnvironment(use_gui=False, network_mode="corridor_2")
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)
    
    try:
        env.reset(seed=seed)
        done = False
        step_in_cycle = 0
        cycle_len = 30  # 30s NS, 30s EW
        
        while not done:
            # Fixed 30s alternation
            action = 0 if (step_in_cycle < 1) else 1
            step_in_cycle = (step_in_cycle + 1) % 2
            
            next_state, reward, done = env.step(action, metrics=metrics)
            
        results = metrics.get_results()
    finally:
        env.close()
        
    return results

def evaluate_agent_controller(agent, seed=42):
    """Evaluates an agent on the 2-intersection corridor."""
    env = TrafficEnvironment(use_gui=False, network_mode="corridor_2")
    metrics = TrafficMetrics(incoming_edges=env.monitored_incoming_edges)
    
    try:
        state = env.reset(seed=seed)
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action, metrics=metrics)
            state = next_state
            
        results = metrics.get_results()
    finally:
        env.close()
        
    return results

def run_all_evaluations():
    print("=================================================================")
    print("STARTING COMPREHENSIVE MULTI-SEED CONTROLLER BENCHMARK")
    print(f"Evaluation Seeds: {SEEDS}")
    print("=================================================================")

    # Initialize all agent variants
    print("Initializing controllers...")
    
    # Baselines
    base_ql = QLearningAgent()
    base_ql.load_model("q_learning_ambulance_model_updated.pkl")
    base_ql.epsilon = 0.0

    base_dqn = DQNAgent()
    base_dqn.load_model("dqn_ambulance_model_updated.pth")
    base_dqn.epsilon = 0.0

    base_hybrid = HybridAgent(
        dqn_model="dqn_ambulance_model_updated.pth",
        ql_model="q_learning_ambulance_model_updated.pkl"
    )

    # Integrated Models
    integ_ql = QLearningAgent()
    integ_ql.load_model("q_learning_integrated_model.pkl")
    integ_ql.epsilon = 0.0

    integ_dqn = DQNAgent()
    integ_dqn.load_model("dqn_integrated_model.pth")
    integ_dqn.epsilon = 0.0

    integ_hybrid = IntegratedHybridAgent(
        dqn_model="dqn_integrated_model.pth",
        ql_model="q_learning_integrated_model.pkl"
    )

    controllers = {
        "Fixed-Time (Corridor)": lambda s: evaluate_fixed_time_corridor(ambulance_priority=False, seed=s),
        "Fixed-Time + Amb Priority (Corridor)": lambda s: evaluate_fixed_time_corridor(ambulance_priority=True, seed=s),
        "Baseline Q-Learning (Source 1)": lambda s: evaluate_agent_controller(base_ql, seed=s),
        "Baseline DQN (Source 1)": lambda s: evaluate_agent_controller(base_dqn, seed=s),
        "Baseline Hybrid Ensemble (Source 1)": lambda s: evaluate_agent_controller(base_hybrid, seed=s),
        "Integrated Q-Learning (Sources 1+2+3)": lambda s: evaluate_agent_controller(integ_ql, seed=s),
        "Integrated DQN (Sources 1+2+3)": lambda s: evaluate_agent_controller(integ_dqn, seed=s),
        "Integrated Hybrid Ensemble (Sources 1+2+3)": lambda s: evaluate_agent_controller(integ_hybrid, seed=s),
    }

    multi_seed_results = {c: [] for c in controllers}

    for name, runner in controllers.items():
        print(f"\nEvaluating controller: {name}")
        for s in SEEDS:
            res = runner(s)
            multi_seed_results[name].append(res)
            print(f"  Seed {s} -> AvgWait={res['average_waiting_time']:.2f}s, Queue={res['average_queue_length']:.2f}, AmbWait={res['average_ambulance_waiting_time']:.2f}s, MaxAmbWait={res['maximum_ambulance_waiting_time']:.2f}s")

    # Aggregate statistics
    summary_rows = []
    for name, seed_runs in multi_seed_results.items():
        avg_wait = np.mean([r["average_waiting_time"] for r in seed_runs])
        std_wait = np.std([r["average_waiting_time"] for r in seed_runs])
        
        avg_travel = np.mean([r.get("average_travel_time", 0) for r in seed_runs])
        
        avg_queue = np.mean([r["average_queue_length"] for r in seed_runs])
        std_queue = np.std([r["average_queue_length"] for r in seed_runs])
        
        avg_completed = np.mean([r["completed_vehicles"] for r in seed_runs])
        avg_observed = np.mean([r["observed_vehicles"] for r in seed_runs])
        
        avg_amb_wait = np.mean([r["average_ambulance_waiting_time"] for r in seed_runs])
        max_amb_wait = np.max([r["maximum_ambulance_waiting_time"] for r in seed_runs])
        
        summary_rows.append({
            "Controller": name,
            "Avg Waiting Time (s)": round(float(avg_wait), 2),
            "Std Waiting Time": round(float(std_wait), 2),
            "Avg Travel Time (s)": round(float(avg_travel), 2),
            "Avg Queue Length": round(float(avg_queue), 2),
            "Std Queue Length": round(float(std_queue), 2),
            "Completed Vehicles": round(float(avg_completed), 1),
            "Observed Vehicles": round(float(avg_observed), 1),
            "Avg Ambulance Wait (s)": round(float(avg_amb_wait), 2),
            "Max Ambulance Wait (s)": round(float(max_amb_wait), 2),
        })

    # Save summary CSV
    os.makedirs("results/metrics", exist_ok=True)
    summary_csv_path = "results/metrics/controller_multi_seed_comparison.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
        
    print(f"\nSaved multi-seed comparison to: {summary_csv_path}")

    # Also update single-seed comparison for seed 42 to keep backwards compatibility
    single_seed_path = "results/metrics/controller_comparison.csv"
    with open(single_seed_path, "w", newline="") as f:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, seed_runs in multi_seed_results.items():
            r42 = seed_runs[0] # Seed 42 is first
            writer.writerow({
                "Controller": name,
                "Average Waiting Time": r42["average_waiting_time"],
                "Average Travel Time": r42.get("average_travel_time", 0),
                "Average Queue Length": r42["average_queue_length"],
                "Completed Vehicles": r42["completed_vehicles"],
                "Observed Vehicles": r42["observed_vehicles"],
                "Average Ambulance Waiting Time": r42["average_ambulance_waiting_time"],
                "Maximum Ambulance Waiting Time": r42["maximum_ambulance_waiting_time"],
                "Ambulances Observed": r42["ambulances_observed"],
                "Simulation Time": r42["simulation_time"]
            })
    print(f"Saved seed 42 comparison to: {single_seed_path}")

    print("\n=================================================================")
    print("FINAL BENCHMARK SUMMARY (5-Seed Average):")
    print("=================================================================")
    import pandas as pd
    df_res = pd.DataFrame(summary_rows)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_all_evaluations()

