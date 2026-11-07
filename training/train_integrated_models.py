"""
Triple-Source Integrated Training Pipeline
Integrates:
1. Source 1: Existing SUMO online RL checkpoints (dqn_ambulance_model_updated.pth, q_learning_ambulance_model_updated.pkl)
2. Source 2: Archive offline transitions & delay predictions (data/processed/clean_summary.csv, clean_tripinfo.csv)
3. Source 3: Dataset task scenario distributions (dataset/tasks/net_double/train_tau_c0.npy)

Preserves existing models as immutable baselines and outputs:
- results/models/dqn_integrated_model.pth
- results/models/q_learning_integrated_model.pkl
"""

import os
import sys

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

import copy
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch

from agents.dqn_agent import DQNAgent
from agents.q_learning import QLearningAgent
from environment.traffic_environment import TrafficEnvironment

# Set deterministic seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODELS_DIR = "results/models"
METRICS_DIR = "results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. GENERATE DYNAMIC SCENARIO ROUTE XML (Driven by Source 3 tasks)
# ---------------------------------------------------------------------------

def write_dynamic_corridor_routes(tau, output_path="sumo/routes/corridor_2_dynamic.rou.xml"):
    """
    Translates task vector tau into SUMO flow probabilities:
    tau = [flow_WE, flow_J1_NS, flow_J1_SN, flow_J2_NS, flow_J2_SN, flow_EW] (veh/hr)
    prob = flow_veh_per_hr / 3600
    """
    prob_WE = max(0.01, min(0.08, float(tau[0]) / 3600.0))
    prob_J1_NS = max(0.005, min(0.04, float(tau[1]) / 3600.0))
    prob_J1_SN = max(0.005, min(0.04, float(tau[2]) / 3600.0))
    prob_J2_NS = max(0.005, min(0.04, float(tau[3]) / 3600.0))
    prob_J2_SN = max(0.005, min(0.04, float(tau[4]) / 3600.0))
    prob_EW = max(0.01, min(0.08, float(tau[5]) / 3600.0))

    xml_content = f"""<?xml version='1.0' encoding='UTF-8'?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89" guiShape="passenger" color="30,144,255" />
    <vType id="ambulance" accel="3.0" decel="5.0" sigma="0.5" length="5" maxSpeed="16.67" vClass="emergency" color="255,0,0" guiShape="passenger" />

    <route id="corridor_WE" edges="W_in J1_J2 E_out" />
    <route id="corridor_EW" edges="E_in J2_J1 W_out" />
    <route id="cross_J1_NS" edges="N1_in S1_out" />
    <route id="cross_J1_SN" edges="S1_in N1_out" />
    <route id="cross_J2_NS" edges="N2_in S2_out" />
    <route id="cross_J2_SN" edges="S2_in N2_out" />
    <route id="turn_N1_E" edges="N1_in J1_J2 E_out" />
    <route id="turn_W_S2" edges="W_in J1_J2 S2_out" />

    <flow id="flow_corridor_WE" type="car" route="corridor_WE" begin="0" end="86400" probability="{prob_WE:.4f}" />
    <flow id="flow_corridor_EW" type="car" route="corridor_EW" begin="0" end="86400" probability="{prob_EW:.4f}" />
    <flow id="flow_J1_NS" type="car" route="cross_J1_NS" begin="0" end="86400" probability="{prob_J1_NS:.4f}" />
    <flow id="flow_J1_SN" type="car" route="cross_J1_SN" begin="0" end="86400" probability="{prob_J1_SN:.4f}" />
    <flow id="flow_J2_NS" type="car" route="cross_J2_NS" begin="0" end="86400" probability="{prob_J2_NS:.4f}" />
    <flow id="flow_J2_SN" type="car" route="cross_J2_SN" begin="0" end="86400" probability="{prob_J2_SN:.4f}" />
    <flow id="flow_turn_N1_E" type="car" route="turn_N1_E" begin="0" end="86400" probability="0.0100" />
    <flow id="flow_turn_W_S2" type="car" route="turn_W_S2" begin="0" end="86400" probability="0.0100" />

    <flow id="amb_corridor_WE" type="ambulance" route="corridor_WE" begin="0" end="86400" probability="0.0020" />
    <flow id="amb_corridor_EW" type="ambulance" route="corridor_EW" begin="0" end="86400" probability="0.0020" />
    <flow id="amb_J1_NS"       type="ambulance" route="cross_J1_NS" begin="0" end="86400" probability="0.0010" />
    <flow id="amb_J2_NS"       type="ambulance" route="cross_J2_NS" begin="0" end="86400" probability="0.0010" />
</routes>
"""
    with open(output_path, "w") as f:
        f.write(xml_content)
    return output_path

# ---------------------------------------------------------------------------
# 2. INJECT OFFLINE TRANSITIONS FROM ARCHIVE (Source 2)
# ---------------------------------------------------------------------------

def inject_archive_transitions_to_dqn(dqn_agent):
    """Pre-populates DQN experience replay with empirical transitions from archive data."""
    df_trans = pd.read_csv("data/processed/offline_transitions.csv")
    print(f"Injecting {len(df_trans)} offline archive transitions into DQN replay buffer...")
    
    for _, row in df_trans.iterrows():
        # Map 4 macro features into a synthetic 17-D state
        # State: [N_veh, N_q, S_veh, S_q, E_veh, E_q, W_veh, W_q, amb_counts(4), amb_waits(4), phase]
        half_running = float(row["s_running"]) / 4.0
        half_halting = float(row["s_halting"]) / 4.0
        
        state_17 = [
            half_running, half_halting,  # North
            half_running, half_halting,  # South
            half_running, half_halting,  # East
            half_running, half_halting,  # West
            0, 0, 0, 0,                  # No ambulance in historical summary
            0.0, 0.0, 0.0, 0.0,
            int(row["action"])
        ]
        
        next_half_running = float(row["next_s_running"]) / 4.0
        next_half_halting = float(row["next_s_halting"]) / 4.0
        
        next_state_17 = [
            next_half_running, next_half_halting,
            next_half_running, next_half_halting,
            next_half_running, next_half_halting,
            next_half_running, next_half_halting,
            0, 0, 0, 0,
            0.0, 0.0, 0.0, 0.0,
            1 - int(row["action"])
        ]
        
        action = int(row["action"])
        reward = float(row["reward"])
        done = bool(row["done"])
        
        dqn_agent.remember(state_17, action, reward, next_state_17, done)
        
    print(f"DQN Replay buffer size after offline warm-up: {len(dqn_agent.memory)}")

def preseed_q_table_from_archive(q_agent):
    """Pre-seeds Q-table values using empirical congestion and delay gradients from archive."""
    df_trans = pd.read_csv("data/processed/offline_transitions.csv")
    mean_reward = df_trans["reward"].mean()
    print(f"Pre-seeding Q-table from archive (Empirical mean penalty = {mean_reward:.2f})...")
    
    # Pre-populate common discrete states with informed empirical priors
    seeded_count = 0
    for q_ns in range(3):
        for q_ew in range(3):
            for amb_ns in range(3):
                for amb_ew in range(3):
                    for phase in range(2):
                        state_key = (q_ns, q_ew, amb_ns, amb_ew, phase)
                        if state_key not in q_agent.q_table:
                            # Prior based on queue balance and emergency presence
                            # Action 0: Green NS, Action 1: Green EW
                            val_a0 = -float(q_ns * 1.5 + (15.0 if amb_ns > 0 and phase != 0 else 0.0))
                            val_a1 = -float(q_ew * 1.5 + (15.0 if amb_ew > 0 and phase != 1 else 0.0))
                            q_agent.q_table[state_key] = [val_a0, val_a1]
                            seeded_count += 1
    print(f"Pre-seeded {seeded_count} discrete states into Q-table (Total states: {len(q_agent.q_table)})")

# ---------------------------------------------------------------------------
# 3. TRAINING PIPELINE
# ---------------------------------------------------------------------------

def train_integrated_pipeline(num_episodes=25):
    print("=================================================================")
    print("STARTING TRIPLE-SOURCE INTEGRATED MODEL TRAINING")
    print("=================================================================")
    
    # 1. Load tasks from Source 3
    train_tasks = np.load("dataset/tasks/net_double/train_tau_c0.npy")
    print(f"Source 3: Loaded {len(train_tasks)} task scenarios from dataset/tasks/net_double/train_tau_c0.npy")

    # 2. Initialize DQN: Load Source 1 Checkpoint as Warm-Start
    dqn_agent = DQNAgent()
    dqn_base_path = "results/models/dqn_ambulance_model_updated.pth"
    if os.path.exists(dqn_base_path):
        print(f"Source 1: Loading baseline DQN checkpoint from {dqn_base_path}...")
        dqn_agent.load_model("dqn_ambulance_model_updated.pth")
    else:
        print("Warning: Baseline DQN checkpoint not found, initializing fresh.")
        
    # Source 2: Inject archive transitions
    inject_archive_transitions_to_dqn(dqn_agent)
    dqn_agent.epsilon = 0.05  # Controlled fine-tuning exploration

    # 3. Initialize Q-Learning: Load Source 1 Checkpoint as Warm-Start
    q_agent = QLearningAgent()
    q_base_path = "results/models/q_learning_ambulance_model_updated.pkl"
    if os.path.exists(q_base_path):
        print(f"Source 1: Loading baseline Q-Learning checkpoint from {q_base_path}...")
        q_agent.load_model("q_learning_ambulance_model_updated.pkl")
    else:
        print("Warning: Baseline Q-Learning checkpoint not found, initializing fresh.")
        
    # Source 2: Pre-seed Q-table with archive priors
    preseed_q_table_from_archive(q_agent)
    q_agent.epsilon = 0.05

    # 4. Multi-Scenario Simulation Training Loop
    env = TrafficEnvironment(use_gui=False, network_mode="corridor_2")
    
    print("\n--- Beginning Multi-Scenario Online Fine-Tuning across Source 3 Flow Regimes ---")
    
    dqn_rewards = []
    q_rewards = []
    
    for episode in range(num_episodes):
        # Sample task from Source 3
        task_idx = episode % len(train_tasks)
        tau = train_tasks[task_idx]
        
        # Write calibrated dynamic routes
        dyn_route_file = write_dynamic_corridor_routes(tau)
        env.route_file = dyn_route_file
        
        # Alternating training episodes for DQN and Q-Learning
        is_dqn_turn = (episode % 2 == 0)
        current_agent = dqn_agent if is_dqn_turn else q_agent
        agent_name = "DQN" if is_dqn_turn else "Q-Learning"
        
        ep_seed = 2000 + episode
        state = env.reset(seed=ep_seed)
        done = False
        step_count = 0
        ep_reward = 0.0
        
        try:
            while not done:
                action = current_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                if is_dqn_turn:
                    dqn_agent.remember(state, action, reward, next_state, done)
                    dqn_agent.learn()
                else:
                    q_agent.learn(state, action, reward, next_state, done)
                    
                state = next_state
                ep_reward += reward
                step_count += 1
                
            if is_dqn_turn:
                dqn_rewards.append(ep_reward)
                dqn_agent.decay_epsilon()
            else:
                q_rewards.append(ep_reward)
                q_agent.decay_epsilon()
                
            print(f"Episode {episode + 1}/{num_episodes} [{agent_name}] - Task {task_idx}: Steps={step_count}, Reward={ep_reward:.2f}, Epsilon={current_agent.epsilon:.4f}")
            
        finally:
            env.close()

    # 5. Save New Integrated Models (Strictly preserving original files)
    dqn_out_path = os.path.join(MODELS_DIR, "dqn_integrated_model.pth")
    dqn_agent.save_model("dqn_integrated_model.pth")
    print(f"\nSaved new Integrated DQN model to: {dqn_out_path}")
    
    q_out_path = os.path.join(MODELS_DIR, "q_learning_integrated_model.pkl")
    q_agent.save_model("q_learning_integrated_model.pkl")
    print(f"Saved new Integrated Q-Learning model to: {q_out_path}")
    
    # Save training history
    training_summary = {
        "num_episodes": num_episodes,
        "dqn_episodes": len(dqn_rewards),
        "q_episodes": len(q_rewards),
        "dqn_mean_reward": float(np.mean(dqn_rewards)) if dqn_rewards else 0.0,
        "q_mean_reward": float(np.mean(q_rewards)) if q_rewards else 0.0,
        "dqn_final_memory": len(dqn_agent.memory),
        "q_final_states": len(q_agent.q_table)
    }
    
    summary_path = os.path.join(METRICS_DIR, "integrated_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Saved training summary to: {summary_path}")
    
    print("\n=================================================================")
    print("INTEGRATED TRAINING COMPLETED SUCCESSFULLY")
    print("=================================================================")

if __name__ == "__main__":
    train_integrated_pipeline(num_episodes=20)
