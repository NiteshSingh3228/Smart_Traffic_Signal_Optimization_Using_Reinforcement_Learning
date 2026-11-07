"""
Data Preprocessing and Transformation Pipeline
Processes:
1. `archive/dataset_summary.csv`
2. `archive/dataset_tripinfo_raw.csv`
3. `dataset/tasks/net_double/*.npy`

Generates clean, validated datasets in `data/processed/` without modifying originals.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_summary():
    print("[1/4] Preprocessing archive/dataset_summary.csv...")
    df = pd.read_csv("archive/dataset_summary.csv")
    
    # 1. Clean meanTravelTime (-1 for initial time before any vehicle finishes)
    df["meanTravelTime_clean"] = df["meanTravelTime"].apply(lambda x: 0.0 if x < 0 else x)
    
    # 2. Engineer dynamic traffic indicators
    df["delta_halting"] = df["halting"].diff().fillna(0.0)
    df["delta_waiting"] = df["waiting"].diff().fillna(0.0)
    df["congestion_ratio"] = df["halting"] / (df["running"] + 1e-5)
    df["speed_efficiency"] = df["meanSpeedRelative"]
    
    # 3. Target proxy: instantaneous step penalty (queue + delay)
    df["step_penalty"] = -(df["halting"] + df["meanWaitingTime"] * 0.1)
    
    # Save clean summary
    output_path = os.path.join(PROCESSED_DIR, "clean_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"  -> Saved {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df

def preprocess_tripinfo():
    print("[2/4] Preprocessing archive/dataset_tripinfo_raw.csv...")
    df = pd.read_csv("archive/dataset_tripinfo_raw.csv")
    
    # 1. Drop vaporized (100% NaN) and uninformative constant columns
    drop_cols = ["vaporized"]
    for c in ["stopTime", "rerouteNo"]:
        if c in df.columns and df[c].nunique() <= 1:
            drop_cols.append(c)
    df = df.drop(columns=drop_cols)
    
    # 2. Extract approach ID and lane index
    df["approach_id"] = df["departLane"].apply(lambda x: str(x).split("_")[0])
    df["arrival_approach_id"] = df["arrivalLane"].apply(lambda x: str(x).split("_")[0])
    
    # 3. Engineer delay and stop metrics
    df["delay_ratio"] = df["waitingTime"] / (df["duration"] + 1e-5)
    df["time_loss_ratio"] = df["timeLoss"] / (df["duration"] + 1e-5)
    df["stop_frequency"] = df["waitingCount"] / (df["duration"] + 1e-5)
    df["effective_speed"] = df["routeLength"] / (df["duration"] + 1e-5)
    
    # 4. One-hot encode vehicle type
    vtype_dummies = pd.get_dummies(df["vType"], prefix="vtype", drop_first=False)
    df = pd.concat([df, vtype_dummies], axis=1)
    
    # Save clean tripinfo
    output_path = os.path.join(PROCESSED_DIR, "clean_tripinfo.csv")
    df.to_csv(output_path, index=False)
    print(f"  -> Saved {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df

def generate_offline_transitions(df_summary):
    print("[3/4] Generating Offline Transitions for RL Value Prior...")
    
    # Aggregate into 30-second decision intervals matching the environment's GREEN_DURATION (30s)
    interval_sec = 30
    df_summary["interval_id"] = (df_summary["time"] // interval_sec).astype(int)
    
    grouped = df_summary.groupby("interval_id").agg({
        "running": "mean",
        "halting": "mean",
        "waiting": "mean",
        "meanWaitingTime": "mean",
        "meanSpeed": "mean",
        "step_penalty": "mean"
    }).reset_index()
    
    transitions = []
    for i in range(len(grouped) - 1):
        curr_row = grouped.iloc[i]
        next_row = grouped.iloc[i + 1]
        
        # State proxy: [running, halting, waiting, meanSpeed]
        state = [
            curr_row["running"],
            curr_row["halting"],
            curr_row["waiting"],
            curr_row["meanSpeed"]
        ]
        
        # Action proxy (inferred from queue trend: if halting decreased, phase favored congested lane)
        action = 0 if next_row["halting"] <= curr_row["halting"] else 1
        
        reward = curr_row["step_penalty"]
        
        next_state = [
            next_row["running"],
            next_row["halting"],
            next_row["waiting"],
            next_row["meanSpeed"]
        ]
        
        done = (i == len(grouped) - 2)
        
        transitions.append({
            "step": i,
            "s_running": state[0],
            "s_halting": state[1],
            "s_waiting": state[2],
            "s_speed": state[3],
            "action": action,
            "reward": reward,
            "next_s_running": next_state[0],
            "next_s_halting": next_state[1],
            "next_s_waiting": next_state[2],
            "next_s_speed": next_state[3],
            "done": int(done)
        })
        
    df_transitions = pd.DataFrame(transitions)
    output_path = os.path.join(PROCESSED_DIR, "offline_transitions.csv")
    df_transitions.to_csv(output_path, index=False)
    print(f"  -> Saved {output_path} ({df_transitions.shape[0]} transitions)")
    return df_transitions

def process_task_distributions():
    print("[4/4] Processing task distributions from dataset/tasks/net_double/...")
    
    train_tau = np.load("dataset/tasks/net_double/train_tau_c0.npy")
    valid_tau = np.load("dataset/tasks/net_double/valid_tau_c0.npy")
    test_tau = np.load("dataset/tasks/net_double/test_tau_c0.npy")
    
    stats = {
        "train": {
            "num_tasks": int(train_tau.shape[0]),
            "mean": train_tau.mean(axis=0).tolist(),
            "std": train_tau.std(axis=0).tolist(),
            "min": train_tau.min(axis=0).tolist(),
            "max": train_tau.max(axis=0).tolist(),
            "percentiles_25": np.percentile(train_tau, 25, axis=0).tolist(),
            "percentiles_75": np.percentile(train_tau, 75, axis=0).tolist(),
        },
        "valid": {
            "num_tasks": int(valid_tau.shape[0]),
            "mean": valid_tau.mean(axis=0).tolist(),
            "std": valid_tau.std(axis=0).tolist()
        },
        "test": {
            "num_tasks": int(test_tau.shape[0]),
            "mean": test_tau.mean(axis=0).tolist(),
            "std": test_tau.std(axis=0).tolist()
        }
    }
    
    output_path = os.path.join(PROCESSED_DIR, "task_distribution_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  -> Saved {output_path}")

if __name__ == "__main__":
    df_sum = preprocess_summary()
    df_trip = preprocess_tripinfo()
    generate_offline_transitions(df_sum)
    process_task_distributions()
    print("Preprocessing completed successfully!")

