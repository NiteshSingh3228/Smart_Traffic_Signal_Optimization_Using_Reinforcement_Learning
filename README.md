# 🚦 Smart Traffic Signal Optimization Using Reinforcement Learning

An intelligent, multi-agent traffic signal management system that uses **Reinforcement Learning** (Deep Q-Networks, Tabular Q-Learning, and Hybrid Boltzmann Ensemble Fusion) to optimize traffic flow, minimize delays, and provide **emergency vehicle (ambulance) priority preemption** across both arterial corridors and isolated intersections. 

The system simulates dynamic traffic via **Eclipse SUMO 1.28.0 (TraCI)** and integrates three distinct data and experience sources to train and evaluate adaptive traffic policies.

---

## 📌 Table of Contents

- [System Architecture](#system-architecture)
- [Three-Source Data Integration](#three-source-data-integration)
- [Controller Portfolio](#controller-portfolio)
- [State Representation (17-D)](#state-representation-17-d)
- [Action Space & Emergency Preemption](#action-space--emergency-preemption)
- [Combined (-ve and +ve) Reward Function](#combined--ve-and-ve-reward-function)
- [5-Seed Empirical Benchmark Results](#5-seed-empirical-benchmark-results)
- [Autonomous Traffic Signal Command Center (Dashboard)](#autonomous-traffic-signal-command-center-dashboard)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage & Execution Guide](#usage--execution-guide)
- [Testing & Verification](#testing--verification)

---

## 🏗️ System Architecture

The simulation environment supports two primary network topologies:
1. **2-Intersection Arterial Corridor (`corridor_2`) [Default]**: An interconnected urban arterial with two signalized junctions (**J1 West** and **J2 East**), handling internal turning flows, cross-street traffic, and coordinated east-west arterial throughput.
2. **Single Isolated Intersection (`single`)**: A classic four-way intersection (North, South, East, West) with single-junction signal timing.

---

## 🔄 Three-Source Data Integration

To achieve state-of-the-art policy generalization and prevent catastrophic forgetting, the system synthesizes three complementary data sources:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRIPLE-SOURCE TRAINING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  SOURCE 1: Learned Online Experience                                        │
│  - Baseline DQN weights (dqn_ambulance_model_updated.pth)                   │
│  - Baseline Q-table checkpoints (q_learning_ambulance_model_updated.pkl)    │
│                                                                             │
│  SOURCE 2: Historical Traffic Dataset (archive/)                           │
│  - 1,749 trip records, 59 multi-intersection offline transitions            │
│  - Random Forest Supervised Delay Model (R² = 0.5345, MAE = 27.78s)         │
│                                                                             │
│  SOURCE 3: Multi-Scenario Traffic Dataset (dataset/)                        │
│  - 1,000 traffic tasks (train_tau_c0.npy) across variable arrival rates    │
│  - Arterial corridor turn-definitions and dynamic route profiles            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINE-TUNED & INTEGRATED MODELS                     │
│  • dqn_integrated_model.pth (L2 Delta: 8.1738)                              │
│  • q_learning_integrated_model.pkl (166 discrete states)                    │
│  • Integrated Hybrid Agent (Softmax Boltzmann Ensemble)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Controller Portfolio

| Controller | Algorithm / Type | Key Characteristic | Model File |
|---|---|---|---|
| **Integrated Hybrid Ensemble** | Softmax Boltzmann Fusion | Synthesizes microscopic DQN with macroscopic Tabular QL | `dqn_integrated_model.pth` + `q_learning_integrated_model.pkl` |
| **Integrated DQN** | 3-Layer Deep Q-Network | Fine-tuned across multi-scenario arrival matrices | `dqn_integrated_model.pth` |
| **Integrated Q-Learning** | Tabular Q-Learning (216 states) | Offline experience replay + state space expanded to 166 states | `q_learning_integrated_model.pkl` |
| **Baseline Hybrid Ensemble** | Ensemble Fusion | Online RL baseline fusion | `dqn_ambulance_model_updated.pth` + `q_learning_ambulance_model_updated.pkl` |
| **Baseline DQN** | Deep Q-Network | Online experience on single-seed simulation | `dqn_ambulance_model_updated.pth` |
| **Baseline Q-Learning** | Tabular Q-Learning | Online baseline policy (53 learned states) | `q_learning_ambulance_model_updated.pkl` |
| **Fixed-Time Baseline** | Deterministic 30s Cycles | Uncoordinated static phase transitions | N/A |

---

## 📐 State Representation (17-D)

The observation vector fed to both the DQN and the Tabular state discretizer comprises 17 real-valued continuous dimensions:

$$\mathbf{S} = [N_1, S_1, N_2, S_2, E, W, J_{2\to 1}, J_{1\to 2}, A_{N1}, A_{S1}, A_E, A_W, W_{N1}, W_{S1}, W_E, W_W, \Phi]$$

1. **Indices 0–7 (Traffic Densities & Queues)**: Approaching vehicle count and halt count across all monitored approaches.
2. **Indices 8–11 (Ambulance Counts)**: Active emergency vehicles in North, South, East, and West corridors.
3. **Indices 12–15 (Ambulance Waiting Times)**: Cumulative seconds elapsed while emergency vehicles wait at red lights.
4. **Index 16 (Signal Phase)**: Active green signal index ($0 = \text{North-South Green}, 1 = \text{East-West Green}$).

---

## ⚡ Action Space & Emergency Preemption

The RL agents select from a discrete action space of size 2:
- **Action 0**: North-South Green (serves cross-street approaches $N_1, S_1, N_2, S_2$).
- **Action 1**: East-West Green (serves arterial corridor flow $W, E, J_{1\to 2}, J_{2\to 1}$).

### Safe Preemption Guard
When an ambulance (`vClass="emergency"`) is detected waiting $\ge 15$ seconds on a red approach:
1. Normal RL step is safely interrupted.
2. A mandatory **3-second yellow transition** (`YELLOW_DURATION = 3`) is enforced to eliminate vehicle collisions.
3. Target green is granted until the ambulance clears the intersection.
4. Normal RL dispatch automatically resumes without destabilizing the policy.

---

## 🎯 Combined (-ve and +ve) Reward Function

The reward balances arterial clearance with congestion minimization:

$$\text{Reward} = + (2.5 \times \text{Completed Vehicles}) - \Big(\text{Average Queue Length} + \min(1.5 \times \text{Ambulance Wait}, 15.0)\Big)$$

- **Positive Reward ($> 0$)**: Awarded during smooth flow clearance (typically $+10$ to $+25$).
- **Negative Penalty ($< 0$)**: Penalizes queue build-up and emergency vehicle delays.

---

## 📊 5-Seed Empirical Benchmark Results

Evaluated across seeds `[42, 101, 202, 303, 404]` under identical stochastic arrivals on the 2-intersection corridor:

| Controller | Mean Wait Time | Wait Std | Mean Queue | Completed Vehicles | Ambulance Wait |
|---|---|---|---|---|---|
| **Integrated DQN (Sources 1+2+3)** | **11.57 s** | $\pm 1.19$ | **2.26 veh** | **105.4 veh** | 4.09 s |
| **Fixed-Time Baseline** | 11.98 s | $\pm 0.67$ | 2.35 veh | 105.0 veh | 8.07 s |
| **Baseline DQN (Source 1)** | 12.03 s | $\pm 0.87$ | 2.36 veh | 105.2 veh | **0.55 s** |
| **Baseline Hybrid Ensemble** | 12.03 s | $\pm 0.87$ | 2.36 veh | 105.2 veh | **0.55 s** |
| **Baseline Q-Learning** | 14.51 s | $\pm 0.98$ | 2.84 veh | 105.4 veh | 10.05 s |
| **Integrated Q-Learning** | 38.10 s | $\pm 50.97$ | 8.17 veh | 95.6 veh | 4.59 s |

> **Key Finding**: Integrated DQN achieved the **lowest vehicle waiting time (11.57s)** and **lowest queue length (2.26 veh)** across all controllers.

---

## 🖥️ Autonomous Traffic Signal Command Center (Dashboard)

The project includes an interactive web dashboard built with **Streamlit** and **Plotly**:

```bash
streamlit run dashboard/app.py
```

### Dashboard Features:
- **Pure White Background & High Contrast**: Specially tuned for clean visualization across macOS Light and Dark modes.
- **Interactive Dispatch Controls**: Toggle between 7 AI controllers, adjust DQN/Q-Learning ensemble weights, and configure arrival distributions.
- **Real-Time Live Digital Twin**: Launches native desktop `SUMO-GUI` in sync with real-time web telemetry.
- **Interactive Plotly Analytics**: Real-time line charts and cross-validation bar charts rendered with pure `#ffffff` canvas.
- **Live Decision Consensus**: Visualizes Boltzmann softmax agreement rates between DQN and Q-Learning agents in real-time.

---

## 📁 Project Directory Structure

```
Smart_Traffic_Signal_Optimization_Using_Reinforcement_Learning/
│
├── agents/                                  # RL Agent Architectures
│   ├── dqn_agent.py                         # PyTorch Deep Q-Network
│   ├── q_learning.py                        # Tabular Q-Learning with state discretizer
│   ├── hybrid_agent.py                      # Baseline Boltzmann Softmax Ensemble
│   └── hybrid_integrated_agent.py           # Triple-Source Integrated Hybrid Ensemble
│
├── environment/                             # SUMO TraCI Simulation Environment
│   ├── traffic_environment.py               # Multi-intersection environment wrapper
│   └── traffic_monitor.py                   # Edge sensor telemetry collector
│
├── training/                                # Training Pipelines
│   ├── train_dqn.py                         # Online DQN training script
│   ├── train_q_learning.py                  # Online Q-Learning training script
│   └── train_integrated_models.py           # Multi-scenario triple-source training
│
├── evaluation/                              # Benchmarking & Metrics
│   ├── evaluate_all_controllers.py          # 5-seed comprehensive evaluation matrix
│   ├── evaluate_controllers.py              # Single-run controller benchmark
│   ├── evaluate_dqn.py                      # Standalone DQN evaluation
│   ├── evaluate_multiple_seeds.py           # Multi-seed evaluation runner
│   ├── fixed_time_controller.py             # Deterministic cycle controller
│   ├── metrics.py                           # Vehicle and ambulance metric logger
│   └── plot_results.py                      # Matplotlib static chart generator
│
├── models/                                  # Supervised Machine Learning
│   └── supervised_delay_model.py            # Random Forest Delay Regressor
│
├── dashboard/                               # Web Command Center
│   └── app.py                               # Streamlit + Plotly interactive UI
│
├── sumo/                                    # Simulation Definitions
│   ├── network/                             # Network XML files (corridor_2, corridor_3, intersection)
│   └── routes/                              # Traffic demand XML files
│
├── tests/                                   # Automated Unit & Integration Tests
│   ├── test_dqn_agent.py                    # DQN inference test
│   ├── test_dqn_state.py                    # 17-D tensor normalization test
│   ├── test_q_learning.py                   # Q-table update test
│   ├── test_q_learning_state.py             # State discretization test
│   ├── test_load_q_learning.py              # Q-table checkpoint load test
│   ├── test_metrics.py                      # Metrics calculations unit test
│   ├── test_integrated_models.py            # Triple-source model validation
│   ├── test_ambulance_crossing.py           # Priority preemption test
│   └── test_traffic_environment.py          # Full simulation step test
│
├── results/                                 # Models, Metrics & Graphs
│   ├── models/                              # Trained PyTorch and pickle checkpoints
│   ├── metrics/                             # CSV benchmark summaries
│   └── graphs/                              # Saved performance figures
│
├── requirements.txt                         # Pinned Python package dependencies
└── README.md                                # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. Install Eclipse SUMO
Ensure SUMO is installed on your operating system:
- **macOS**: `brew install sumo` or install Eclipse SUMO framework (`/Library/Frameworks/EclipseSUMO.framework`).
- **Ubuntu/Debian**: `sudo apt install sumo sumo-tools sumo-doc`.
- Ensure `SUMO_HOME` is exported in your shell profile.

### 2. Python Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage & Execution Guide

### 1. Launch the Command Center Dashboard
```bash
streamlit run dashboard/app.py
```

### 2. Run Comprehensive 5-Seed Benchmark
```bash
python -m evaluation.evaluate_all_controllers
```

### 3. Fine-Tune Integrated Models
```bash
python training/train_integrated_models.py
```

---

## ✅ Testing & Verification

Run all core agent and integration verification tests:

```bash
python tests/test_dqn_agent.py
python tests/test_dqn_state.py
python tests/test_q_learning.py
python tests/test_q_learning_state.py
python tests/test_load_q_learning.py
python tests/test_metrics.py
python tests/test_integrated_models.py
```
