# 🚦 Smart Traffic Signal Optimization Using Reinforcement Learning

An intelligent traffic signal control system that uses **Reinforcement Learning** (Q-Learning and Deep Q-Network) to optimize traffic flow at a four-way intersection. The system integrates with the **SUMO** (Simulation of Urban Mobility) traffic simulator and includes **emergency vehicle (ambulance) priority override** functionality.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
  - [Running Tests](#running-tests)
- [State Representation](#state-representation)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Emergency Vehicle Priority](#emergency-vehicle-priority)
- [Results](#results)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)

---

## Overview

Traditional traffic signals operate on fixed timing cycles regardless of actual traffic conditions. This project replaces fixed-time control with **adaptive RL agents** that learn optimal signal switching strategies by interacting with a realistic traffic simulation.

The system simulates a **single four-way intersection** with traffic arriving from North, South, East, and West. Two RL agents — **Q-Learning** (tabular) and **DQN** (neural network) — are trained to minimize vehicle queue lengths by deciding when to switch between North-South and East-West green phases.

An **ambulance priority system** automatically overrides RL decisions when an emergency vehicle has been waiting longer than a configurable threshold (default: 15 seconds), ensuring the ambulance receives a green signal to pass through the intersection.

---

## Features

- **Two RL Algorithms**: Q-Learning with state discretization and DQN with neural network function approximation
- **SUMO Integration**: Realistic microscopic traffic simulation via TraCI (Traffic Control Interface)
- **Emergency Vehicle Priority**: Automatic detection of ambulances and signal override to clear their path
- **Fixed-Time Baseline**: Traditional controller for performance comparison
- **Comprehensive Metrics**: Average waiting time, queue length, throughput, and simulation time
- **Ambulance-Aware Training**: RL updates are skipped when ambulance priority overrides the agent's action, preventing the agent from learning incorrect state-action mappings
- **State Normalization (DQN)**: Input normalization for stable neural network training
- **State Discretization (Q-Learning)**: Continuous state values are bucketed into discrete categories for tabular lookup
- **GUI Support**: Optional SUMO-GUI mode for visual inspection of traffic flow
- **Reproducible Experiments**: Configurable random seeds for deterministic simulation runs

---

## Project Structure

```
Smart_Traffic_Signal_Optimization_Using_Reinforcement_Learning/
│
├── agents/                          # Reinforcement Learning agents
│   ├── q_learning.py                # Q-Learning agent with discretized state space
│   └── dqn_agent.py                 # Deep Q-Network agent with PyTorch
│
├── environment/                     # SUMO environment wrapper
│   ├── traffic_environment.py       # Main RL environment (state, step, reward, ambulance priority)
│   └── traffic_monitor.py           # Traffic data collection utility
│
├── training/                        # Training scripts
│   ├── train_q_learning.py          # Train Q-Learning agent (100 episodes)
│   └── train_dqn.py                 # Train DQN agent (100 episodes)
│
├── evaluation/                      # Evaluation and comparison
│   ├── evaluate_controllers.py      # Compare Fixed-Time vs Q-Learning vs DQN
│   ├── evaluate_dqn.py              # Standalone DQN evaluation
│   ├── fixed_time_controller.py     # Fixed-time signal controller (baseline)
│   ├── metrics.py                   # TrafficMetrics class (waiting time, queue, throughput)
│   ├── plot_results.py              # Generate comparison bar charts
│   └── save_results.py              # Save evaluation results to CSV
│
├── tests/                           # Test scripts
│   ├── test_traffic_environment.py  # Full episode simulation test
│   ├── test_traffic_monitor.py      # Traffic data collection test
│   ├── test_q_learning.py           # Q-Learning agent unit test
│   ├── test_q_learning_state.py     # Q-Learning 17-value state test
│   ├── test_load_q_learning.py      # Model loading test
│   ├── test_dqn_agent.py            # DQN agent unit test
│   ├── test_dqn_state.py            # DQN normalization test
│   ├── test_fixed_time.py           # Fixed-time controller test
│   ├── test_metrics.py              # Metrics collection test
│   ├── test_ambulance_detection.py  # Ambulance detection test
│   ├── test_ambulance_state.py      # Ambulance state inclusion test
│   ├── test_ambulance_priority.py   # Ambulance signal override test
│   └── test_ambulance_crossing.py   # End-to-end ambulance crossing test
│
├── sumo/                            # SUMO simulation files
│   ├── network/
│   │   ├── intersection.nod.xml     # Intersection node definitions (5 nodes)
│   │   ├── intersection.edg.xml     # Road edge definitions (8 edges)
│   │   └── intersection.net.xml     # Compiled SUMO network
│   ├── routes/
│   │   └── routes.rou.xml           # Vehicle types, routes, traffic flows, and ambulance
│   ├── config/                      # (Reserved for SUMO configuration)
│   └── additional/                  # (Reserved for additional SUMO files)
│
├── results/                         # Training and evaluation outputs
│   ├── models/
│   │   ├── q_learning_model.pkl             # Trained Q-Learning Q-table
│   │   ├── q_learning_ambulance_model.pkl   # Ambulance-aware Q-Learning model
│   │   ├── dqn_model.pth                    # Trained DQN model checkpoint
│   │   └── dqn_ambulance_model.pth          # Ambulance-aware DQN model
│   ├── metrics/
│   │   └── controller_comparison.csv        # Side-by-side performance comparison
│   ├── graphs/
│   │   ├── waiting_time_comparison.png      # Waiting time bar chart
│   │   ├── queue_length_comparison.png      # Queue length bar chart
│   │   └── throughput_comparison.png        # Throughput bar chart
│   └── logs/                                # (Reserved for training logs)
│
├── dashboard/                       # (Reserved for web dashboard)
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
│                                                             │
│   ┌───────────┐     action      ┌──────────────────────┐   │
│   │  RL Agent  │───────────────►│  TrafficEnvironment  │   │
│   │            │                │                      │   │
│   │ Q-Learning │◄───────────────│   SUMO Simulator     │   │
│   │   or DQN   │  state, reward │   (via TraCI)        │   │
│   └───────────┘     done        │                      │   │
│                                 │  ┌──────────────┐    │   │
│                                 │  │  Ambulance   │    │   │
│                                 │  │  Priority    │    │   │
│                                 │  │  Override    │    │   │
│                                 │  └──────────────┘    │   │
│                                 └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Signal Phase Cycle:**

```
North-South GREEN ──► North-South YELLOW ──► East-West GREEN ──► East-West YELLOW ──► (repeat)
    (Phase 0)             (Phase 1)             (Phase 2)            (Phase 3)
```

---

## Prerequisites

- **Python** 3.10+
- **SUMO** (Eclipse SUMO) — [Installation Guide](https://sumo.dlr.de/docs/Installing/index.html)
- **PyTorch** (for DQN agent)
- **NumPy**
- **pandas** and **matplotlib** (for plotting)

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/NiteshSingh3228/Smart_Traffic_Signal_Optimization_Using_Reinforcement_Learning.git
cd Smart_Traffic_Signal_Optimization_Using_Reinforcement_Learning
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

3. **Install dependencies:**

```bash
pip install traci numpy torch pandas matplotlib
```

4. **Verify SUMO installation:**

```bash
sumo --version
```

> **Note:** If `SUMO_HOME` is not set, you may see XML validation warnings. These can be safely ignored — the simulation will still run correctly.

---

## Usage

### Training

**Train the Q-Learning agent (100 episodes):**

```bash
python -m training.train_q_learning
```

This trains a tabular Q-Learning agent and saves the model to `results/models/q_learning_ambulance_model.pkl`.

**Train the DQN agent (100 episodes):**

```bash
python -m training.train_dqn
```

This trains a Deep Q-Network agent with experience replay and a target network, saving to `results/models/dqn_ambulance_model.pth`.

> Both training scripts use different random seeds per episode (`seed=42+episode`) for diverse training scenarios. When an ambulance priority override occurs during a step, the training update for that step is skipped to avoid learning from overridden actions.

### Evaluation

**Compare all three controllers:**

```bash
python -m evaluation.evaluate_controllers
```

This runs the Fixed-Time, Q-Learning, and DQN controllers on the same simulation (seed 42) and saves a comparison CSV to `results/metrics/controller_comparison.csv`.

**Evaluate DQN only:**

```bash
python -m evaluation.evaluate_dqn
```

### Visualization

**Generate comparison charts:**

```bash
python -m evaluation.plot_results
```

This reads the CSV results and generates three bar charts in `results/graphs/`:
- `waiting_time_comparison.png`
- `queue_length_comparison.png`
- `throughput_comparison.png`

### Running Tests

All tests are executed as Python modules from the project root:

```bash
# Environment tests
python -m tests.test_traffic_environment
python -m tests.test_traffic_monitor

# Agent tests
python -m tests.test_q_learning
python -m tests.test_q_learning_state
python -m tests.test_dqn_agent
python -m tests.test_dqn_state
python -m tests.test_load_q_learning

# Ambulance tests
python -m tests.test_ambulance_detection
python -m tests.test_ambulance_state
python -m tests.test_ambulance_priority
python -m tests.test_ambulance_crossing

# Evaluation tests
python -m tests.test_fixed_time
python -m tests.test_metrics
```

---

## State Representation

The environment produces a **17-dimensional state vector**:

| Index | Description | Range |
|-------|-------------|-------|
| 0–1 | North road: vehicle count, queue length | 0–20+ |
| 2–3 | South road: vehicle count, queue length | 0–20+ |
| 4–5 | East road: vehicle count, queue length | 0–20+ |
| 6–7 | West road: vehicle count, queue length | 0–20+ |
| 8–11 | Ambulance count per direction (N, S, E, W) | 0–5+ |
| 12–15 | Max ambulance waiting time per direction (N, S, E, W) | 0–30+ seconds |
| 16 | Current traffic signal phase | 0 or 2 |

**Q-Learning** discretizes this into categorical buckets (0–3 per feature).

**DQN** normalizes each feature to the [0, 1] range using domain-specific scaling factors.

---

## Action Space

| Action | Signal Phase |
|--------|-------------|
| 0 | North-South GREEN |
| 1 | East-West GREEN |

When the agent selects an action, the environment:
1. Switches through the appropriate **yellow phase** (3 seconds) if changing direction
2. Holds the selected **green phase** for 30 seconds
3. Monitors for **ambulance priority** every second during the green phase

---

## Reward Function

```
reward = −(total halted vehicles across all incoming edges)
```

A lower queue length yields a less negative reward (closer to 0), guiding the agent toward signal timings that minimize waiting.

---

## Emergency Vehicle Priority

The system includes a real-time **ambulance priority override** mechanism:

1. **Detection**: Every simulation second, the environment scans for vehicles of type `ambulance` on incoming edges
2. **Activation**: If any ambulance has been waiting ≥ **15 seconds**, priority is activated for its direction
3. **Override**: The signal immediately switches (through yellow) to give the ambulance a green light, overriding the RL agent's decision
4. **Release**: Priority is released once the ambulance crosses the intersection or exits the simulation
5. **Training Safety**: When an override occurs, the training step is skipped (`emergency_override_during_step` flag) so the agent does not learn from an action it did not truly execute

---

## Results

Performance comparison across controllers (600-second simulation, seed 42):

| Controller | Avg Waiting Time (s) | Avg Queue Length | Completed Vehicles | Observed Vehicles |
|------------|---------------------|-----------------|-------------------|------------------|
| Fixed-Time | 8.19 | 1.09 | 76 | 80 |
| Q-Learning | 3.92 | 0.52 | 76 | 80 |
| DQN | 2.94 | 0.39 | 76 | 80 |

> **Key takeaway**: Both RL agents significantly outperform the fixed-time controller. DQN achieves the lowest average waiting time (**64% reduction** vs fixed-time) and queue length (**64% reduction**), while maintaining the same vehicle throughput.

---

## Configuration

Key parameters can be adjusted in the source files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `GREEN_DURATION` | `traffic_environment.py` | 30 s | Duration of each green phase |
| `YELLOW_DURATION` | `traffic_environment.py` | 3 s | Duration of yellow transition |
| `MAX_EPISODE_TIME` | `traffic_environment.py` | 600 s | Maximum simulation time per episode |
| `AMBULANCE_WAIT_THRESHOLD` | `traffic_environment.py` | 15 s | Seconds before ambulance gets priority |
| `NUM_EPISODES` | `train_q_learning.py` / `train_dqn.py` | 100 | Number of training episodes |
| `learning_rate` | `q_learning.py` | 0.1 | Q-Learning step size (α) |
| `discount_factor` | `q_learning.py` | 0.95 | Future reward discount (γ) |
| `epsilon_decay` | `q_learning.py` / `dqn_agent.py` | 0.995 | Exploration decay rate |
| `batch_size` | `dqn_agent.py` | 64 | Experience replay batch size |
| `learning_rate` | `dqn_agent.py` | 0.001 | DQN neural network learning rate |

**SUMO Network:**
- 5 nodes: 1 traffic-light-controlled center + 4 priority endpoints (N, S, E, W)
- 8 edges: 2 lanes per direction, speed limit 13.89 m/s (≈50 km/h)
- 480 cars total: 120 vehicles per direction over 3600 seconds
- 1 ambulance: departs at t=10s, travels North→South

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| [SUMO](https://eclipse.dev/sumo/) | Microscopic traffic simulation |
| [TraCI](https://sumo.dlr.de/docs/TraCI.html) | Python interface to control SUMO |
| [PyTorch](https://pytorch.org/) | Neural network for DQN agent |
| [NumPy](https://numpy.org/) | Numerical computations |
| [pandas](https://pandas.pydata.org/) | Data manipulation for results |
| [matplotlib](https://matplotlib.org/) | Plotting comparison charts |
| Python | Core programming language |

---

## Author

**Nitesh Singh** — [@NiteshSingh3228](https://github.com/NiteshSingh3228)

---

## License

This project is open source and available for academic and research purposes.