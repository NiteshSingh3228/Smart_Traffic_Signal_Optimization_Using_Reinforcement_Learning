import sys
import os
import random
import time
import pandas as pd

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
import streamlit as st

st.set_page_config(
    page_title="Smart Traffic Signal Optimization",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Signal Optimization")
st.subheader("Using Reinforcement Learning")

st.markdown("""
An intelligent traffic signal management system comparing
Fixed-Time, Q-Learning and Deep Q-Network controllers,
with ambulance-priority support.
""")

# =====================================================
# LIVE SUMO SIMULATION
# =====================================================

st.divider()
st.markdown("## 🚗 Live SUMO Simulation")

st.write(
    "Watch vehicles move in SUMO-GUI while Q-Learning and DQN "
    "control the traffic signal. Both controllers use the same "
    "random seed for a fair comparison."
)

if st.button("▶ Run Q-Learning + DQN Comparison", type="primary"):

    from agents.q_learning import QLearningAgent
    from agents.dqn_agent import DQNAgent
    from environment.traffic_environment import TrafficEnvironment
    from evaluation.metrics import TrafficMetrics

    live_status = st.empty()
    progress_bar = st.progress(0)
    live_metrics = st.empty()
    reward_display = st.empty()
    chart_area = st.empty()
    history_table = st.empty()

    comparison_results = []

    def run_live_controller(controller_name, seed):

        live_status.info(
            f"Starting {controller_name} — SUMO-GUI will open."
        )

        # Load the already-trained model
        if controller_name == "Q-Learning":

            agent = QLearningAgent()
            agent.load_model("q_learning_ambulance_model_updated.pkl")

        else:

            agent = DQNAgent()
            agent.load_model("dqn_ambulance_model_updated.pth")

        agent.epsilon = 0

        env = TrafficEnvironment(use_gui=True)

        chart_history = []
        step_history = []

        reward_data = {
            "total_reward": 0,
            "last_reward": 0,
            "override_steps": 0,
            "action_counts": {
                "North-South Green": 0,
                "East-West Green": 0
            }
        }

        progress_bar.progress(0)

        def update_live(results):

            sim_time = results["simulation_time"]

            progress_bar.progress(
                min(int(sim_time / 900 * 100), 100)
            )

            # Throttle Streamlit re-renders to every 10 simulated seconds
            if int(sim_time) % 10 == 0 or sim_time == 1:

                try:
                    active_vehicles = len(traci.vehicle.getIDList())
                    current_phase = traci.trafficlight.getPhase("center")

                    phase_names = {
                        0: "North-South Green",
                        1: "North-South Yellow",
                        2: "East-West Green",
                        3: "East-West Yellow"
                    }

                    signal_status = phase_names.get(
                        current_phase, "Unknown"
                    )

                except Exception:
                    active_vehicles = 0
                    signal_status = "Unavailable"

                live_status.info(
                    f"{controller_name} running | "
                    f"Simulation Time: {sim_time:.0f} / 900 seconds | "
                    f"Signal: {signal_status}"
                )

                with live_metrics.container():

                    st.markdown(f"### 📊 {controller_name} Live Metrics")

                    row1 = st.columns(4)

                    row1[0].metric(
                        "Average Waiting Time",
                        f"{results['average_waiting_time']:.2f} sec"
                    )

                    row1[1].metric(
                        "Average Travel Time",
                        f"{results['average_travel_time']:.2f} sec"
                    )

                    row1[2].metric(
                        "Average Queue Length",
                        f"{results['average_queue_length']:.2f}"
                    )

                    row1[3].metric(
                        "Completed Vehicles",
                        results["completed_vehicles"]
                    )

                    row2 = st.columns(4)

                    row2[0].metric(
                        "Active Vehicles",
                        active_vehicles
                    )

                    row2[1].metric(
                        "Ambulance Avg Waiting",
                        f"{results['average_ambulance_waiting_time']:.2f} sec"
                    )

                    row2[2].metric(
                        "Maximum Ambulance Waiting",
                        f"{results['maximum_ambulance_waiting_time']:.2f} sec"
                    )

                    row2[3].metric(
                        "Observed Vehicles",
                        results["observed_vehicles"]
                    )

                # Update live chart every 10 simulation seconds
                chart_history.append({
                    "Simulation Time": sim_time,
                    "Average Waiting Time": results["average_waiting_time"],
                    "Average Travel Time": results["average_travel_time"],
                    "Average Queue": results["average_queue_length"]
                })

                if chart_history:

                    chart_df = pd.DataFrame(chart_history).set_index(
                        "Simulation Time"
                    )

                    chart_area.line_chart(
                        chart_df[
                            [
                                "Average Waiting Time",
                                "Average Travel Time",
                                "Average Queue"
                            ]
                        ],
                        x_label="Simulation Time (seconds)",
                        y_label="Metric Value",
                        height=350
                    )

        metrics = TrafficMetrics(on_update=update_live)

        try:

            state = env.reset(seed=seed)
            done = False

            while not done:

                action = agent.choose_action(state)

                if action == 0:
                    reward_data["action_counts"]["North-South Green"] += 1
                else:
                    reward_data["action_counts"]["East-West Green"] += 1

                next_state, reward, done = env.step(
                    action,
                    metrics=metrics
                )

                reward_data["last_reward"] = reward
                reward_data["total_reward"] += reward

                # Record one row after every RL decision
                step_results = metrics.get_results()

                active_vehicles = len(traci.vehicle.getIDList())

                current_queue = sum(
                    traci.edge.getLastStepHaltingNumber(edge)
                    for edge in ["N_in", "S_in", "E_in", "W_in"]
                )

                phase_names = {
                    0: "North-South Green",
                    1: "North-South Yellow",
                    2: "East-West Green",
                    3: "East-West Yellow"
                }

                current_phase = traci.trafficlight.getPhase("center")

                step_history.append({
                    "Simulation Time (sec)": round(
                        step_results["simulation_time"], 1
                    ),
                    "Reward": round(reward, 2),
                    "Active Vehicles": active_vehicles,
                    "Completed Vehicles": step_results["completed_vehicles"],
                    "Current Queue": current_queue,
                    "Average Waiting (sec)": step_results["average_waiting_time"],
                    "Average Travel (sec)": step_results["average_travel_time"],
                    "Signal Phase": phase_names.get(
                        current_phase, "Unknown"
                    ),
                    "Ambulance Avg Wait (sec)": (
                        step_results["average_ambulance_waiting_time"]
                    )
                })

                history_table.dataframe(
                    pd.DataFrame(step_history),
                    width="stretch",
                    hide_index=True
                )

                if env.emergency_override_during_step:
                    reward_data["override_steps"] += 1

                reward_display.metric(
                    f"{controller_name} Reward",
                    f"{reward:.2f}",
                    help="Reward for the latest RL action. "
                         "The reward is negative queue length."
                )

                state = next_state

            final_results = metrics.get_results()

            final_results.update({
                "controller": controller_name,
                "total_reward": reward_data["total_reward"],
                "last_reward": reward_data["last_reward"],
                "ambulance_override_steps": reward_data["override_steps"],
                "north_south_actions": reward_data["action_counts"][
                    "North-South Green"
                ],
                "east_west_actions": reward_data["action_counts"][
                    "East-West Green"
                ]
            })

            return final_results

        finally:
            env.close()

    try:

        comparison_seed = random.SystemRandom().randint(1, 100000)
        st.info(f"Traffic simulation seed: {comparison_seed}")

        # Run Q-Learning first
        q_results = run_live_controller("Q-Learning", seed=comparison_seed)
        comparison_results.append(q_results)

        live_status.success(
            "Q-Learning simulation completed. Starting DQN..."
        )

        time.sleep(1.5)

        # Run DQN after Q-Learning has closed
        dqn_results = run_live_controller("DQN", seed=comparison_seed)
        comparison_results.append(dqn_results)

        st.session_state["live_comparison"] = comparison_results

        progress_bar.progress(100)
        live_status.success("Both simulations completed successfully!")

    except Exception as error:

        live_status.error(
            f"Simulation stopped because of an error: {error}"
        )

        st.exception(error)

# =====================================================
# LIVE COMPARISON RESULTS
# =====================================================

if st.session_state.get("live_comparison"):

    st.divider()
    st.markdown("### 🏆 Live Q-Learning vs DQN Results")

    live_df = pd.DataFrame(
        st.session_state["live_comparison"]
    )

    comparison_columns = {
        "controller": "Controller",
        "average_waiting_time": "Average Waiting (sec)",
        "average_travel_time": "Average Travel (sec)",
        "average_queue_length": "Average Queue",
        "completed_vehicles": "Completed Vehicles",
        "average_ambulance_waiting_time": "Ambulance Waiting (sec)",
        "total_reward": "Cumulative Reward",
        "ambulance_override_steps": "Ambulance Override Steps"
    }

    display_live_df = live_df[
        list(comparison_columns.keys())
    ].rename(columns=comparison_columns)

    st.dataframe(
        display_live_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("#### 📈 Controller Comparison")

    chart_data = live_df.set_index("controller")[[
        "average_waiting_time",
        "average_travel_time",
        "average_queue_length",
        "completed_vehicles"
    ]]

    chart_data.columns = [
        "Average Waiting Time",
        "Average Travel Time",
        "Average Queue",
        "Completed Vehicles"
    ]

    st.bar_chart(chart_data)

    st.markdown("#### 🚦 Signal Action Counts")

    action_df = live_df.set_index("controller")[[
        "north_south_actions",
        "east_west_actions"
    ]]

    action_df.columns = [
        "North-South Green Actions",
        "East-West Green Actions"
    ]

    st.bar_chart(action_df)

# =====================================================
# EXISTING FIVE-SEED EVALUATION DASHBOARD
# =====================================================

st.divider()
st.markdown("## 📊 Five-Seed Evaluation Results")

CSV_PATH = "results/metrics/controller_multi_seed_summary.csv"

if not os.path.exists(CSV_PATH):

    st.warning("Five-seed evaluation results not found.")

else:

    raw = pd.read_csv(CSV_PATH)

    df = raw.pivot(
        index="Controller",
        columns="Metric",
        values=["Mean", "Std Dev"]
    )

    df.columns = [
        f"{metric}_{stat}"
        for stat, metric in df.columns
    ]

    df = df.reset_index()

    best_wait = df.loc[
        df["average_waiting_time_Mean"].idxmin()
    ]

    best_queue = df.loc[
        df["average_queue_length_Mean"].idxmin()
    ]

    best_ambulance = df.loc[
        df["average_ambulance_waiting_time_Mean"].idxmin()
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best Vehicle Waiting Time",
        f"{best_wait['average_waiting_time_Mean']:.3f} sec"
    )
    c1.caption(best_wait["Controller"])

    c2.metric(
        "Best Queue Length",
        f"{best_queue['average_queue_length_Mean']:.3f}"
    )
    c2.caption(best_queue["Controller"])

    c3.metric(
        "Best Ambulance Waiting Time",
        f"{best_ambulance['average_ambulance_waiting_time_Mean']:.3f} sec"
    )
    c3.caption(best_ambulance["Controller"])

    c4.metric(
        "Maximum Throughput",
        f"{df['completed_vehicles_Mean'].max():.1f}"
    )
    c4.caption("Average completed vehicles")

    st.markdown("### 📈 Controller Performance Comparison")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Waiting Time",
        "Queue Length",
        "Throughput",
        "Ambulance Waiting"
    ])

    with tab1:
        st.bar_chart(
            df.set_index("Controller")["average_waiting_time_Mean"]
        )

    with tab2:
        st.bar_chart(
            df.set_index("Controller")["average_queue_length_Mean"]
        )

    with tab3:
        st.bar_chart(
            df.set_index("Controller")["completed_vehicles_Mean"]
        )

    with tab4:
        st.bar_chart(
            df.set_index("Controller")[
                "average_ambulance_waiting_time_Mean"
            ]
        )

    st.markdown("### 📋 Detailed Five-Seed Evaluation Results")

    st.caption(
        "Values are mean and standard deviation across five random seeds."
    )

    display_df = df[[
        "Controller",
        "average_waiting_time_Mean",
        "average_waiting_time_Std Dev",
        "average_queue_length_Mean",
        "average_queue_length_Std Dev",
        "completed_vehicles_Mean",
        "average_ambulance_waiting_time_Mean"
    ]].copy()

    display_df.columns = [
        "Controller",
        "Mean Waiting Time (sec)",
        "Waiting Time Std Dev",
        "Mean Queue Length",
        "Queue Std Dev",
        "Completed Vehicles (Mean)",
        "Ambulance Waiting Time (sec)"
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

# =====================================================
# PROJECT INFORMATION
# =====================================================

st.divider()
st.markdown("### 🧠 Project Information")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    **Traffic Controllers**
    - Fixed-Time
    - Fixed-Time + Ambulance Priority
    - Q-Learning + Ambulance Priority
    - DQN + Ambulance Priority
    """)

with col2:

    st.markdown("""
    **Evaluation Metrics**
    - Average vehicle waiting time
    - Average vehicle travel time
    - Average queue length
    - Completed vehicles
    - Ambulance waiting time
    - RL reward and ambulance overrides
    """)

st.caption(
    "Smart Traffic Signal Optimization | SUMO + Python + TraCI + Streamlit"
)
