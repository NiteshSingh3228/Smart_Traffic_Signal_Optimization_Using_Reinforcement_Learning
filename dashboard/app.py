
import streamlit as st
import pandas as pd
import os

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Traffic Optimization",
    page_icon="🚦",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30384a;
}

.metric-label {
    color: #aab4c5;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.section-title {
    color: #58a6ff;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# TITLE
# =====================================

st.title("🚦 Smart Traffic Signal Optimization")
st.subheader("Using Reinforcement Learning")

st.markdown("""
An intelligent traffic signal management system that compares
Fixed-Time, Q-Learning and Deep Q-Network (DQN) controllers,
with ambulance-priority support.
""")

st.divider()

# =====================================
# LOAD DATA
# =====================================

CSV_PATH = "results/metrics/controller_comparison.csv"

if not os.path.exists(CSV_PATH):
    st.error("Evaluation results not found. Run the evaluation first.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# =====================================
# TOP METRICS
# =====================================

st.markdown("### 📊 Overall Performance")

best_waiting = df.loc[
    df["Average Waiting Time"].idxmin()
]

best_queue = df.loc[
    df["Average Queue Length"].idxmin()
]

best_ambulance = df.loc[
    df["Average Ambulance Waiting Time"].idxmin()
]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Best Vehicle Waiting Time",
        f"{best_waiting['Average Waiting Time']} sec"
    )
    st.caption(best_waiting["Controller"])

with col2:
    st.metric(
        "Best Queue Length",
        f"{best_queue['Average Queue Length']}"
    )
    st.caption(best_queue["Controller"])

with col3:
    st.metric(
        "Best Ambulance Waiting Time",
        f"{best_ambulance['Average Ambulance Waiting Time']} sec"
    )
    st.caption(best_ambulance["Controller"])

with col4:
    st.metric(
        "Maximum Throughput",
        f"{df['Completed Vehicles'].max()}"
    )
    st.caption("Completed vehicles")

st.divider()

# =====================================
# CONTROLLER COMPARISON
# =====================================

st.markdown("### 📈 Controller Performance Comparison")

tab1, tab2, tab3, tab4 = st.tabs([
    "Vehicle Waiting Time",
    "Queue Length",
    "Throughput",
    "Ambulance Waiting Time"
])

with tab1:
    st.bar_chart(
        df,
        x="Controller",
        y="Average Waiting Time",
        color="#3498db"
    )

with tab2:
    st.bar_chart(
        df,
        x="Controller",
        y="Average Queue Length",
        color="#e67e22"
    )

with tab3:
    st.bar_chart(
        df,
        x="Controller",
        y="Completed Vehicles",
        color="#2ecc71"
    )

with tab4:
    st.bar_chart(
        df,
        x="Controller",
        y="Average Ambulance Waiting Time",
        color="#e74c3c"
    )

st.divider()

# =====================================
# DETAILED RESULTS
# =====================================

st.markdown("### 📋 Detailed Evaluation Results")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =====================================
# PROJECT INFORMATION
# =====================================

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
    - Average queue length
    - Completed vehicles
    - Ambulance waiting time
    """)

st.caption(
    "Smart Traffic Signal Optimization Using Reinforcement Learning | SUMO + Python + TraCI + Streamlit"
)
