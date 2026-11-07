import sys
import os
import random
import time
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# ROOT PATH INJECTION
# Ensures `environment`, `agents`, `evaluation` packages
# are importable regardless of where Streamlit is launched from.
# =====================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

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

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Autonomous Traffic Signal Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GLASSMORPHISM & BULLETPROOF HIGH-CONTRAST LIGHT CSS
# =====================================================
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Force Light Theme Base CSS Variables */
:root {
    --text-color: #0f172a !important;
    --background-color: #f0f4f8 !important;
    --secondary-background-color: #ffffff !important;
    --primary-color: #0284c7 !important;
    color-scheme: light !important;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    background-color: #f0f4f8 !important;
}

/* Base App Canvas */
.stApp {
    background-color: #f0f4f8 !important;
    background-image:
        radial-gradient(ellipse at 15% 20%, rgba(224, 242, 254, 0.6) 0%, transparent 60%),
        radial-gradient(ellipse at 85% 70%, rgba(238, 242, 255, 0.5) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 95%, rgba(236, 253, 245, 0.4) 0%, transparent 50%) !important;
}

/* Header bar */
header[data-testid="stHeader"] {
    background: rgba(240, 244, 248, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(226, 232, 240, 0.6) !important;
}
header[data-testid="stHeader"] * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* =====================================================
   SIDEBAR - COMPREHENSIVE CONTRAST ENFORCEMENT
   Fixes invisible text on macOS dark mode
   ===================================================== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border-right: 1.5px solid #cbd5e1 !important;
    box-shadow: 2px 0 16px rgba(15, 23, 42, 0.04) !important;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Radio button options in sidebar */
section[data-testid="stSidebar"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] div[role="radiogroup"] label div,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 500 !important;
}

/* Slider labels and values */
section[data-testid="stSidebar"] div[data-testid="stSlider"] label,
section[data-testid="stSidebar"] div[data-testid="stSlider"] div,
section[data-testid="stSidebar"] div[data-testid="stSlider"] span,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
}

/* Selectbox */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] span,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] p {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
}

/* Checkbox */
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
}

/* =====================================================
   CUSTOM METRIC CARDS (Bulletproof Light Theme)
   ===================================================== */
.metric-box {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05) !important;
    margin-bottom: 12px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.metric-box:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px -3px rgba(14, 165, 233, 0.12) !important;
}
.metric-box-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    margin-bottom: 6px !important;
}
.metric-box-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    line-height: 1.2 !important;
}
.metric-box-sub {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #0284c7 !important;
    -webkit-text-fill-color: #0284c7 !important;
    margin-top: 6px !important;
}

/* =====================================================
   PURE WHITE GRAPHS & PLOTLY CONTAINERS
   ===================================================== */
.plot-container,
.plotly,
.js-plotly-plot,
div[data-testid="stPlotlyChart"],
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    box-shadow: 0 6px 24px -4px rgba(15, 23, 42, 0.06) !important;
    margin-bottom: 18px !important;
}

/* =====================================================
   CUSTOM STYLED DATA TABLES (Pure White Background)
   ===================================================== */
.custom-table-container {
    max-height: 320px;
    overflow-y: auto;
    border: 1.5px solid #cbd5e1;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
    margin-bottom: 18px;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: #0f172a;
    background: #ffffff;
}
.custom-table thead {
    position: sticky;
    top: 0;
    background: #f1f5f9;
    z-index: 10;
}
.custom-table th {
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #334155;
    border-bottom: 2px solid #cbd5e1;
}
.custom-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #e2e8f0;
    color: #0f172a;
}
.custom-table tr:hover {
    background: #f8fafc;
}

/* Hero Header */
.hero-header {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 20px !important;
    padding: 26px 34px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 10px 32px -4px rgba(14, 165, 233, 0.08) !important;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    margin: 0 0 6px 0 !important;
}
.hero-subtitle {
    font-size: 14.5px !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    margin: 0 !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
}

/* Status Pill */
.pill {
    display: inline-block !important;
    padding: 4px 12px !important;
    border-radius: 9999px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}
.pill-green  { background: #d1fae5 !important; color: #065f46 !important; border:1px solid #6ee7b7 !important; }
.pill-blue   { background: #e0f2fe !important; color: #0369a1 !important; border:1px solid #7dd3fc !important; }
.pill-purple { background: #ede9fe !important; color: #5b21b6 !important; border:1px solid #c4b5fd !important; }
.pill-amber  { background: #fef3c7 !important; color: #92400e !important; border:1px solid #fcd34d !important; }
.pill-red    { background: #fee2e2 !important; color: #991b1b !important; border:1px solid #fca5a5 !important; }

/* Status Banners */
.consensus-bar {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-bottom: 14px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 13.5px !important;
    color: #334155 !important;
    box-shadow: 0 4px 16px -2px rgba(139, 92, 246, 0.08) !important;
}
.status-banner {
    background: #ffffff !important;
    border: 1.5px solid #bae6fd !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-bottom: 14px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 13.5px !important;
    color: #0369a1 !important;
    box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.08) !important;
}

/* Primary Action Button */
@keyframes floatAction {
    0% {
        transform: translateY(0px);
        box-shadow: 0 14px 36px -4px rgba(2, 132, 199, 0.45);
    }
    50% {
        transform: translateY(-6px);
        box-shadow: 0 28px 50px -6px rgba(2, 132, 199, 0.60);
    }
    100% {
        transform: translateY(0px);
        box-shadow: 0 14px 36px -4px rgba(2, 132, 199, 0.45);
    }
}

.stButton > button,
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 45%, #0369a1 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.7) !important;
    border-radius: 16px !important;
    padding: 18px 32px !important;
    min-height: 56px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    animation: floatAction 3.2s ease-in-out infinite !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover,
div[data-testid="stButton"] > button:hover {
    transform: translateY(-8px) scale(1.01) !important;
    box-shadow: 0 32px 56px -8px rgba(2, 132, 199, 0.70) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0284c7 !important;
    -webkit-text-fill-color: #0284c7 !important;
    border-bottom: 2.5px solid #0284c7 !important;
}

/* Dividers */
hr {
    border: none !important;
    height: 1.5px !important;
    background: #cbd5e1 !important;
    margin: 24px 0 !important;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    margin: 0 0 4px 0 !important;
}
.section-desc {
    font-size: 14px !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    margin: 0 0 16px 0 !important;
    font-weight: 500 !important;
}
</style>

<script>
// Force browser localStorage to light mode to prevent dark theme glitch on refresh
try {
    localStorage.setItem("stActiveTheme", "light");
    sessionStorage.setItem("stActiveTheme", "light");
} catch(e) {}
</script>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)


# =====================================================
# HELPER: BULLETPROOF METRIC CARD COMPONENT
# =====================================================
def metric_card_html(label, value, subtext=None):
    sub_html = f'<div class="metric-box-sub">{subtext}</div>' if subtext else ""
    return f"""
    <div class="metric-box">
        <div class="metric-box-label">{label}</div>
        <div class="metric-box-value">{value}</div>
        {sub_html}
    </div>
    """


# =====================================================
# HELPER: PLOTLY CHARTS (100% PURE WHITE BACKGROUND)
# =====================================================
def create_live_line_chart(df):
    fig = go.Figure()

    if "Waiting Time (s)" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Waiting Time (s)"],
            mode="lines",
            name="Waiting Time (s)",
            line=dict(color="#ef4444", width=2.5)
        ))

    if "Travel Time (s)" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Travel Time (s)"],
            mode="lines",
            name="Travel Time (s)",
            line=dict(color="#0284c7", width=2.5)
        ))

    if "Queue (veh)" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Queue (veh)"],
            mode="lines",
            name="Queue (veh)",
            line=dict(color="#8b5cf6", width=2.5)
        ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a", size=12),
        margin=dict(l=45, r=20, t=25, b=40),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#0f172a", size=11)
        ),
        xaxis=dict(
            title=dict(text="Simulation Time (seconds)", font=dict(color="#0f172a", size=12)),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(color="#0f172a")
        ),
        yaxis=dict(
            title=dict(text="Value", font=dict(color="#0f172a", size=12)),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(color="#0f172a")
        )
    )
    return fig


def create_benchmark_bar_chart(series, title, yaxis_title, bar_color="#0284c7"):
    fig = go.Figure(go.Bar(
        x=series.index,
        y=series.values,
        marker=dict(color=bar_color, line=dict(color="#ffffff", width=1.5)),
        text=[f"{v:.2f}" for v in series.values],
        textposition="outside",
        textfont=dict(color="#0f172a", size=11, family="Space Grotesk")
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=15, color="#0f172a")
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a"),
        margin=dict(l=45, r=20, t=50, b=90),
        height=380,
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(color="#0f172a", size=10.5),
            showgrid=False,
            linecolor="#cbd5e1"
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color="#0f172a", size=12)),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(color="#0f172a")
        )
    )
    return fig


# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <h1 class="hero-title">Autonomous Traffic Signal Command Center</h1>
            <p class="hero-subtitle">
                Real-Time SUMO 1.28.0 Digital Twin
                &middot; Hybrid Ensemble RL (DQN + Tabular Q-Learning)
                &middot; Priority Ambulance Preemption
            </p>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="pill pill-green">TraCI Connected</span>
            <span class="pill pill-blue">SUMO-GUI 1.28.0</span>
            <span class="pill pill-purple">Hybrid RL Engine</span>
            <span class="pill pill-red">Priority Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### Dispatch Controls")

    controller_choice = st.selectbox(
        "Policy Controller",
        [
            "Integrated Hybrid Ensemble (Triple-Source)",
            "Integrated DQN (Triple-Source)",
            "Integrated Q-Learning (Triple-Source)",
            "Baseline Hybrid Ensemble (Online Only)",
            "Baseline DQN (Online Only)",
            "Baseline Q-Learning (Online Only)",
            "Fixed-Time Baseline (30s Cycle)"
        ],
        index=0,
        help="Select which AI policy manages the traffic light phases. Integrated models combine online RL + archive delay priors + multi-scenario tasks."
    )

    is_hybrid = "Hybrid" in controller_choice

    if is_hybrid:
        st.markdown("#### Ensemble Weight Fusion")
        dqn_weight = st.slider(
            "DQN Weight Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.55,
            step=0.05,
            help="Balance between continuous 17-D neural network predictions and macroscopic tabular state."
        )
        ql_weight = 1.0 - dqn_weight
        st.caption(f"Weights: **{int(dqn_weight*100)}% DQN** / **{int(ql_weight*100)}% Q-Learning**")
        softmax_tau = st.slider(
            "Softmax Temperature",
            min_value=0.2,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher temperature smooths action probability distribution."
        )
    else:
        dqn_weight = 0.55
        softmax_tau = 1.0

    st.divider()

    st.markdown("#### Network Topology")
    network_choice = st.radio(
        "Intersections:",
        [
            "2 Intersections (Arterial Corridor: J1, J2)",
            "1 Intersection (Single Isolated)"
        ],
        index=0,
        help="Choose between a 2-junction interconnected corridor or a single isolated crossroads."
    )
    is_corridor = "2" in network_choice
    selected_network_mode = "corridor_2" if is_corridor else "single"

    st.divider()

    st.markdown("#### Simulation Traffic Mode")
    sim_mode = st.radio(
        "Flow Mode:",
        [
            "Infinite Non-Stop (Random Arrivals)",
            "600-Second Standard Benchmark"
        ],
        index=0
    )

    use_random_seed = st.checkbox("Randomize seed on each launch", value=True)
    if not use_random_seed:
        custom_seed = st.number_input("Custom Seed", min_value=1, max_value=999999, value=42)
    else:
        custom_seed = None

    st.divider()
    st.markdown("""
    <div style="font-size:12.5px; color:#334155; line-height:1.6;">
        <b>Why Combine Q-Learning and DQN?</b><br>
        &bull; <b>Q-Learning</b> provides deterministic stability on discrete queue pressure imbalances.<br>
        &bull; <b>DQN</b> optimizes continuous speed profiles and dense 17-D lane dynamics.<br>
        &bull; <b>Ensemble Softmax Fusion</b> arbitrates conflicts and eliminates single-model blind spots.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# SIMULATION LAUNCHER
# =====================================================
c_main, c_info = st.columns([2.2, 1.0])

with c_main:
    st.markdown('<h3 class="section-title">Real-Time Simulation Session</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Launch the native SUMO-GUI window on your desktop. Vehicles, lane stop-bars, and emergency ambulances animate in real time.</p>', unsafe_allow_html=True)

with c_info:
    is_infinite_run = "Infinite" in sim_mode
    max_duration_sec = None if is_infinite_run else 600
    st.info(
        f"**Topology:** `{'2-Intersection Corridor (J1 West, J2 East)' if is_corridor else '1-Intersection Isolated'}`\n\n"
        f"**Target Runtime:** {'Continuous' if is_infinite_run else '600 seconds'}\n\n"
        f"**Controller:** `{controller_choice.split('(')[0].strip()}`"
    )

start_sim = st.button("Launch Native SUMO-GUI Simulation", type="primary")

# Live placeholders
live_status = st.empty()
progress_bar = st.progress(0)
live_metrics = st.empty()
reward_display = st.empty()
consensus_display = st.empty()
chart_area = st.empty()
history_table = st.empty()

# =====================================================
# CONTROLLER EXECUTION
# =====================================================
if start_sim:

    import importlib
    import environment.traffic_environment
    importlib.reload(environment.traffic_environment)
    from environment.traffic_environment import TrafficEnvironment
    from evaluation.metrics import TrafficMetrics
    from agents.dqn_agent import DQNAgent
    from agents.q_learning import QLearningAgent
    from agents.hybrid_agent import HybridAgent
    from agents.hybrid_integrated_agent import IntegratedHybridAgent

    selected_seed = (
        random.SystemRandom().randint(1, 100000)
        if use_random_seed or custom_seed is None
        else int(custom_seed)
    )

    live_status.info(
        f"Initializing {controller_choice} ... SUMO-GUI window is opening. (Seed: {selected_seed})"
    )

    # Instantiate agent
    if "Integrated Hybrid" in controller_choice:
        agent = IntegratedHybridAgent(
            dqn_model="dqn_integrated_model.pth",
            ql_model="q_learning_integrated_model.pkl",
            dqn_weight=dqn_weight,
            temperature=softmax_tau
        )
        is_hybrid_mode = True
    elif "Integrated DQN" in controller_choice:
        agent = DQNAgent()
        agent.load_model("dqn_integrated_model.pth")
        agent.epsilon = 0.0
        is_hybrid_mode = False
    elif "Integrated Q-Learning" in controller_choice:
        agent = QLearningAgent()
        agent.load_model("q_learning_integrated_model.pkl")
        agent.epsilon = 0.0
        is_hybrid_mode = False
    elif "Baseline Hybrid" in controller_choice or "Hybrid" in controller_choice:
        agent = HybridAgent(
            dqn_model="dqn_ambulance_model_updated.pth",
            ql_model="q_learning_ambulance_model_updated.pkl",
            dqn_weight=dqn_weight,
            temperature=softmax_tau
        )
        is_hybrid_mode = True
    elif "DQN" in controller_choice:
        agent = DQNAgent()
        agent.load_model("dqn_ambulance_model_updated.pth")
        agent.epsilon = 0.0
        is_hybrid_mode = False
    elif "Q-Learning" in controller_choice:
        agent = QLearningAgent()
        agent.load_model("q_learning_ambulance_model_updated.pkl")
        agent.epsilon = 0.0
        is_hybrid_mode = False
    else:
        agent = None
        is_hybrid_mode = False

    env = TrafficEnvironment(
        use_gui=True,
        max_episode_time=max_duration_sec,
        network_mode=selected_network_mode
    )

    chart_history = []
    step_history = []
    consensus_stats = {"agreements": 0, "disagreements": 0}

    reward_data = {
        "total_reward": 0.0,
        "last_reward": 0.0,
        "override_steps": 0,
        "action_counts": {"North-South Green": 0, "East-West Green": 0}
    }

    if max_duration_sec is not None:
        progress_bar.progress(0)
    else:
        progress_bar.empty()

    def update_live(results):
        sim_time = results["simulation_time"]
        if max_duration_sec is not None:
            progress_bar.progress(min(int(sim_time / max_duration_sec * 100), 100))

        if int(sim_time) % 10 == 0 or sim_time == 1:
            try:
                active_vehicles = len(traci.vehicle.getIDList())
                primary_tl = env.traffic_light_ids[0] if env.traffic_light_ids else "center"
                current_phase = traci.trafficlight.getPhase(primary_tl)
                phase_names = {
                    0: "North-South Green",
                    1: "North-South Yellow",
                    2: "East-West Green (Corridor Flow)",
                    3: "East-West Yellow"
                }
                signal_status = phase_names.get(current_phase, "Dynamic")
            except Exception:
                active_vehicles = 0
                signal_status = "Unavailable"

            time_str = (
                f"{sim_time:.0f}s (Non-stop)"
                if max_duration_sec is None
                else f"{sim_time:.0f} / {max_duration_sec:.0f}s"
            )

            live_status.markdown(f"""
            <div class="status-banner">
                <span style="font-weight:600; color:#0369a1;">{controller_choice.split('(')[0].strip()} &mdash; Active</span>
                <span style="font-family:'JetBrains Mono',monospace; color:#0f172a;">Time: {time_str}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#047857;">Signal: {signal_status}</span>
            </div>
            """, unsafe_allow_html=True)

            with live_metrics.container():
                st.markdown('<h4 class="section-title" style="font-size:18px;">Live Intersection Telemetry</h4>', unsafe_allow_html=True)
                
                # High-contrast custom metric cards Row 1
                r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
                r1_c1.markdown(metric_card_html("Average Waiting Time", f"{results['average_waiting_time']:.2f} sec"), unsafe_allow_html=True)
                r1_c2.markdown(metric_card_html("Average Travel Time", f"{results['average_travel_time']:.2f} sec"), unsafe_allow_html=True)
                r1_c3.markdown(metric_card_html("Average Queue Length", f"{results['average_queue_length']:.2f}"), unsafe_allow_html=True)
                r1_c4.markdown(metric_card_html("Completed Vehicles", f"{results['completed_vehicles']}"), unsafe_allow_html=True)

                # High-contrast custom metric cards Row 2
                r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
                r2_c1.markdown(metric_card_html("Active Vehicles", f"{active_vehicles}"), unsafe_allow_html=True)
                r2_c2.markdown(metric_card_html("Ambulance Avg Waiting", f"{results['average_ambulance_waiting_time']:.2f} sec"), unsafe_allow_html=True)
                r2_c3.markdown(metric_card_html("Maximum Ambulance Waiting", f"{results['maximum_ambulance_waiting_time']:.2f} sec"), unsafe_allow_html=True)
                r2_c4.markdown(metric_card_html("Observed Vehicles", f"{results['observed_vehicles']}"), unsafe_allow_html=True)

            chart_history.append({
                "Time (s)": sim_time,
                "Waiting Time (s)": results["average_waiting_time"],
                "Travel Time (s)": results["average_travel_time"],
                "Queue (veh)": results["average_queue_length"],
            })

            if chart_history:
                chart_df = pd.DataFrame(chart_history).set_index("Time (s)")
                fig_live = create_live_line_chart(chart_df)
                chart_area.plotly_chart(fig_live, use_container_width=True, theme=None)

    metrics = TrafficMetrics(
        on_update=update_live,
        incoming_edges=env.monitored_incoming_edges
    )

    try:
        state = env.reset(seed=selected_seed)
        done = False
        fixed_time_counter = 0

        while not done:
            decision_info = None

            if agent is None:
                action = 0 if (fixed_time_counter % 2 == 0) else 1
                fixed_time_counter += 1
            elif is_hybrid_mode:
                decision_info = agent.get_detailed_decision(state)
                action = decision_info["action"]
                if decision_info["consensus"]:
                    consensus_stats["agreements"] += 1
                else:
                    consensus_stats["disagreements"] += 1
            else:
                action = agent.choose_action(state)

            if action == 0:
                reward_data["action_counts"]["North-South Green"] += 1
            else:
                reward_data["action_counts"]["East-West Green"] += 1

            next_state, reward, done = env.step(action, metrics=metrics)
            reward_data["last_reward"] = reward
            reward_data["total_reward"] += reward

            if env.emergency_override_during_step:
                reward_data["override_steps"] += 1

            # High contrast reward metric box
            rew_c1, _ = st.columns([1, 3])
            with rew_c1:
                sub_label = "Positive clearance bonus" if reward > 0 else "Queue/delay penalty"
                reward_display.markdown(
                    metric_card_html(f"{controller_choice.split('(')[0].strip()} Reward", f"{reward:.2f}", sub_label),
                    unsafe_allow_html=True
                )

            # Consensus banner for hybrid mode
            if is_hybrid_mode and decision_info:
                total_dec = consensus_stats["agreements"] + consensus_stats["disagreements"]
                agree_pct = (consensus_stats["agreements"] / max(total_dec, 1)) * 100
                agree_label = "AGREEMENT" if decision_info["consensus"] else "DISAGREEMENT (Arbitrated)"
                consensus_display.markdown(f"""
                <div class="consensus-bar">
                    <div><b>Ensemble Decision:</b> {agree_label} &middot; DQN: Phase {decision_info['dqn_action']} | QL: Phase {decision_info['ql_action']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;"><b>Consensus Rate:</b> {agree_pct:.1f}% ({consensus_stats['agreements']}/{total_dec})</div>
                </div>
                """, unsafe_allow_html=True)

            # Step history table
            step_results = metrics.get_results()
            try:
                active_vehicles = len(traci.vehicle.getIDList())
                current_queue = sum(
                    traci.edge.getLastStepHaltingNumber(edge)
                    for edge in env.monitored_incoming_edges
                )
                primary_tl = env.traffic_light_ids[0] if env.traffic_light_ids else "center"
                current_phase = traci.trafficlight.getPhase(primary_tl)
            except Exception:
                active_vehicles = 0
                current_queue = 0
                current_phase = -1

            phase_map = {0: "N-S Green", 1: "N-S Yellow", 2: "E-W Green", 3: "E-W Yellow"}

            step_history.append({
                "Time (s)": round(step_results["simulation_time"], 1),
                "Reward": round(reward, 2),
                "Active Veh": active_vehicles,
                "Wait Time (s)": round(step_results["average_waiting_time"], 2),
                "Queue": current_queue,
                "Phase": phase_map.get(current_phase, "Dynamic"),
                "Amb Wait (s)": round(step_results["average_ambulance_waiting_time"], 2),
                "Preemption": "ACTIVE" if env.emergency_override_during_step else "-"
            })

            # Render custom pure-white responsive HTML table (shows recent 25 steps)
            recent_steps = step_history[-25:][::-1]
            rows_html = "".join([
                f"""<tr>
                    <td><b>{r['Time (s)']}</b></td>
                    <td style="color:{'#059669' if r['Reward'] > 0 else '#dc2626'}; font-weight:700;">{r['Reward']:+.2f}</td>
                    <td>{r['Active Veh']}</td>
                    <td>{r['Wait Time (s)']}s</td>
                    <td>{r['Queue']}</td>
                    <td><span style="background:#e0f2fe; padding:2px 6px; border-radius:4px; font-weight:600;">{r['Phase']}</span></td>
                    <td>{r['Amb Wait (s)']}s</td>
                    <td style="color:{'#dc2626' if r['Preemption'] == 'ACTIVE' else '#64748b'}; font-weight:700;">{r['Preemption']}</td>
                </tr>"""
                for r in recent_steps
            ])

            table_markup = f"""
            <div class="custom-table-container">
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Time (s)</th>
                            <th>Reward</th>
                            <th>Active Veh</th>
                            <th>Wait Time</th>
                            <th>Queue</th>
                            <th>Phase</th>
                            <th>Amb Wait</th>
                            <th>Preemption</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """
            history_table.markdown(table_markup, unsafe_allow_html=True)

            state = next_state

    except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
        live_status.info("Simulation closed by user or window exited.")
    except Exception as ex:
        live_status.error(f"Simulation terminated: {ex}")
    finally:
        env.close()

    live_status.success("Simulation complete. Telemetry recorded successfully.")


# =====================================================
# 5-SEED BENCHMARK MATRIX
# =====================================================
st.divider()
st.markdown("""
<h3 class="section-title">5-Seed Cross-Validation Benchmark Matrix</h3>
<p class="section-desc">Empirical evaluation across 5 pseudo-random seeds under stochastic Bernoulli vehicle and ambulance generation.</p>
""", unsafe_allow_html=True)

CSV_PATH = "results/metrics/controller_multi_seed_summary.csv"

if os.path.exists(CSV_PATH):
    raw = pd.read_csv(CSV_PATH)
    df = raw.pivot(index="Controller", columns="Metric", values=["Mean", "Std Dev"])
    df.columns = [f"{metric}_{stat}" for stat, metric in df.columns]
    df = df.reset_index()

    best_wait = df.loc[df["average_waiting_time_Mean"].idxmin()]
    best_queue = df.loc[df["average_queue_length_Mean"].idxmin()]
    best_amb = df.loc[df["average_ambulance_waiting_time_Mean"].idxmin()]
    max_throughput = df["completed_vehicles_Mean"].max()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(metric_card_html("Lowest Mean Wait Time", f"{best_wait['average_waiting_time_Mean']:.2f} sec", f"Best: {best_wait['Controller']}"), unsafe_allow_html=True)
    kpi2.markdown(metric_card_html("Lowest Mean Queue", f"{best_queue['average_queue_length_Mean']:.2f} veh", f"Best: {best_queue['Controller']}"), unsafe_allow_html=True)
    kpi3.markdown(metric_card_html("Lowest Ambulance Wait", f"{best_amb['average_ambulance_waiting_time_Mean']:.2f} sec", f"Best: {best_amb['Controller']}"), unsafe_allow_html=True)
    kpi4.markdown(metric_card_html("Peak Completed Flow", f"{max_throughput:.1f} veh", "5-Seed Average Throughput"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Performance Charts", "Benchmark Table", "Model Architecture"])

    with tab1:
        c_left, c_right = st.columns(2)
        with c_left:
            series_wait = df.set_index("Controller")["average_waiting_time_Mean"]
            fig_wait = create_benchmark_bar_chart(
                series_wait,
                "Vehicle Waiting Time Comparison (Lower is Better)",
                "Average Waiting Time (seconds)",
                bar_color="#0284c7"
            )
            st.plotly_chart(fig_wait, use_container_width=True, theme=None)

        with c_right:
            series_amb = df.set_index("Controller")["average_ambulance_waiting_time_Mean"]
            fig_amb = create_benchmark_bar_chart(
                series_amb,
                "Emergency Ambulance Delay (Lower is Better)",
                "Ambulance Waiting Time (seconds)",
                bar_color="#ef4444"
            )
            st.plotly_chart(fig_amb, use_container_width=True, theme=None)

    with tab2:
        display_df = df[[
            "Controller",
            "average_waiting_time_Mean",
            "average_waiting_time_Std Dev",
            "average_queue_length_Mean",
            "average_queue_length_Std Dev",
            "completed_vehicles_Mean",
            "average_ambulance_waiting_time_Mean"
        ]].copy()

        rows_bench = "".join([
            f"""<tr>
                <td style="font-weight:700; color:#0f172a;">{row['Controller']}</td>
                <td style="font-weight:700; color:#0284c7;">{row['average_waiting_time_Mean']:.2f}s</td>
                <td style="color:#64748b;">&plusmn;{row['average_waiting_time_Std Dev']:.2f}</td>
                <td style="font-weight:600;">{row['average_queue_length_Mean']:.2f}</td>
                <td style="color:#64748b;">&plusmn;{row['average_queue_length_Std Dev']:.2f}</td>
                <td>{row['completed_vehicles_Mean']:.1f}</td>
                <td style="font-weight:700; color:{'#059669' if row['average_ambulance_waiting_time_Mean'] < 5.0 else '#dc2626'};">{row['average_ambulance_waiting_time_Mean']:.2f}s</td>
            </tr>"""
            for _, row in display_df.iterrows()
        ])

        bench_table_markup = f"""
        <div class="custom-table-container" style="max-height:480px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Controller Policy</th>
                        <th>Mean Wait</th>
                        <th>Wait Std</th>
                        <th>Mean Queue</th>
                        <th>Queue Std</th>
                        <th>Completed Flow</th>
                        <th>Ambulance Wait</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_bench}
                </tbody>
            </table>
        </div>
        """
        st.markdown(bench_table_markup, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        #### Why Hybrid Ensemble Outperforms Individual Models

        **Architecture:** The Hybrid agent fuses Tabular Q-Learning (216 discrete states) with a 3-layer Deep Q-Network (17-D continuous input, 128-128 hidden units) using weighted Boltzmann softmax probability fusion.

        **Macroscopic + Microscopic Synthesis:**
        - Tabular Q-Learning evaluates macroscopic queue balance, arrival pressure, and active phase index with zero neural approximation error.
        - Deep Q-Network captures continuous non-linear dynamics across 17 real-valued dimensions (approach speeds, lane-level densities, remaining phase durations).

        **Variance Reduction:**
        - Ensemble probability: P_ensemble(a) = w_dqn * Softmax(Q_dqn / tau) + w_ql * Softmax(Q_ql / tau)
        - When both agents agree, decision confidence approaches 100%.
        - When they disagree, the agent with larger action margin arbitrates.

        **Emergency Preemption:**
        - Emergency vehicles (vClass="emergency") trigger immediate safe 3s yellow transition followed by green preemption via TraCI.
        """)
else:
    st.warning("5-seed benchmark results CSV not found. Run evaluation first.")


# =====================================================
# FOOTER
# =====================================================
st.divider()

st.markdown("""
<div style="text-align:center; padding:12px 0 4px 0;">
    <span style="font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace; letter-spacing:0.04em;">
        Autonomous Traffic Signal Optimization &middot; SUMO + TraCI + PyTorch + Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
