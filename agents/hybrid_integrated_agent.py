"""
Triple-Source Integrated Hybrid Ensemble Agent
Combines:
- Integrated Deep Q-Network (dqn_integrated_model.pth)
- Integrated Tabular Q-Learning (q_learning_integrated_model.pkl)

Benefits from:
1. Source 1: Baseline online RL experience & ambulance preemption
2. Source 2: Archive macro/micro delay priors & offline replay
3. Source 3: Multi-scenario flow regime calibration

Fuses policy probabilities via Boltzmann Softmax arbitration.
"""

import math
import torch
from agents.dqn_agent import DQNAgent, DEVICE
from agents.q_learning import QLearningAgent


class IntegratedHybridAgent:
    def __init__(
        self,
        dqn_model="dqn_integrated_model.pth",
        ql_model="q_learning_integrated_model.pkl",
        dqn_weight=0.55,
        temperature=1.0,
    ):
        self.dqn_weight = dqn_weight
        self.ql_weight = 1.0 - dqn_weight
        self.temperature = max(temperature, 0.01)
        self.epsilon = 0.0

        # Initialize underlying agents with integrated checkpoints
        self.dqn_agent = DQNAgent()
        self.dqn_agent.load_model(dqn_model)
        self.dqn_agent.epsilon = 0.0

        self.ql_agent = QLearningAgent()
        self.ql_agent.load_model(ql_model)
        self.ql_agent.epsilon = 0.0

    @staticmethod
    def _softmax(values, temperature=1.0):
        shifted = [v / temperature for v in values]
        max_val = max(shifted)
        exp_vals = [math.exp(v - max_val) for v in shifted]
        total = sum(exp_vals)
        return [e / total for e in exp_vals]

    def get_detailed_decision(self, state):
        # 1. DQN Q-values
        state_tensor = self.dqn_agent.prepare_state(state).unsqueeze(0)
        with torch.no_grad():
            dqn_q_tensor = self.dqn_agent.policy_network(state_tensor).squeeze(0)
            dqn_q = [float(v) for v in dqn_q_tensor.tolist()]

        dqn_probs = self._softmax(dqn_q, self.temperature)
        dqn_action = 0 if dqn_q[0] >= dqn_q[1] else 1

        # 2. Q-Learning Q-values
        discrete_state = self.ql_agent.discretize_state(state)
        ql_q_raw = self.ql_agent.get_q_values(discrete_state)
        ql_q = [float(ql_q_raw[0]), float(ql_q_raw[1])]
        ql_probs = self._softmax(ql_q, self.temperature)
        ql_action = 0 if ql_q[0] >= ql_q[1] else 1

        # 3. Ensemble Weighted Fusion
        ensemble_p0 = self.dqn_weight * dqn_probs[0] + self.ql_weight * ql_probs[0]
        ensemble_p1 = self.dqn_weight * dqn_probs[1] + self.ql_weight * ql_probs[1]
        ensemble_probs = [ensemble_p0, ensemble_p1]

        final_action = 0 if ensemble_p0 >= ensemble_p1 else 1
        consensus = (dqn_action == ql_action)
        confidence = abs(ensemble_p0 - ensemble_p1)

        return {
            "action": final_action,
            "consensus": consensus,
            "confidence": confidence,
            "dqn_action": dqn_action,
            "ql_action": ql_action,
            "dqn_q": dqn_q,
            "ql_q": ql_q,
            "dqn_probs": dqn_probs,
            "ql_probs": ql_probs,
            "ensemble_probs": ensemble_probs,
        }

    def choose_action(self, state):
        decision = self.get_detailed_decision(state)
        return decision["action"]

