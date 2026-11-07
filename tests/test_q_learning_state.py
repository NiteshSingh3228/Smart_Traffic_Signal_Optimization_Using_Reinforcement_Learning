
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.q_learning import QLearningAgent

agent = QLearningAgent()

# Example 17-value state
state = (
    2, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0,
    20, 0, 0, 0,
    0
)

discrete_state = agent.discretize_state(state)

print("Original state:", state)
print("Original state length:", len(state))

print("\nDiscretized state:", discrete_state)
print("Discretized state length:", len(discrete_state))

action = agent.choose_action(state)

print("\nSelected action:", action)
