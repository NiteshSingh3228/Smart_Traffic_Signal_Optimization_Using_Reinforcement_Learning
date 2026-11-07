
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.dqn_agent import DQNAgent

agent = DQNAgent()

state = (
    2, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0,
    20, 0, 0, 0,
    0
)

print("Original state:", state)
print("State length:", len(state))

normalized_state = agent.normalize_state(state)

print("\nNormalized state:", normalized_state)
print("Normalized state length:", len(normalized_state))

state_tensor = agent.prepare_state(state)

print("\nTensor shape:", state_tensor.shape)

action = agent.choose_action(state)

print("\nSelected action:", action)

print("\nDQN test completed successfully!")
