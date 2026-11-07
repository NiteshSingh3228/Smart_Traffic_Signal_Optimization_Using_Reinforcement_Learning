
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.dqn_agent import DQNAgent

agent = DQNAgent()

# FIX 1.1: Use a valid 17-element state vector
# (8 traffic values, 4 ambulance counts, 4 ambulance wait times, 1 phase)
state = (
    2, 1, 3, 2, 1, 0, 4, 2,
    0, 1, 0, 0,
    0, 8, 0, 0,
    0
)

action = agent.choose_action(state)

print("Selected action:", action)

print("Action meaning:")

if action == 0:
    print("North-South Green")
else:
    print("East-West Green")

print("Memory size:", len(agent.memory))

# FIX: Also test with epsilon=0 to exercise the neural network path
agent.epsilon = 0

action_exploit = agent.choose_action(state)

print("Exploit action (epsilon=0):", action_exploit)

print("DQN agent created and tested successfully!")
