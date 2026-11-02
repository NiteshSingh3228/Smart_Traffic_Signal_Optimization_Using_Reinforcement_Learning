
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
