
from agents.q_learning import QLearningAgent

# Create Q-Learning agent
agent = QLearningAgent()

# Sample traffic state
state = (4, 2, 3, 1, 5, 2, 2, 1, 0)

print("Initial Q-table:", agent.q_table)

# Select an action
action = agent.choose_action(state)

print("\nSelected action:", action)

if action == 0:
    print("Selected signal: North-South Green")
else:
    print("Selected signal: East-West Green")

# Sample next traffic state
next_state = (2, 1, 3, 1, 2, 1, 1, 0, 0)

# Sample reward
reward = 2

# Update Q-table
agent.learn(
    state,
    action,
    reward,
    next_state,
    done=False
)

print("\nUpdated Q-table:", agent.q_table)

# Reduce exploration
agent.decay_epsilon()

print("\nUpdated epsilon:", agent.epsilon)
