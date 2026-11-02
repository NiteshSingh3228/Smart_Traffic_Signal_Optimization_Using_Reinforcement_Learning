
from agents.dqn_agent import DQNAgent

agent = DQNAgent()

state = (2, 1, 3, 2, 1, 0, 4, 2, 0)

action = agent.choose_action(state)

print("Selected action:", action)

print("Action meaning:")

if action == 0:
    print("North-South Green")
else:
    print("East-West Green")

print("Memory size:", len(agent.memory))

print("DQN agent created successfully!")
