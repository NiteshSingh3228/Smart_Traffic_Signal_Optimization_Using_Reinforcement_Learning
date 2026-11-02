
from agents.q_learning import QLearningAgent

# Create a new agent
agent = QLearningAgent()

print("Q-table before loading:")
print(agent.q_table)

# Load the saved model
agent.load_model("q_learning_model.pkl")

print("\nQ-table after loading:")
print(agent.q_table)

print("\nTotal learned states:", len(agent.q_table))

print("\nModel loading test completed!")
