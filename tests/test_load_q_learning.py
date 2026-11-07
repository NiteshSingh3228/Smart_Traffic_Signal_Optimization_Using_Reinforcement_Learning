
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.q_learning import QLearningAgent

# Create a new agent
agent = QLearningAgent()

print("Q-table before loading:")
print(agent.q_table)

# FIX: Load the current model (ambulance-aware), not the old 9-dim model
try:
    agent.load_model("q_learning_ambulance_model_updated.pkl")
    print("\nLoaded: q_learning_ambulance_model_updated.pkl")
except FileNotFoundError:
    try:
        agent.load_model("q_learning_ambulance_model.pkl")
        print("\nLoaded: q_learning_ambulance_model.pkl")
    except FileNotFoundError:
        print("\nNo model file found — using empty Q-table.")

print("\nTotal learned states:", len(agent.q_table))
print("Epsilon:", agent.epsilon)

print("\nModel loading test completed!")
