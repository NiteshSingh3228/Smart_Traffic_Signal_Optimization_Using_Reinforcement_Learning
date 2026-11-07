
import os
import sys
import random
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

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

from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment


# =====================================
# REPRODUCIBILITY
# =====================================

random.seed(0)
torch.manual_seed(0)


# =====================================
# TRAINING SETTINGS
# =====================================

NUM_EPISODES = 200

agent = DQNAgent()
try:
    agent.load_model("dqn_ambulance_model.pth")
except (FileNotFoundError, KeyError, ValueError):
    print("No compatible checkpoint found. Starting DQN training from scratch.")

env = TrafficEnvironment(use_gui=False)


# =====================================
# DQN TRAINING
# =====================================

for episode in range(NUM_EPISODES):

    print(f"\n========== EPISODE {episode + 1} ==========")

    # FIX 6: Use seeds outside evaluation range (42, 123 used for eval)
    state = env.reset(seed=1000 + episode)

    total_reward = 0
    total_loss = 0
    training_updates = 0
    ambulance_override_steps = 0

    done = False
    step_count = 0

    try:

        while not done:

            # 1. Select an action
            action = agent.choose_action(state)

            # 2. Execute action in SUMO
            next_state, reward, done = env.step(action)

            # 3. Store experience and train
            # FIX: Always store and learn — reward now includes
            # ambulance penalty, so the agent learns from all steps
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.learn()

            if loss is not None:
                total_loss += loss
                training_updates += 1

            if env.emergency_override_during_step:
                ambulance_override_steps += 1

            # 4. Move to next state
            state = next_state

            total_reward += reward
            step_count += 1

        # =================================
        # EPISODE RESULTS
        # =================================

        print(f"\nEpisode: {episode + 1}")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Replay memory: {len(agent.memory)}")
        print(f"Training updates: {training_updates}")
        print(f"Ambulance override steps: {ambulance_override_steps}")

        average_loss = (
            total_loss / training_updates
            if training_updates > 0 else 0
        )

        print(f"Average loss: {average_loss:.4f}")

        # Reduce exploration once per episode
        agent.decay_epsilon()

        print(f"Epsilon: {agent.epsilon:.4f}")

    finally:
        env.close()


# =====================================
# SAVE AMBULANCE-AWARE MODEL
# =====================================

agent.save_model("dqn_ambulance_model_updated.pth")


# =====================================
# TRAINING SUMMARY
# =====================================

print("\n========== DQN TRAINING COMPLETED ==========")

print(f"Total episodes: {NUM_EPISODES}")
print(f"Replay memory size: {len(agent.memory)}")
print(f"Final epsilon: {agent.epsilon:.4f}")
