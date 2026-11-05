
from agents.q_learning import QLearningAgent
from environment.traffic_environment import TrafficEnvironment


# ======================================
# TRAINING CONFIGURATION
# ======================================

NUM_EPISODES = 200

agent = QLearningAgent()

# FIX 2.2: load_model now also restores epsilon from checkpoint
agent.load_model("q_learning_ambulance_model.pkl")

env = TrafficEnvironment(use_gui=False)


# ======================================
# TRAINING LOOP
# ======================================

for episode in range(NUM_EPISODES):

    # FIX 6: Use seeds outside the evaluation range (42-123 used for eval)
    # Training seeds start at 1000 to avoid overlap
    state = env.reset(seed=1000 + episode)

    total_reward = 0
    done = False
    step_count = 0
    training_updates = 0
    skipped_updates = 0

    try:

        while not done:

            # 1. Select action
            action = agent.choose_action(state)

            # 2. Execute action in SUMO
            next_state, reward, done = env.step(action)

            # 3. Always learn — even during ambulance override steps
            # The reward function now encodes ambulance urgency
            # so the agent CAN learn from ambulance-influenced transitions
            agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            training_updates += 1

            if env.emergency_override_during_step:
                skipped_updates += 1

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
        print(f"Training updates: {training_updates}")
        print(f"Ambulance override steps: {skipped_updates}")

    finally:

        env.close()

    # Reduce exploration
    agent.decay_epsilon()

    print(f"Epsilon: {agent.epsilon:.4f}")


# ======================================
# SAVE AMBULANCE-AWARE MODEL
# FIX 2.2: epsilon is now saved inside save_model()
# ======================================

agent.save_model("q_learning_ambulance_model_updated.pkl")


# ======================================
# TRAINING SUMMARY
# ======================================

print("\n========== Q-LEARNING TRAINING COMPLETED ==========")

print(f"Total episodes: {NUM_EPISODES}")
print(f"Total learned states: {len(agent.q_table)}")
print(f"Final epsilon: {agent.epsilon:.4f}")
