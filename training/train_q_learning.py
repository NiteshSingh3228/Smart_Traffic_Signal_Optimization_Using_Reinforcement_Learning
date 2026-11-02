
from agents.q_learning import QLearningAgent
from environment.traffic_environment import TrafficEnvironment


# ======================================
# TRAINING CONFIGURATION
# ======================================

NUM_EPISODES = 100

agent = QLearningAgent()

env = TrafficEnvironment(use_gui=False)


# ======================================
# TRAINING LOOP
# ======================================

for episode in range(NUM_EPISODES):

    print(f"\n========== EPISODE {episode + 1} ==========")

    # Use a different seed for each episode
    state = env.reset(seed=42 + episode)

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

            # 3. Check ambulance priority override
            if not env.emergency_override_during_step:

                # 4. Learn from valid experience
                agent.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                training_updates += 1

            else:

                # Skip learning from overridden action
                skipped_updates += 1

                print(
                    "Skipping Q-Learning update:",
                    "Ambulance priority override occurred."
                )

            # 5. Move to next state
            state = next_state

            total_reward += reward
            step_count += 1

        # =================================
        # EPISODE RESULTS
        # =================================

        print("Episode:", episode + 1)
        print("Total steps:", step_count)
        print("Total reward:", total_reward)
        print("Training updates:", training_updates)
        print("Skipped updates:", skipped_updates)

    finally:

        env.close()

    # Reduce exploration
    agent.decay_epsilon()

    print("Epsilon:", round(agent.epsilon, 4))


# ======================================
# SAVE AMBULANCE-AWARE MODEL
# ======================================

agent.save_model("q_learning_ambulance_model.pkl")


# ======================================
# TRAINING SUMMARY
# ======================================

print("\n========== Q-LEARNING TRAINING COMPLETED ==========")

print("Total episodes:", NUM_EPISODES)
print("Total learned states:", len(agent.q_table))
print("Final epsilon:", round(agent.epsilon, 4))
