
from agents.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment


# =====================================
# TRAINING SETTINGS
# =====================================

NUM_EPISODES = 100

agent = DQNAgent()

env = TrafficEnvironment(use_gui=False)


# =====================================
# DQN TRAINING
# =====================================

for episode in range(NUM_EPISODES):

    print(f"\n========== EPISODE {episode + 1} ==========")

    state = env.reset(seed=42 + episode)

    total_reward = 0
    total_loss = 0
    training_updates = 0
    skipped_updates = 0

    done = False
    step_count = 0

    try:

        while not done:

            # 1. Select an action
            action = agent.choose_action(state)

            # 2. Execute action in SUMO
            next_state, reward, done = env.step(action)

            # 3. Check whether ambulance priority overrode the RL action
            if not env.emergency_override_during_step:

                # 4. Store valid experience
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                # 5. Train neural network
                loss = agent.learn()

                if loss is not None:
                    total_loss += loss
                    training_updates += 1

            else:

                # Do not learn from an overridden action
                skipped_updates += 1

                print(
                    "Skipping DQN update:",
                    "Ambulance priority override occurred."
                )

            # 6. Move to next state
            state = next_state

            total_reward += reward
            step_count += 1

        # =================================
        # EPISODE RESULTS
        # =================================

        print("\nEpisode:", episode + 1)
        print("Total steps:", step_count)
        print("Total reward:", total_reward)
        print("Replay memory:", len(agent.memory))
        print("Training updates:", training_updates)
        print("Skipped updates:", skipped_updates)

        average_loss = (
            total_loss / training_updates
            if training_updates > 0 else 0
        )

        print("Average loss:", round(average_loss, 4))

        # Reduce exploration once per episode
        agent.decay_epsilon()

        print("Epsilon:", round(agent.epsilon, 4))

    finally:
        env.close()


# =====================================
# SAVE AMBULANCE-AWARE MODEL
# =====================================

agent.save_model("dqn_ambulance_model.pth")


# =====================================
# TRAINING SUMMARY
# =====================================

print("\n========== DQN TRAINING COMPLETED ==========")

print("Total episodes:", NUM_EPISODES)
print("Replay memory size:", len(agent.memory))
print("Final epsilon:", round(agent.epsilon, 4))
