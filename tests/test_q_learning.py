from agents.q_learning import QLearningAgent

def test_q_learning_agent():

    agent = QLearningAgent()

    # 17-value traffic state
    state = (
        4, 2, 3, 1, 5, 2, 2, 1,
        0, 1, 0, 0,
        0, 5, 0, 0,
        0
    )

    next_state = (
        2, 1, 3, 1, 2, 1, 1, 0,
        0, 1, 0, 0,
        0, 3, 0, 0,
        1
    )

    # Test action selection
    action = agent.choose_action(state)

    assert action in [0, 1]

    # Test Q-learning update
    agent.learn(
        state,
        action,
        reward=2,
        next_state=next_state,
        done=False
    )

    discrete_state = agent.discretize_state(state)

    assert discrete_state in agent.q_table

    # Test epsilon decay
    old_epsilon = agent.epsilon
    agent.decay_epsilon()

    assert agent.epsilon <= old_epsilon

    print("Q-Learning test passed!")


if __name__ == "__main__":
    test_q_learning_agent()
