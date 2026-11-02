
import random
import pickle
import os


STATE_SIZE = 17


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05
    ):

        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.actions = [0, 1]

        self.q_table = {}

    # ======================================
    # DISCRETIZE STATE
    # ======================================

    def discretize_state(self, state):

        if len(state) != STATE_SIZE:
            raise ValueError(
                f"Expected {STATE_SIZE} state values, got {len(state)}"
            )

        discrete_state = []

        # 1. Normal traffic values (first 8 values)
        for value in state[:8]:

            if value == 0:
                discrete_state.append(0)

            elif value <= 2:
                discrete_state.append(1)

            elif value <= 5:
                discrete_state.append(2)

            else:
                discrete_state.append(3)

        # 2. Ambulance counts (next 4 values)
        for value in state[8:12]:

            if value == 0:
                discrete_state.append(0)

            elif value == 1:
                discrete_state.append(1)

            elif value <= 3:
                discrete_state.append(2)

            else:
                discrete_state.append(3)

        # 3. Ambulance waiting times (next 4 values)
        for value in state[12:16]:

            if value == 0:
                discrete_state.append(0)

            elif value <= 5:
                discrete_state.append(1)

            elif value <= 15:
                discrete_state.append(2)

            else:
                discrete_state.append(3)

        # 4. Current traffic signal phase
        discrete_state.append(state[16])

        return tuple(discrete_state)

    # ======================================
    # GET Q-VALUES
    # ======================================

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]

        return self.q_table[state]

    # ======================================
    # CHOOSE ACTION
    # ======================================

    def choose_action(self, state):

        state = self.discretize_state(state)

        if random.random() < self.epsilon:

            action = random.choice(self.actions)

        else:

            q_values = self.get_q_values(state)

            action = q_values.index(max(q_values))

        return action

    # ======================================
    # LEARN
    # ======================================

    def learn(self, state, action, reward, next_state, done):

        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        current_q = self.get_q_values(state)[action]

        if done:

            max_future_q = 0

        else:

            max_future_q = max(
                self.get_q_values(next_state)
            )

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

        self.q_table[state][action] = new_q

    # ======================================
    # DECAY EPSILON
    # ======================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    # ======================================
    # SAVE Q-TABLE
    # ======================================

    def save_model(self, filename):

        os.makedirs(
            "results/models",
            exist_ok=True
        )

        filepath = os.path.join(
            "results/models",
            filename
        )

        with open(filepath, "wb") as file:

            pickle.dump(
                self.q_table,
                file
            )

        print("Model saved:", filepath)

    # ======================================
    # LOAD Q-TABLE
    # ======================================

    def load_model(self, filename):

        filepath = os.path.join(
            "results/models",
            filename
        )

        with open(filepath, "rb") as file:

            self.q_table = pickle.load(file)

        print("Model loaded:", filepath)
