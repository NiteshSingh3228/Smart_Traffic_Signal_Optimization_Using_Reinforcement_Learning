
import os
import pickle
import random


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

            # FIX 2.3: Randomly break ties among maximum Q-values
            # prevents always picking action 0 for unseen states
            max_val = max(q_values)
            best_actions = [
                i for i, val in enumerate(q_values)
                if val == max_val
            ]
            action = random.choice(best_actions)

        return action

    # ======================================
    # LEARN
    # ======================================

    def learn(self, state, action, reward, next_state, done):

        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        current_q = self.get_q_values(state)[action]

        if done:

            # FIX: Time-limit truncation should NOT zero the future value
            # Use max next-Q so target is not biased toward zero
            max_future_q = max(self.get_q_values(next_state))

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
    # SAVE MODEL
    # FIX 2.2: Persist epsilon alongside q_table
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

        checkpoint = {
            "q_table": self.q_table,
            "epsilon": self.epsilon
        }

        with open(filepath, "wb") as file:

            pickle.dump(checkpoint, file)

        print("Model saved:", filepath)

    # ======================================
    # LOAD MODEL
    # FIX 2.2: Restore epsilon from checkpoint
    # ======================================

    def load_model(self, filename):

        filepath = os.path.join(
            "results/models",
            filename
        )

        with open(filepath, "rb") as file:

            data = pickle.load(file)

        # Support both old format (plain dict of states) and new format
        if isinstance(data, dict) and "q_table" in data:
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon)
        else:
            # Legacy: data is the raw q_table dict
            self.q_table = data

        print("Model loaded:", filepath)
