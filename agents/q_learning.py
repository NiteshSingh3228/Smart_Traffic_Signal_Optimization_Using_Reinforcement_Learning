
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
        """
        Compress the 17-D continuous traffic state into a compact, highly learnable discrete tuple.
        Solves the curse of dimensionality by shrinking 8.5 billion states into 216 states:
          1. Queue imbalance: NS heavier (0), EW heavier (1), Balanced (2)
          2. Total queue pressure: Empty (0), Light (1), Heavy (2)
          3. Approaching traffic imbalance: NS more (0), EW more (1), Balanced (2)
          4. Emergency vehicle presence: None (0), NS (1), EW (2), Both (3)
          5. Active signal phase: NS Green (0), EW Green (1)
        Total state space: 3 * 3 * 3 * 4 * 2 = 216 states.
        """

        if len(state) != STATE_SIZE:
            raise ValueError(
                f"Expected {STATE_SIZE} state values, got {len(state)}"
            )

        # 1. Queue imbalance between directions
        ns_queue = state[1] + state[3]
        ew_queue = state[5] + state[7]
        diff_queue = ns_queue - ew_queue

        if diff_queue > 1:
            q_balance = 0
        elif diff_queue < -1:
            q_balance = 1
        else:
            q_balance = 2

        # 2. Total queue congestion level
        total_q = ns_queue + ew_queue

        if total_q == 0:
            q_pressure = 0
        elif total_q <= 4:
            q_pressure = 1
        else:
            q_pressure = 2

        # 3. Approaching traffic flow imbalance
        ns_cars = state[0] + state[2]
        ew_cars = state[4] + state[6]
        diff_cars = ns_cars - ew_cars

        if diff_cars > 1:
            car_balance = 0
        elif diff_cars < -1:
            car_balance = 1
        else:
            car_balance = 2

        # 4. Emergency vehicle demand
        ns_amb = (state[8] + state[9] > 0) or (state[12] + state[13] > 0)
        ew_amb = (state[10] + state[11] > 0) or (state[14] + state[15] > 0)

        if ns_amb and not ew_amb:
            amb_state = 1
        elif ew_amb and not ns_amb:
            amb_state = 2
        elif ns_amb and ew_amb:
            amb_state = 3
        else:
            amb_state = 0

        # 5. Current traffic signal phase (0: NS Green, 1: EW Green)
        phase_state = 0 if state[16] == 0 else 1

        return (q_balance, q_pressure, car_balance, amb_state, phase_state)

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
