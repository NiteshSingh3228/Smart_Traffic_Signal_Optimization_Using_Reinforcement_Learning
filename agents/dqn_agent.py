
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Use CPU for training
DEVICE = torch.device("cpu")

STATE_SIZE = 17
ACTION_SIZE = 2


# =====================================
# 1. NEURAL NETWORK
# =====================================

class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):

        return self.network(state)


# =====================================
# 2. DQN AGENT
# =====================================

class DQNAgent:

    def __init__(self):

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

        self.learning_rate = 0.001
        self.batch_size = 64

        # FIX 2.5: Store normalized float tensors directly in memory
        # avoids re-normalizing 128 states on every learn() call
        self.memory = deque(maxlen=10000)

        # Main neural network
        self.policy_network = DQNNetwork(
            self.state_size,
            self.action_size
        ).to(DEVICE)

        # Target neural network
        self.target_network = DQNNetwork(
            self.state_size,
            self.action_size
        ).to(DEVICE)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # FIX: Use Huber loss for robustness against large Q-value errors
        self.loss_function = nn.SmoothL1Loss()

        self.training_steps = 0

    # =================================
    # 3. PREPARE STATE
    # =================================

    def normalize_state(self, state):

        if len(state) != self.state_size:
            raise ValueError(
                f"Expected {self.state_size} state values, "
                f"got {len(state)}"
            )

        normalized = []

        # First 8 values: Normal traffic
        for value in state[:8]:

            normalized.append(
                min(max(value / 20.0, 0.0), 1.0)
            )

        # Next 4 values: Ambulance counts
        for value in state[8:12]:

            normalized.append(
                min(max(value / 5.0, 0.0), 1.0)
            )

        # Next 4 values: Ambulance waiting times
        for value in state[12:16]:

            normalized.append(
                min(max(value / 30.0, 0.0), 1.0)
            )

        # FIX 2.4: state[16] can be 0,1,2,3 — divide by 3.0 not 2.0
        # prevents overflow when phase == 3 (yellow phase)
        normalized.append(
            min(max(state[16] / 3.0, 0.0), 1.0)
        )

        return normalized

    def prepare_state(self, state):

        normalized_state = self.normalize_state(state)

        return torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=DEVICE
        )

    # =================================
    # 4. STORE EXPERIENCE
    # FIX 2.5: Pre-normalize here so learn() gets tensors directly
    # =================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        # Store as float tensors to avoid repeated normalization in learn()
        state_tensor = torch.tensor(
            self.normalize_state(state),
            dtype=torch.float32,
            device=DEVICE
        )

        next_state_tensor = torch.tensor(
            self.normalize_state(next_state),
            dtype=torch.float32,
            device=DEVICE
        )

        self.memory.append(
            (state_tensor, action, reward, next_state_tensor, done)
        )

    # =================================
    # 5. CHOOSE ACTION
    # =================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(self.action_size)

        # Exploitation
        state_tensor = self.prepare_state(state)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(state_tensor)

        return q_values.argmax(dim=1).item()

    # =================================
    # 6. TRAIN THE NETWORK
    # =================================

    def learn(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        # FIX 2.5: Tensors are already normalized — stack directly
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE
        )

        next_states = torch.stack(next_states)

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=DEVICE
        )

        # Current Q-values
        current_q_values = self.policy_network(states).gather(
            1, actions
        ).squeeze(1)

        # Future Q-values from target network
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(dim=1)[0]

            # In traffic signal control, episode termination is always a time-limit
            # truncation, not an absorbing death/failure state. Therefore, future
            # value is continuously discounted rather than zeroed out.
            target_q_values = rewards + (
                self.gamma * next_q_values
            )

        # Calculate loss with Huber loss (FIX: replaces MSELoss)
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Update neural network
        self.optimizer.zero_grad()

        loss.backward()

        # FIX: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % 100 == 0:

            self.target_network.load_state_dict(
                self.policy_network.state_dict()
            )

        return loss.item()

    # =================================
    # 7. DECAY EPSILON
    # =================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    # =================================
    # 8. SAVE MODEL
    # =================================

    def save_model(self, filename):

        os.makedirs("results/models", exist_ok=True)

        filepath = os.path.join(
            "results/models",
            filename
        )

        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps
            },
            filepath
        )

        print("DQN model saved:", filepath)

    # =================================
    # 9. LOAD MODEL
    # =================================

    def load_model(self, filename):

        filepath = os.path.join(
            "results/models",
            filename
        )

        checkpoint = torch.load(
            filepath,
            map_location=DEVICE,
            weights_only=True
        )

        if checkpoint["state_size"] != self.state_size:
            raise ValueError(
                "Model state size does not match current agent."
            )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.training_steps = checkpoint["training_steps"]

        print("DQN model loaded:", filepath)
