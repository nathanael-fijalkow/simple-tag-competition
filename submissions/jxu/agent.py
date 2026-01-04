"""
StudentAgent using DQN for predator (no pretrained weights required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random

# --- Neural network for DQN ---
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=5, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Simple DQN agent ---
class StudentAgent:
    def __init__(self):
        self.input_dim = 16  # observation size for predator
        self.output_dim = 5  # discrete actions
        self.model = DQNNetwork(self.input_dim, self.output_dim, hidden_dim=128)
        self.model.eval()  # mode inference

        # Replay buffer & hyperparams for later training
        self.memory = []
        self.max_memory = 5000
        self.batch_size = 32
        self.gamma = 0.99

        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_action(self, observation, agent_id: str):
        """
        Returns action based on epsilon-greedy policy.
        For testing, epsilon can be 1 -> random actions.
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return np.random.randint(0, self.output_dim)
        else:
            with torch.no_grad():
                q_values = self.model(obs_tensor)
                return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  # not enough samples

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    print("Testing DQN StudentAgent...")
    agent = StudentAgent()
    obs = np.random.randn(16)
    action = agent.get_action(obs, "adversary_0")
    print(f"Observation shape: {obs.shape}, Action: {action} (0-4)")
    print("✓ DQN agent is ready for testing and training")
