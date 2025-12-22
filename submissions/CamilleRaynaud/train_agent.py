"""
train_agent.py

Entraînement PPO et DQN pour Simple Tag (PettingZoo).
Génère ppo_model.pth et dqn_model.pth.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3
from collections import deque
import random

# =========================
# Réseau simple
# =========================
class ExampleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=5, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# =========================
# PPO simplifié
# =========================
class PPOAgent:
    def __init__(self, input_dim=14, output_dim=5, lr=1e-3):
        self.model = ExampleNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.model(obs_tensor)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[0, action])

    def update(self, log_probs, rewards):
        if not log_probs:
            return
        loss = -torch.stack(log_probs) * torch.tensor(rewards, dtype=torch.float32)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# =========================
# DQN simplifié
# =========================
class DQNAgent:
    def __init__(self, input_dim=14, output_dim=5, lr=1e-3):
        self.model = ExampleNetwork(input_dim, output_dim)
        self.target_model = ExampleNetwork(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99

    def get_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0,5)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        obs_batch = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        action_batch = torch.tensor([b[1] for b in batch])
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_obs_batch = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_values = self.model(obs_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_model(next_obs_batch)
            next_q_value = next_q_values.max(1)[0]
            target = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_model.load_state_dict(self.model.state_dict())

# =========================
# Entraînement multi-agent
# =========================
def train_agent(agent_type="PPO", episodes=200, max_steps=25):
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=max_steps,
        continuous_actions=False
    )

    obs, infos = env.reset()
    input_dim = len(obs['adversary_0'])  # 14 pour predator
    if agent_type.upper() == "PPO":
        agent = PPOAgent(input_dim=input_dim)
    else:
        agent = DQNAgent(input_dim=input_dim)

    for ep in range(episodes):
        obs, infos = env.reset()
        done = {k: False for k in env.agents}
        log_probs = []
        rewards = []

        for t in range(max_steps):
            actions = {}
            for agent_id in env.agents:
                if "adversary" in agent_id:
                    if agent_type.upper() == "PPO":
                        a, lp = agent.get_action(obs[agent_id])
                        actions[agent_id] = a
                        log_probs.append(lp)
                        rewards.append(0)  # placeholder simple
                    else:
                        a = agent.get_action(obs[agent_id], epsilon=0.1)
                        actions[agent_id] = a
                        # Pour DQN, stocker transition
                else:
                    actions[agent_id] = np.random.randint(0,5)
            obs, rew, term, trunc, info = env.step(actions)
        if agent_type.upper() == "PPO":
            agent.update(log_probs, rewards)

    model_path = Path(f"{agent_type.lower()}_model.pth")
    torch.save(agent.model.state_dict(), model_path)
    print(f"{agent_type} model saved to {model_path}")

# =========================
# Lancer entraînement
# =========================
if __name__ == "__main__":
    print("Training PPO agent...")
    train_agent("PPO", episodes=1000)

    print("\nTraining DQN agent...")
    train_agent("DQN", episodes=200)
