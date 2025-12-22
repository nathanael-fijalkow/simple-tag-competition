"""
Agent submission for Simple Tag Competition.

Includes both PPO and DQN implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# =========================
# Neural network
# =========================
class ExampleNetwork(nn.Module):
    """
    Simple MLP for discrete actions.
    """
    def __init__(self, input_dim, output_dim=5, hidden_dim=128):
        super(ExampleNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # logits / Q-values
        )
    
    def forward(self, x):
        return self.network(x)

# =========================
# PPO Agent
# =========================
class PPOAgent:
    def __init__(self, input_dim=16, output_dim=5):
        self.model = ExampleNetwork(input_dim, output_dim)
        self.model.eval()  # mode évaluation

    def get_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(obs)
            action = torch.argmax(logits, dim=1).item()
        return action

# =========================
# DQN Agent
# =========================
class DQNAgent:
    def __init__(self, input_dim=16, output_dim=5):
        self.model = ExampleNetwork(input_dim, output_dim)
        self.model.eval()

    def get_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs)
            action = torch.argmax(q_values, dim=1).item()
        return action

# =========================
# Classe exposée pour GitHub
# =========================
class StudentAgent:
    """
    Choisir l'algorithme à utiliser pour l'évaluation :
        algo="PPO" ou algo="DQN"
    """
    def __init__(self, algo="PPO"):
        self.submission_dir = Path(__file__).parent
        self.algo = algo.upper()
        if self.algo == "PPO":
            self.agent = PPOAgent()
            self._load_model("ppo_model.pth")
        elif self.algo == "DQN":
            self.agent = DQNAgent()
            self._load_model("dqn_model.pth")
        else:
            raise ValueError(f"Algo inconnu : {algo}")

    def _load_model(self, filename):
        path = self.submission_dir / filename
        if path.exists():
            self.agent.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.agent.model.eval()
            print(f"Loaded {self.algo} model from {filename}")
        else:
            print(f"No model found at {path}, using random-initialized network")

    def get_action(self, observation, agent_id: str):
        return self.agent.get_action(observation)

# =========================
# Test rapide local
# =========================
if __name__ == "__main__":
    print("Testing StudentAgent...")

    agent = StudentAgent(algo="PPO")
    obs = np.random.randn(14)
    print(f"PPO action: {agent.get_action(obs, 'adversary_0')}")

    agent = StudentAgent(algo="DQN")
    obs = np.random.randn(14)
    print(f"DQN action: {agent.get_action(obs, 'adversary_0')}")

    print("✓ Agent template with PPO and DQN is working!")
