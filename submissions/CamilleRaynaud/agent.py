# ===============================
# agent.py
# ===============================

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=16, action_dim=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)   # 🔑 OBLIGATOIRE

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)



class StudentAgent:
    def __init__(self):
        self.model = ActorCritic(obs_dim=16, action_dim=5)

        model_path = Path(__file__).parent / "ppo_predator.pth"
        if not model_path.exists():
            raise FileNotFoundError("ppo_predator.pth not found")

        print("Loading model from", model_path)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def get_action(self, observation, agent_id: str):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs)
        return torch.argmax(logits, dim=1).item()

