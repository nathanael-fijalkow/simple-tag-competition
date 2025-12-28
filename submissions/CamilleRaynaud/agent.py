import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 5

# ===== Actor Network =====
class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512, action_dim=ACTION_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

# ===== Submission Agent =====
class StudentAgent:
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        model_path = self.submission_dir / "predator_model_final.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.obs_dim = 16  # placeholder
        self.actor = Actor(obs_dim=self.obs_dim).to(DEVICE)
        self.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.actor.eval()

    def get_action(self, observation, agent_id: str):
        obs = np.array(observation, dtype=np.float32)
        if obs.shape[0] != self.obs_dim:
            self.obs_dim = obs.shape[0]
            self.actor = Actor(obs_dim=self.obs_dim).to(DEVICE)
            model_path = self.submission_dir / "predator_model_final.pth"
            self.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.actor.eval()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
        return int(action)
