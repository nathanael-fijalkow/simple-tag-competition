import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ---------------------------
# Actor Network
# ---------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=5, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Student Predator Agent
# ---------------------------
class StudentAgent:
    def __init__(self):
        self.device = "cpu"
        # Charger le modèle sauvegardé
        path = Path(__file__).parent / "predator_model.pth"
        self.model = None

        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.obs_dim = ckpt.get("obs_dim", 16)
            self.model = Actor(self.obs_dim)
            self.model.load_state_dict(ckpt["actor"])
            self.model.eval()
        else:
            # Valeur par défaut si pas de modèle
            self.obs_dim = 16

    def get_action(self, observation, agent_id: str):
        obs = np.asarray(observation, dtype=np.float32)

        # Pad ou trim pour matcher obs_dim
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[:self.obs_dim]

        if self.model is None:
            return int(np.random.randint(0, 5))  # action aléatoire si pas de modèle

        # Prédire l'action
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()

        return int(action)
