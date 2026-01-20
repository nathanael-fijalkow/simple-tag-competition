import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import hashlib


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim=16, hidden_dim=256, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

def stable_role(agent_id: str) -> int:
    h = hashlib.md5(agent_id.encode("utf-8")).hexdigest()
    return int(h, 16) % 3

def role_one_hot(role: int, k: int = 3) -> np.ndarray:
    v = np.zeros(k, dtype=np.float32)
    v[role % k] = 1.0
    return v

class StudentAgent:
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        self.device = torch.device("cpu")

        
        self.model = SharedPolicy(obs_dim=16, hidden_dim=256, action_dim=5).to(self.device)
        model_path = self.submission_dir / "pretrained_policy.pth"  
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _build_input(self, observation: np.ndarray, agent_id: str) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)

        

        r = role_one_hot(stable_role(agent_id), 3)
        x = np.concatenate([obs, r], axis=0).astype(np.float32)  
        return x

    def get_action(self, observation, agent_id: str):
        x = self._build_input(observation, agent_id)
        xt = torch.from_numpy(x).to(self.device).unsqueeze(0)  
        with torch.no_grad():
            logits = self.model(xt).squeeze(0)  
            action = int(torch.argmax(logits).item())  
        return action
