import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, state_dim=16, action_dim=5, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state):
        return F.log_softmax(self.network(state), dim=-1)

_GLOBAL = {"loaded": False, "actor": None}

class StudentAgent:
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        self.device = torch.device("cpu")

        if not _GLOBAL["loaded"]:
            actor = Actor(state_dim=16, action_dim=5, hidden_size=128).to(self.device)
            model_path = self.submission_dir / "shared_predator_model.pth"
            if model_path.exists():
                ckpt = torch.load(model_path, map_location="cpu")
                if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
                    actor.load_state_dict(ckpt["actor_state_dict"])
                else:
                    actor.load_state_dict(ckpt)

            actor.eval()
            _GLOBAL["actor"] = actor
            _GLOBAL["loaded"] = True

        self.actor = _GLOBAL["actor"]

    def get_action(self, observation, agent_id: str):
        obs = torch.tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.actor(obs)
            return int(torch.argmax(log_probs, dim=1).item())
