import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# -------- Actor network --------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        # IMPORTANT: internal name = "net"
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(self):
        self.device = "cpu"

        path = Path(__file__).parent / "predator_model.pth"
        ckpt = torch.load(path, map_location="cpu")

        # -------- detect checkpoint type --------
        if "actor" in ckpt:
            self.obs_dim = ckpt.get("obs_dim", 16)
            state_dict = ckpt["actor"]

        else:
            state_dict = ckpt
            # try to infer obs_dim
            first_tensor = next(iter(state_dict.values()))
            self.obs_dim = first_tensor.shape[1]

        # -------- build network --------
        self.actor = Actor(self.obs_dim)

        # -------- handle 'model.' vs 'net.' mismatch --------
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                remapped["net." + k[len("model."):]] = v
            else:
                remapped[k] = v

        self.actor.load_state_dict(remapped, strict=False)
        self.actor.eval()



    # ======================================================
    # Method expected by the evaluation script
    # ======================================================
    @torch.no_grad()
    def get_action(self, observation, agent_id: str):
        obs = np.asarray(observation, dtype=np.float32)

        # pad / trim defensively
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[:self.obs_dim]

        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(x)

        return int(torch.argmax(logits, dim=1).item())
