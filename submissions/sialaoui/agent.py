import os
import re
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


def _as_np(obs):
    if isinstance(obs, dict):
        if "observation" in obs:
            obs = obs["observation"]
        elif "obs" in obs:
            obs = obs["obs"]
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    return x


def _agent_index(agent_id: str, n=3):
    m = re.search(r"(\d+)$", agent_id or "")
    if not m:
        return 0
    return int(m.group(1)) % n


class _MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self._pending_state = None
        self._use_model = False

        self.weights_path = os.path.join(os.path.dirname(__file__), "predator_model.pth")
        if torch is not None and os.path.isfile(self.weights_path):
            try:
                self._pending_state = torch.load(self.weights_path, map_location=self.device)
                self._use_model = True
            except Exception:
                self._pending_state = None
                self._use_model = False

    def _ensure_model(self, obs_dim: int):
        if not self._use_model or torch is None:
            return
        if self.model is not None:
            return

        in_dim = obs_dim + 3  # obs + one-hot predator id (3 predators)
        self.model = _MLPPolicy(in_dim=in_dim, hidden=128, out_dim=5).to(self.device)
        self.model.eval()

        if self._pending_state is not None:
            try:
                if isinstance(self._pending_state, dict) and "state_dict" in self._pending_state:
                    self.model.load_state_dict(self._pending_state["state_dict"], strict=False)
                else:
                    self.model.load_state_dict(self._pending_state, strict=False)
            except Exception:
                self.model = None
                self._use_model = False

    def _heuristic_action(self, obs: np.ndarray):
        # For adversary obs, prey relative position is typically right before the last 2 dims (prey vel),
        # i.e. obs[-4:-2] = (dx, dy) from predator to prey.
        if obs.shape[0] < 4:
            return 0
        dx, dy = float(obs[-4]), float(obs[-3])

        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 1  # right / left
        else:
            return 4 if dy > 0 else 3  # up / down

    def get_action(self, observation, agent_id: str):
        obs = _as_np(observation)

        if self._use_model and torch is not None:
            self._ensure_model(obs_dim=obs.shape[0])
            if self.model is not None:
                idx = _agent_index(agent_id, n=3)
                onehot = np.zeros(3, dtype=np.float32)
                onehot[idx] = 1.0
                x = np.concatenate([obs, onehot], axis=0)

                with torch.no_grad():
                    xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
                    logits = self.model(xt).squeeze(0)
                    action = int(torch.argmax(logits).item())
                return action

        return self._heuristic_action(obs)
