import torch
import torch.nn as nn
from pathlib import Path

# -------------------------------
# Policy network (inference-only)
# -------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.policy(self.shared(x))


# -------------------------------
# StudentAgent
# -------------------------------
class StudentAgent:
    def __init__(self):
        self.obs_dim = 16   # MUST match training
        self.model = ActorCritic(obs_dim=self.obs_dim)

        model_path = Path(__file__).parent / "predator_model.pth"
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported checkpoint format for predator_model.pth")

        model_state = self.model.state_dict()
        sanitized = {}
        for key, value in checkpoint.items():
            normalized_key = key.removeprefix("module.")
            if (
                normalized_key in model_state
                and getattr(value, "shape", None) == model_state[normalized_key].shape
            ):
                sanitized[normalized_key] = value

        self.model.load_state_dict(sanitized, strict=False)
        self.model.eval()

    def get_action(self, observation, agent_id):
        obs = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(obs)
        return int(torch.argmax(logits))
