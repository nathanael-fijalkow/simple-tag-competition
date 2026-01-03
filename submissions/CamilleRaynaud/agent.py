import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        model_path = self.submission_dir / "predator_model.pth"

        self.model = ExampleNetwork(input_dim=14)

        if model_path.exists():
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            self.model = None  # fallback random policy

    def get_action(self, observation, agent_id: str):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # fallback random if no model
        if self.model is None:
            return int(np.random.randint(0, 5))

        with torch.no_grad():
            logits = self.model(obs)
            action = torch.argmax(logits, dim=-1).item()

        return int(action)


class ExampleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
