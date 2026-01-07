from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

class PredatorNetwork(nn.Module):
    def __init__(self, obs_dim=16, n_actions=5, hidden_size=64):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)


class StudentAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.submission_dir = Path(__file__).parent.resolve()
        model_path = self.submission_dir / "predator_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.model = PredatorNetwork().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_action(self, observation, agent_id: str):
        obs = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(obs)
            action = torch.argmax(logits, dim=1)

        return int(action.item())
