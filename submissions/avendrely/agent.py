import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim=14, num_actions=5, architecture=(128, 128)):
        super().__init__()
        self.actor = self.build_mlp(obs_dim, num_actions, architecture)
    def build_mlp(self, input_dim, output_dim, hidden_sizes):
        modules = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            modules.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        modules.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*modules)
    def policy(self, x):
        return self.actor(x)

class StudentAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = 16
        self.num_actions = 5
        self.model = ReinforcePolicy(self.obs_dim, self.num_actions).to(self.device)
        model_path = Path(__file__).parent / "predator_model.pth"
        if model_path.exists():
            state = torch.load(model_path, map_location=self.device)
            # Compatible with both direct and actor-only state_dict
            if any(k.startswith('actor.') for k in state.keys()):
                self.model.actor.load_state_dict({k.replace('actor.', ''): v for k, v in state.items()})
            else:
                self.model.actor.load_state_dict(state)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    def get_action(self, observation, agent_id: str):
        obs = np.asarray(observation, dtype=np.float32).flatten()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.policy(obs_tensor)
            action = int(logits.argmax(dim=-1).item())
        return action
