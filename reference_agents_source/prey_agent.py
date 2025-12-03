"""
Reference Prey Agent - Pure PyTorch Implementation
Public reference implementation used for evaluation (no SB3 dependency).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class PreyPolicyNetwork(nn.Module):
    """
    MLP policy network for prey agent.
    """
    def __init__(self, obs_dim=14, hidden_dim=256, action_dim=5):
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


class StudentAgent:
    """
    Reference prey agent using pure PyTorch.
    Used as the public opponent for student predator evaluation.
    """
    
    def __init__(self):
        model_path = Path(__file__).parent / "prey_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Reference prey model not found at {model_path}")
        
        try:
            self.model = PreyPolicyNetwork(obs_dim=14, hidden_dim=256, action_dim=5)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load reference prey model: {e}")
    
    def get_action(self, observation, agent_id: str):
        """Get action using the trained policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        return action
