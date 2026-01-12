"""
PPO-trained Predator Agent for Simple Tag Competition.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ActorCritic(nn.Module):
    """Actor-Critic network with shared backbone."""
    
    def __init__(self, obs_dim=16, action_dim=5, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class StudentAgent:
    """PPO-trained predator agent for Simple Tag competition."""
    
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        model_path = self.submission_dir / "predator_model.pth"
        
        self.model = ActorCritic(obs_dim=16, action_dim=5, hidden_dim=256)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            self.model.eval()
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: numpy array of shape (16,)
            agent_id: identifier like "adversary_0"
            
        Returns:
            action: int in [0, 4]
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        return action

