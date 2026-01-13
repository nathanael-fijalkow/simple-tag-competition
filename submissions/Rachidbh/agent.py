"""
PPO-trained predator agent with orthogonal weight initialization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def orthogonal_init(layer, gain=1.0):
    """
    Orthogonal initialization for neural network layers.
    Improves training stability and convergence speed.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class PredatorNetwork(nn.Module):
    """
    Actor-Critic network with orthogonal initialization.
    """
    def __init__(self, observation_size, action_size, hidden_size=256):
        super(PredatorNetwork, self).__init__()
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Apply orthogonal initialization
        self.shared_encoder.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.policy_head, gain=0.01)
        orthogonal_init(self.value_head, gain=1.0)
    
    def forward(self, observation):
        features = self.shared_encoder(observation)
        return self.policy_head(features), self.value_head(features)


_CACHED_MODEL = {"loaded": False, "network": None}


class StudentAgent:
    """
    Student predator agent using orthogonally-initialized PPO policy.
    """
    
    def __init__(self):
        self.model_directory = Path(__file__).parent
        self.model_file = self.model_directory / "predator_weights.pth"
        self.device = torch.device("cpu")
        self.action_count = 5
        
        if not _CACHED_MODEL["loaded"]:
            self._load_trained_model()
            _CACHED_MODEL["network"] = self.network
            _CACHED_MODEL["loaded"] = True
        else:
            self.network = _CACHED_MODEL["network"]
    
    def _load_trained_model(self):
        if not self.model_file.exists():
            raise FileNotFoundError(f"Trained model not found: {self.model_file}")
        
        checkpoint = torch.load(self.model_file, map_location=self.device, weights_only=False)
        
        if 'observation_dim' in checkpoint:
            obs_dim = checkpoint['observation_dim']
            model_weights = {k: v for k, v in checkpoint.items() if k != 'observation_dim'}
        else:
            first_layer = checkpoint['shared_encoder.0.weight']
            obs_dim = first_layer.shape[1]
            model_weights = checkpoint
        
        self.network = PredatorNetwork(obs_dim, self.action_count).to(self.device)
        self.network.load_state_dict(model_weights)
        self.network.eval()
    
    def get_action(self, observation, agent_id: str):
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, _ = self.network(obs_tensor)
            action = torch.argmax(action_logits, dim=-1)
        
        return int(action.item())