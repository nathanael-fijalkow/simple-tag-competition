"""
PPO Agent with Orthogonal Initialization and Reward Shaping
Inspired by CleanRL and best practices
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """orthogonal initialization for better training."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    policy network with orthogonal initialization.
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """
    value network with orthogonal initialization.
    """
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
    
    def forward(self, state):
        return self.network(state)


class StudentAgent:
    """
    trained predator agent for evaluation.
    """
    # class-level shared models (loaded once)
    _shared_actor = None
    _device = torch.device("cpu")
    _model_loaded = False
    
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 5
        
        # load shared actor only once
        if not StudentAgent._model_loaded:
            self._load_model()
            StudentAgent._model_loaded = True
    
    def _load_model(self):
        """load trained model weights."""
        model_path = os.path.join(
            os.path.dirname(__file__), "predator_model.pth"
        )
        
        StudentAgent._shared_actor = Actor(self.state_dim, self.action_dim).to(self._device)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self._device)
                StudentAgent._shared_actor.load_state_dict(checkpoint["actor_state_dict"])
                StudentAgent._shared_actor.eval()
                print(f"[INFO] Loaded trained predator model from {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
        else:
            print(f"[WARNING] No trained model found at {model_path}")
    
    def get_action(self, observation, agent_id: str):
        """
        select action (greedy for evaluation).
        """
        if isinstance(observation, dict):
            observation = observation["observation"]
        
        state = torch.FloatTensor(observation).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            logits = StudentAgent._shared_actor(state)
            action = torch.argmax(logits, dim=1).item()
        
        return action