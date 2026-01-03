"""
Student Agent for Simple Tag Competition - Predator using PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# --- Actor Network  ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        return F.log_softmax(self.network(state), dim=-1)


class StudentAgent:
    """
    PPO-trained predator agent for Simple Tag competition.
    """
    
    def __init__(self):
        """Initialize the predator agent and load trained model."""
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        # Model parameters (must match training)
        self.state_dim = 16  # Predator observation dimension in Simple Tag
        self.action_dim = 5   # Discrete actions: 0-4
        
        # Initialize actor network (128 hidden units)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_size=128)
        
        # Load trained model weights
        model_path = self.submission_dir / "predator_model.pth"
        if model_path.exists():
            self.load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠ Warning: Model not found at {model_path}")
            print("  Using randomly initialized weights")
        
        # Set to evaluation mode
        self.actor.eval()
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from environment (numpy array, shape (16,))
            agent_id: Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right  
                    3 = move down
                    4 = move up
        """
        # Convert observation to tensor
        state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        
        # Get action from policy (greedy selection during evaluation)
        with torch.no_grad():
            log_probs = self.actor(state_tensor)
            action = torch.argmax(log_probs, dim=-1)
        
        return int(action.item())
    
    def load_model(self, model_path):
        """
        Load trained model weights.
        
        Args:
            model_path: Path to the .pth file containing model weights
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint)
            
            self.actor.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized weights")
