"""
PPO Agent for Simple Tag Competition
Fixed model loading for PyTorch 2.6+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import os


# ========== Network Definitions ==========
class QNetwork(nn.Module):
    """Neural network for action selection - FIXED ARCHITECTURE."""
    
    def __init__(self, state_size: int = 16, action_size: int = 5):
        """Fixed architecture: 16 → 128 → 128 → 5"""
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ========== Student Agent Interface ==========
class StudentAgent:
    """
    Competition interface agent.
    Loads trained model and provides actions.
    """
    
    def __init__(self):
        # Simple Tag environment parameters
        self.observation_dim = 16  # Fixed for Simple Tag
        self.num_env_actions = 5   # Discrete actions: 0-4
        
        # Create network with FIXED architecture
        self.model = QNetwork(self.observation_dim, self.num_env_actions)
        
        # Load trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model weights safely for PyTorch 2.6+."""
        # Try multiple model paths
        model_paths = [
            "predator_model.pth",
            "submissions/amaillal/predator_model.pth",
            "submissions/amaillal/ppo_predator_best.pth",
            "submissions/amaillal/ppo_predator_final.pth"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # Try loading the checkpoint
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    print(f"Loaded checkpoint from {model_path}")
                    
                    # Load model weights - handle different checkpoint formats
                    if 'actor_state_dict' in checkpoint:
                        # PPO trainer format
                        state_dict = checkpoint['actor_state_dict']
                    elif 'model_state_dict' in checkpoint:
                        # Alternative format
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # Direct state dict
                        state_dict = checkpoint
                    
                    # Load the state dict into our model
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print(f"Successfully loaded model from {model_path}")
                    return
                    
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    continue
        
        print("Warning: No model file found or failed to load. Using random policy.")
        # Initialize with random weights (fallback)
        self.model.eval()
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for predator agent.
        
        Args:
            observation: Agent's observation from environment
            agent_id: Unique identifier for this agent
            
        Returns:
            action: Integer action (0-4) for Simple Tag
        """
        try:
            # Convert observation to tensor
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            
            # Handle NaN/Inf values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Ensure correct shape
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            
            # Check dimension (should be 16 for Simple Tag)
            if observation.shape[1] != self.observation_dim:
                # Adjust dimension if needed
                if observation.shape[1] > self.observation_dim:
                    observation = observation[:, :self.observation_dim]
                else:
                    # Pad with zeros
                    padding = np.zeros((1, self.observation_dim - observation.shape[1]))
                    observation = np.concatenate([observation, padding], axis=1)
            
            # Convert to tensor
            observation_tensor = torch.FloatTensor(observation)
            
            # Get action probabilities
            with torch.no_grad():
                logits = self.model(observation_tensor)
                action_probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            
            # Select action with highest probability
            action = int(np.argmax(action_probs))
            
            # Ensure action is valid
            if action < 0 or action >= self.num_env_actions:
                action = np.random.randint(0, self.num_env_actions)
            
            return action
            
        except Exception as e:
            # Fallback to random action on any error
            print(f"Error selecting action: {e}")
            return np.random.randint(0, self.num_env_actions)