"""
StudentAgent using DQN for predator (no pretrained weights required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random

# --- Neural network for DQN ---
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=5, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Simple DQN agent ---
class StudentAgent:
    def __init__(self):
        self.input_dim = 16  # observation size for predator
        self.output_dim = 5  # discrete actions
        

        #debug
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        # Example: Load predator model
        model_path = self.submission_dir / "predator_model.pth"
        
        # On utilise le DQN Standard
        self.model = DQNNetwork(self.input_dim, self.output_dim, hidden_dim=256)

        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Warning: No model found at {model_path}")
       
        self.memory = []
        self.max_memory = 5000
        self.batch_size = 32
        self.gamma = 0.99

        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_action(self, observation, agent_id: str):
        """
        Returns action based on epsilon-greedy policy.
        For testing, epsilon can be 1 -> random actions.
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            return torch.argmax(q_values, dim=1).item()

    
    
    
        


if __name__ == "__main__":
    print("Testing DQN StudentAgent...")
    agent = StudentAgent()
    obs = np.random.randn(16)
    action = agent.get_action(obs, "adversary_0")
    print(f"Observation shape: {obs.shape}, Action: {action} (0-4)")
    print("✓ DQN agent is ready for testing and training")
   
