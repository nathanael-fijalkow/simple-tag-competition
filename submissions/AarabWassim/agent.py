
"""
Template for student agent submission.

Students should implement the StudentAgent class for the predator only.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    """
    Template agent class for Simple Tag competition.
    
    Students must implement this class with their own agent logic.
    The agent should handle only the "predator" type. The prey is provided publicly by the course.
    """
    
    def __init__(self):
        """
        Initialize your predator agent.
        """
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        # Example: Load predator model
        model_path = self.submission_dir / "predator_model.pth"
        self.model = self.load_model(model_path)
        self.obs_pad = 16  # Pad observations to 16 dimensions if needed
        self.model = PredatorPolicy(input_dim=self.obs_pad, output_dim=5)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        
        self.model.eval()
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment (numpy array)
                         - Predator (adversary): shape (14,)
            agent_id (str): Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right  
                    3 = move down
                    4 = move up
        """
        # IMPLEMENT YOUR POLICY HERE
        
        # Example random policy (replace with your trained policy):
        # Action space is Discrete(5) by default
        # Note: During evaluation, RNGs are seeded per episode for determinism
        obs = np.array(observation, dtype=np.float32)
        if obs.shape[0] < self.obs_pad:
            obs = np.pad(obs, (0, self.obs_pad - obs.shape[0]))
        
        action, _ = self.model.predict(obs)
        return int(action)
    
    def load_model(self, model_path):
        """
        Helper method to load a PyTorch model.
        
        Args:
            model_path: Path to the .pth file
            
        Returns:
            Loaded model
        """
        # Example implementation:
        # model = YourNeuralNetwork()
        # if model_path.exists():
        #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
        #     model.eval()
        # return model
        pass


# My predator policy network
class PredatorPolicy(nn.Module):
    def __init__(self, input_dim=16, output_dim=5):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(128, output_dim)

    def forward(self, x):
        return self.action_net(self.policy_net(x))
    
    def predict(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32)
        if obs.ndim == 1: obs = obs.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(obs)
            action = torch.argmax(logits, dim=1).item()
        return action, None


if __name__ == "__main__":
    # Example usage
    print("Testing StudentAgent...")
    
    # Test predator agent (adversary has 14-dim observation)
    predator_agent = StudentAgent()
    predator_obs = np.random.randn(14)  # Predator observation size
    predator_action = predator_agent.get_action(predator_obs, "adversary_0")
    print(f"Predator observation shape: {predator_obs.shape}")
    print(f"Predator action: {predator_action} (should be in [0, 4])")
    
    print("✓ Agent template is working!")

