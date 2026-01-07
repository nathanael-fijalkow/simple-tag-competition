"""
Predator Agent for Simple Tag Competition
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=16, output_size=5, hidden_size=64):
        super().__init__()
        # Match the architecture from your model file
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)


class StudentAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        
        current_folder = Path(__file__).parent.absolute()
        model_file = current_folder / "predator_model.pth"
        
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file missing: {model_file}")
        
        self.network = NeuralNetwork().to(self.device)
        
        model_weights = torch.load(model_file, map_location=self.device)
        self.network.load_state_dict(model_weights)
        
        self.network.eval()
    
    def get_action(self, observation, agent_id: str):
        obs_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.network(obs_tensor)
            chosen_action = torch.argmax(outputs, dim=1)
        
        return int(chosen_action.item())