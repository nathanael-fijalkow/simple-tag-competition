import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# --- ARCHITECTURE DU RÉSEAU (Actor-Critic) ---
class PPOModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.actor = nn.Linear(64, output_dim)
        
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_action_prob(self, x):
        features = self.forward(x)
        action_logits = self.actor(features)
        return F.softmax(action_logits, dim=-1)

    def get_value(self, x):
        features = self.forward(x)
        return self.critic(features)

# --- CLASSE DE SOUMISSION ---
class StudentAgent:
    def __init__(self):
        self.obs_dim = 16 
        self.act_dim = 5 
        
        # Initialisation du modèle
        self.device = torch.device("cpu") 
        self.policy = PPOModel(self.obs_dim, self.act_dim).to(self.device)
        
        # Chargement des poids
        self.load_model()

    def load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "predator_model.pth")
        
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy.eval() # Mode évaluation (fige les couches comme Dropout)
        else:
            print("Attention: Pas de modèle trouvé, comportement aléatoire.")

    def get_action(self, observation, agent_id: str):
        if isinstance(observation, dict):
            observation = observation['observation']
        
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy.get_action_prob(obs_tensor)
            
        #Choix de l'action
        action = torch.argmax(probs).item()
        
        return action