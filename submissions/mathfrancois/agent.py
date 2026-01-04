import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# --- ARCHITECTURE DU RÉSEAU (Actor-Critic) ---
class PPOModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOModel, self).__init__()
        # OPTIMISATION 1 : On augmente la capacité du réseau (64 -> 256)
        # Cela permet de capturer des micro-détails de position
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Tête de l'acteur (Politique)
        self.actor = nn.Linear(256, output_dim)
        
        # Tête du critique (Value function)
        self.critic = nn.Linear(256, 1)

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
        # Paramètres de l'environnement Simple Tag
        self.obs_dim = 16 
        self.act_dim = 5 
        
        self.device = torch.device("cpu") 
        # On instancie le modèle avec la nouvelle architecture
        self.policy = PPOModel(self.obs_dim, self.act_dim).to(self.device)
        
        self.load_model()

    def load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "predator_model.pth")
        
        if os.path.exists(model_path):
            # map_location est important pour éviter les erreurs si entraîné sur GPU
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy.eval() 
        else:
            print("Attention: Pas de modèle trouvé, comportement aléatoire.")

    def get_action(self, observation, agent_id: str):
        # Gestion des dictionnaires parfois renvoyés par PettingZoo
        if isinstance(observation, dict):
            observation = observation['observation']
        
        # Transformation en tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inférence
        with torch.no_grad():
            probs = self.policy.get_action_prob(obs_tensor)
            
        # Choix déterministe (Argmax) pour la compétition
        action = torch.argmax(probs).item()
        
        return action