import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=16, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)


class StudentAgent:
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 5
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        
        self._load_model()
    
    def _load_model(self):
        model_paths = [
            "predator_model.pth",
            "submissions/amaillal/predator_model.pth",
            "submissions/amaillal/ppo_predator_best.pth"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    if 'policy_state_dict' in checkpoint:
                        self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    elif 'actor_state_dict' in checkpoint:
                        self.policy.load_state_dict(checkpoint['actor_state_dict'])
                    else:
                        self.policy.load_state_dict(checkpoint)
                    
                    self.policy.eval()
                    #print(f"Loaded model from {model_path}")
                    return
                except Exception as e:
                    pass
                    #print(f"Failed to load {model_path}: {e}")
        
        print("No model found, using random policy")
        self.policy.eval()
    
    def get_action(self, observation, agent_id: str):
        try:
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            
            if observation.shape[1] != self.state_dim:
                if observation.shape[1] > self.state_dim:
                    observation = observation[:, :self.state_dim]
                else:
                    padding = np.zeros((1, self.state_dim - observation.shape[1]))
                    observation = np.concatenate([observation, padding], axis=1)
            
            observation_tensor = torch.FloatTensor(observation)
            
            with torch.no_grad():
                logits = self.policy(observation_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            if action < 0 or action >= self.action_dim:
                action = np.random.randint(0, self.action_dim)
            
            return action
            
        except Exception as e:
            return np.random.randint(0, self.action_dim)