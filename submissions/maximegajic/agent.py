import torch
import torch.nn as nn
from pathlib import Path

# --- ARCHITECTURE DQN STANDARD (Doit être identique à train.py) ---
class DQN(nn.Module):
    def __init__(self, input_dim=16, output_dim=5, hidden_dim=256):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class StudentAgent:
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        model_path = self.submission_dir / "predator_model.pth"
        
        # On utilise le DQN Standard
        self.model = DQN(hidden_dim=256)
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Warning: No model found at {model_path}")

    def get_action(self, observation, agent_id: str):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = q_values.argmax().item()
        return action