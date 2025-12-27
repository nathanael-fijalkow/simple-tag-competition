import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# --- Actor réseau (doit matcher EXACTEMENT celui de l'entraînement) ---
class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)  # logits


class StudentAgent:
    def __init__(self):
        # -------- Charger le modèle --------
        model_path = Path(__file__).parent / "predator_actor.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        # -------- Déduire obs_dim dynamiquement --------
        # ⚠️ DOIT matcher EXACTEMENT l'environnement d'évaluation
        from pettingzoo.mpe import simple_tag_v3

        tmp_env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,      # ⚠️ IMPORTANT (obs_dim = 16)
            max_cycles=25,
            continuous_actions=False,
        )

        obs, _ = tmp_env.reset()
        predator_id = [a for a in obs if "adversary" in a][0]
        obs_dim = obs[predator_id].shape[0]
        tmp_env.close()

        # -------- Construire l'actor --------
        self.actor = ActorNet(obs_dim=obs_dim, action_dim=5, hidden=256)
        self.actor.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.actor.eval()

    def get_action(self, observation, agent_id: str):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs)
            a = torch.argmax(logits, dim=1).item()

        # --- Mode DISCRET (évaluation officielle) ---
        # Si l'environnement attend un int, retourner int
        if isinstance(observation, (list, tuple, np.ndarray)):
            # Heuristique sûre : si l'observation est 1D (Simple Tag),
            # l'éval est en mode discret
            return int(a)

        # --- Mode CONTINU (fallback, jamais utilisé en éval ici) ---
        action = np.zeros(5, dtype=np.float32)
        action[a] = 1.0
        return action

