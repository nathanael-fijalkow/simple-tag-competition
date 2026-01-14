import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    Categorical = object


class PolicyNetwork(nn.Module):
    """
    Simple MLP pour politique discrète
    observation -> logits sur les actions.
    """

    def __init__(self, obs_dim: int, n_actions: int = 5, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(self):
        """
        Initialise ton agent prédateur.

        La politique est initialisée paresseusement
        à la première observation car on a besoin
        de connaître la dimension de l'observation.
        Si un fichier 'predator_model.pth' est présent
        dans le même dossier que ce fichier, il est chargé.
        """
        self.n_actions = 5 
        self.policy: Optional[PolicyNetwork] = None

        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "predator_model.pth",
        )

        self.device = torch.device("cpu") if TORCH_AVAILABLE else None

    def _lazy_init_policy(self, obs):
        """
        Initialise le réseau au premier appel avec la bonne taille d'observation.
        Charge les poids si le fichier existe.
        """
        if not TORCH_AVAILABLE:
            return
        obs = np.asarray(obs, dtype=np.float32).flatten()
        obs_dim = obs.shape[0]

        self.policy = PolicyNetwork(obs_dim=obs_dim, n_actions=self.n_actions)
        self.policy.to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def get_action(self, observation, agent_id: str):
        """
        Renvoie l'action pour une observation donnée.

        Args
            observation  observation de l'agent dans l'environnement
            agent_id     identifiant unique de l'agent (p. ex. 'adversary_0')
                         non utilisé ici mais fourni par l'API du concours

        Returns
            action       entier (discrete action) pour simple_tag_v3
        """
        if not TORCH_AVAILABLE:
            return np.random.randint(self.n_actions)

        if self.policy is None:
            self._lazy_init_policy(observation)

        obs = np.asarray(observation, dtype=np.float32).flatten()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # shape [1, obs_dim]

        with torch.no_grad():
            logits = self.policy(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()

        return int(action)
