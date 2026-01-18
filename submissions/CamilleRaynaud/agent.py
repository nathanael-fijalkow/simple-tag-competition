import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# Réseau de l'acteur
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        # Réseau simple avec deux couches cachées de 256 unités et ReLU
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(self):
        self.device = "cpu"

        # Chargement du modèle entraîné
        path = Path(__file__).parent / "predator_model.pth"
        ckpt = torch.load(path, map_location="cpu")

        # Détection du type de checkpoint 
        if "actor" in ckpt:
            # Si le checkpoint contient un dictionnaire avec 'actor'
            self.obs_dim = ckpt.get("obs_dim", 16)
            state_dict = ckpt["actor"]
        else:
            # Sinon on essaye de deviner la dimension d'observation
            state_dict = ckpt
            first_tensor = next(iter(state_dict.values()))
            self.obs_dim = first_tensor.shape[1]


        self.actor = Actor(self.obs_dim)

        # Gestion des différences de noms (éviter mes soucis entre anciens et nouveaux model)
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                remapped["net." + k[len("model."):]] = v
            else:
                remapped[k] = v

        self.actor.load_state_dict(remapped, strict=False)
        self.actor.eval()  # on ne fait pas de backprop ici



    
    # Méthode appelée par le script d'évaluation
    @torch.no_grad()
    def get_action(self, observation, agent_id: str):
        obs = np.asarray(observation, dtype=np.float32)

        # Ajustement de la taille de l'observation si nécessaire
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[:self.obs_dim]

        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(x)

        # On renvoie l'action avec la probabilité la plus élevée
        return int(torch.argmax(logits, dim=1).item())




#####  Commentaire 

# Pour ce projet, j’ai commencé par implémenter une version très simple de PPO pour les agents prédateurs
# dans l’environnement Simple Tag (j'ai égalment tester DQN mais ce n'était pas trés concluant, j'ai vite abandoné pour rester avec PPO). Mon objectif principal était de comprendre le pipeline complet 
# de l’apprentissage par renforcement multi-agents, plutôt que de viser directement des performances optimales.

# Au départ, j’ai juste mis en place le PPO "de base", sans heuristiques ni améliorations. Rapidement,
# j’ai rencontré plusieurs difficultés qui m’ont ralenti :
# - la longueur des entraînements : chaque session prend beaucoup de temps, donc tester des idées était long,
# - la différence de scores entre mon entraînement local et le pull request sur le serveur : sur ma machine,
#   le score maximum que j’arrivais à atteindre était ~100, mais après avoir fait un pull et lancé l’évaluation
#   officielle, mon score passait à plus de 1300. Ça m’a beaucoup perturbé et j’ai passé pas mal de temps à comprendre
#   comment les autres faisaient pour atteindre des scores très élevés (j'aurais du faire un pull request bien plus tot pour m'éviter ce souci)

# Petit à petit, j’ai essayé d’améliorer ma stratégie et stabiliser l’apprentissage. J’ai mis en place plusieurs
# éléments pour aider :
# - normalisation des observations pour que les entrées soient mieux centrées et échelonnées,
# - critic centralisé basé sur l’état global des trois prédateurs,
# - partage des paramètres de politique entre les agents pour réduire la variance et simplifier l’apprentissage,
# - légère décroissance de l’entropie pendant l’entraînement pour favoriser l’exploitation,
# - utilisation de plusieurs environnements en parallèle pour accélérer le processus,
# - GAE pour calculer les avantages de manière plus stable.

# Je n’ai pas fait de recherche approfondie sur les hyper-paramètres par manque de temps à cause de la durée
# des entraînements. J’ai dû limiter les modifications et me concentrer sur des stratégies pratiques pour améliorer
# les performances dans le temps disponible.

# j’ai utilisé un assistant IA pour m’aider à identifier des bugs de code et améliorer certaines parties
# de la boucle d’entraînement. 

# Mon agent final reste volontairement simple : lors de l’évaluation, le réseau d’acteur PPO est chargé, un petit
# pré-traitement défensif est appliqué sur l’observation si nécessaire, puis l’action est sélectionnée à partir des logits
# du réseau. Ce n’est pas le modèle le plus optimisé du leaderboard, mais ce projet m’a permis de comprendre en profondeur
# le RL multi-agents, la gestion des entraînements longs, et la logique de PPO étape par étape.
