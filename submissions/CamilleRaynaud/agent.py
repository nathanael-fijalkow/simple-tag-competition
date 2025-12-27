import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3


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

    def forward(self, x):
        return self.net(x)


def extract_relative_obs(obs_dict, agent_id, prey_id, predators):
    """
    Recrée exactement la représentation prey-centric utilisée à l'entraînement
    """
    o = obs_dict[agent_id]
    prey = obs_dict[prey_id]

    # positions = obs[2:4], vitesses = obs[0:2]
    pred_pos = o[2:4]
    pred_vel = o[0:2]

    prey_pos = prey[2:4]
    prey_vel = prey[0:2]

    rel_pos = pred_pos - prey_pos
    rel_vel = pred_vel - prey_vel

    # autres predators (positions relatives à la proie)
    others = [p for p in predators if p != agent_id]
    others = sorted(others)

    other_rel = []
    for p in others:
        other_rel.append(obs_dict[p][2:4] - prey_pos)

    other_rel = np.concatenate(other_rel, axis=0)

    return np.concatenate([
        rel_pos,
        rel_vel,
        prey_vel,
        other_rel
    ], axis=0)


class StudentAgent:
    def __init__(self):
        model_path = Path(__file__).parent / "predator_actor_v2.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        # recréer obs_dim dynamiquement
        env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,
            continuous_actions=False,
        )
        obs, _ = env.reset()
        predators = [a for a in obs if "adversary" in a]
        prey_id = "agent_0" if "agent_0" in obs else "good_0"

        sample = extract_relative_obs(obs, predators[0], prey_id, predators)
        obs_dim = sample.shape[0]
        env.close()

        self.actor = ActorNet(obs_dim, 5, 256)
        self.actor.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.actor.eval()

    def get_action(self, observation, agent_id):
        # L'évaluateur ne fournit que l'obs locale → on renvoie argmax sur logits bruts
        x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(x)
            a = torch.argmax(logits, dim=1).item()
        return int(a)
