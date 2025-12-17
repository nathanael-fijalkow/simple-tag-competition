#fichier du projet de RL simple tag competition; fait par Nicolas Delaere

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, nbr_actions: int = 5, tailles_cachees=(128, 128)):
        super().__init__()
        couches = []
        der_dim = obs_dim
        for h in tailles_cachees:
            couches.append(nn.Linear(der_dim, h))
            couches.append(nn.ReLU())
            der_dim = h
        couches.append(nn.Linear(der_dim, nbr_actions))
        self.net = nn.Sequential(*couches)

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, tailles_cachees=(128, 128)):
        super().__init__()
        couches = []
        der_dim = obs_dim
        for h in tailles_cachees:
            couches.append(nn.Linear(der_dim, h))
            couches.append(nn.ReLU())
            der_dim = h
        couches.append(nn.Linear(der_dim, 1))
        self.net = nn.Sequential(*couches)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StudentAgent:
    def __init__(self):
        self.nbr_actions = 5
        self.policy: Optional[PolicyNetwork] = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predator_model.pth")
        self.device = torch.device("cpu")

    def _lazy_init_policy(self, obs): # pour la première observation reçue
        obs = np.asarray(obs, dtype=np.float32).flatten()
        obs_dim = obs.shape[0]

        self.policy = PolicyNetwork(obs_dim=obs_dim, nbr_actions=self.nbr_actions)
        self.policy.to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            print("modèle chargé:", self.model_path)
        else:
            print("pas de modèle trouvé, utilisation d'une politique aléatoire\n")

        self.policy.eval()

    def get_action(self, observation, agent_id):
        if self.policy is None:
            self._lazy_init_policy(observation)

        obs = np.asarray(observation, dtype=np.float32).flatten()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()

        return int(action)


# ============================================================================
# PARTIE ENTRAÎNEMENT PPO
# ============================================================================

class PPOConfig:
    total_env_steps: int = 500_000
    rollout_env_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 1024
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOPredatorShared:
    """PPO agent with shared policy for all predators"""
    def __init__(self, obs_dim: int, nbr_actions: int, config: PPOConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.policy = PolicyNetwork(obs_dim, nbr_actions).to(device)
        self.value = ValueNetwork(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=config.learning_rate
        )

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        """Get action, log probability, and value for an observation"""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.value(obs)
        return int(action.item()), float(logp.item()), float(value.item())


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
    last_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation"""
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


def train():
    from pettingzoo.mpe import simple_tag_v3
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from reference_agents_source.prey_agent import StudentAgent as PreyAgent

    cfg = PPOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # l'env selectionné
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False
    )
    env.reset()
    predator_names = [a for a in env.possible_agents if "adversary" in a]
    prey_name = [a for a in env.possible_agents if "agent" in a][0]
    first_pred = predator_names[0]

    pred_obs_dim = int(np.prod(env.observation_space(first_pred).shape))
    nbr_actions = env.action_space(first_pred).n
    print("début de l'entraînement\n")
    predator = PPOPredatorShared(pred_obs_dim, nbr_actions, cfg, device)
    prey_agent = PreyAgent()

    env_steps = 0
    best_mean_reward = float('-inf')

    while env_steps < cfg.total_env_steps:
        obs_dict, _ = env.reset()

        # Storage
        traj = {p: {'obs': [], 'act': [], 'logp': [], 'rew': [], 'done': [], 'val': [], 'nextobs': []}
                for p in predator_names}

        # Collect rollout
        for _ in range(cfg.rollout_env_steps):
            if len(env.agents) == 0:
                obs_dict, _ = env.reset()

            actions = {}

            # Predator actions
            for p in predator_names:
                if p in obs_dict:
                    o = np.asarray(obs_dict[p], dtype=np.float32).flatten()
                    a, logp, v = predator.act(o)
                    actions[p] = a
                    traj[p]['obs'].append(o)
                    traj[p]['act'].append(a)
                    traj[p]['logp'].append(logp)
                    traj[p]['val'].append(v)

            # Prey action
            if prey_name in obs_dict:
                prey_o = np.asarray(obs_dict[prey_name], dtype=np.float32).flatten()
                actions[prey_name] = int(prey_agent.get_action(prey_o, prey_name))

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Store transitions
            for p in predator_names:
                if p not in obs_dict:
                    continue
                r = float(rewards.get(p, 0.0))
                done = bool(terminations.get(p, False) or truncations.get(p, False))
                no = np.asarray(next_obs[p], dtype=np.float32).flatten() if p in next_obs else np.zeros(pred_obs_dim, dtype=np.float32)
                traj[p]['rew'].append(r)
                traj[p]['done'].append(done)
                traj[p]['nextobs'].append(no)

            obs_dict = next_obs
            env_steps += 1

            if env_steps >= cfg.total_env_steps:
                break

        # Compute advantages
        all_obs, all_act, all_oldlogp, all_adv, all_ret = [], [], [], [], []

        for p in predator_names:
            T = len(traj[p]['rew'])
            if T == 0:
                continue

            rewards_np = np.asarray(traj[p]['rew'], dtype=np.float32)
            dones_np = np.asarray(traj[p]['done'], dtype=np.float32)
            values_np = np.asarray(traj[p]['val'], dtype=np.float32)

            # Bootstrap
            if traj[p]['done'][-1]:
                last_value = 0.0
            else:
                with torch.no_grad():
                    last_t = torch.as_tensor(traj[p]['nextobs'][-1], dtype=torch.float32, device=device).unsqueeze(0)
                    last_value = float(predator.value(last_t).item())

            adv_np, ret_np = compute_gae(rewards_np, dones_np, values_np, cfg.gamma, cfg.gae_lambda, last_value)

            all_obs.append(np.stack(traj[p]['obs'], axis=0))
            all_act.append(np.asarray(traj[p]['act'], dtype=np.int64))
            all_oldlogp.append(np.asarray(traj[p]['logp'], dtype=np.float32))
            all_adv.append(adv_np)
            all_ret.append(ret_np)

        # Concatenate
        obs_np = np.concatenate(all_obs, axis=0)
        act_np = np.concatenate(all_act, axis=0)
        oldlogp_np = np.concatenate(all_oldlogp, axis=0)
        adv_np = np.concatenate(all_adv, axis=0)
        ret_np = np.concatenate(all_ret, axis=0)

        # Normalize advantages
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        # To tensors
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_np, dtype=torch.int64, device=device)
        oldlogp_t = torch.as_tensor(oldlogp_np, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=device)

        batch_size = obs_t.shape[0]
        idx = np.arange(batch_size)

        # PPO update
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]

                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_oldlogp = oldlogp_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                # Forward
                logits = predator.policy(mb_obs)
                dist = Categorical(logits=logits)
                newlogp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                newv = predator.value(mb_obs)
                ratio = (newlogp - mb_oldlogp).exp()

                # Losses
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (newv - mb_ret).pow(2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                # Backward
                predator.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(predator.policy.parameters()) + list(predator.value.parameters()),
                    cfg.max_grad_norm
                )
                predator.optimizer.step()

        # Logging
        mean_r = float(np.mean([x for p in predator_names for x in traj[p]['rew']])
                      if any(len(traj[p]['rew']) for p in predator_names) else 0.0)

        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predator_model_best.pth")
            torch.save(predator.policy.state_dict(), best_path)

        print(f"steps={env_steps:>7,}  reward={mean_r:>8.3f}  best={best_mean_reward:>8.3f}  batch={batch_size}")

    # Save final model
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predator_model.pth")
    torch.save(predator.policy.state_dict(), save_path)
    print("\n")
    print("======================")
    print("entrainement terminé")
    print("======================")


if __name__ == "__main__":
    train()
