import os
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

# Dossier courant = submissions/CamilleRaynaud/
THIS_DIR = Path(__file__).resolve().parent

# Racine du projet = simple-tag-competition/
PROJECT_ROOT = THIS_DIR.parents[1]

# Dossier du prey
PREY_DIR = PROJECT_ROOT / "reference_agents_source"

assert PREY_DIR.exists(), f"Prey dir not found: {PREY_DIR}"

sys.path.insert(0, str(PREY_DIR))

from pettingzoo.mpe import simple_tag_v3

# ---- IMPORT du prey d'évaluation ----
# Mets ici le bon chemin/nome de fichier (ex: prey_agent.py)
from prey_agent import StudentAgent as ReferencePreyAgent


# ======================
# Utils
# ======================
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_t(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def to_t_long(x, device):
    return torch.tensor(x, dtype=torch.long, device=device)

def split_predators(all_agents):
    preds = [a for a in all_agents if "adversary" in a]
    prey = [a for a in all_agents if "agent_0" in a or "good_0" in a]
    return preds, prey


# ======================
# Réseaux MAPPO light
# ======================
class ActorNet(nn.Module):
    """Actor décentralisé: obs locale -> logits actions"""
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


class CriticNet(nn.Module):
    """Critic centralisé: obs globale -> V"""
    def __init__(self, global_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, gobs):
        return self.net(gobs).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    epochs: int = 6
    minibatch_size: int = 512
    rollout_steps: int = 2048  # transitions "predator" (pas env-steps)
    seed: int = 0
    device: str = "cpu"


class Rollout:
    """Buffer on-policy"""
    def __init__(self):
        self.obs = []
        self.gobs = []
        self.actions = []
        self.logp = []
        self.rew = []
        self.done = []
        self.val = []
        self.next_gobs = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rew)


# ======================
# MAPPO light agent
# ======================
class MAPPOLight:
    def __init__(self, obs_dim, global_dim, action_dim=5, cfg: PPOConfig = PPOConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = ActorNet(obs_dim, action_dim).to(self.device)
        self.critic = CriticNet(global_dim).to(self.device)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs_np):
        obs = to_t(obs_np, self.device).unsqueeze(0)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), logp.squeeze(0)

    @torch.no_grad()
    def value(self, gobs_np):
        gobs = to_t(gobs_np, self.device).unsqueeze(0)
        v = self.critic(gobs)
        return v.squeeze(0)

    def update(self, roll: Rollout):
        cfg = self.cfg
        device = self.device

        obs = to_t(np.array(roll.obs), device)
        gobs = to_t(np.array(roll.gobs), device)
        actions = to_t_long(np.array(roll.actions), device)
        old_logp = to_t(np.array([lp.item() if torch.is_tensor(lp) else lp for lp in roll.logp]), device)
        rewards = to_t(np.array(roll.rew), device)
        dones = to_t(np.array(roll.done, dtype=np.float32), device)
        values = to_t(np.array([v.item() if torch.is_tensor(v) else v for v in roll.val]), device)
        next_gobs = to_t(np.array(roll.next_gobs), device)

        # Bootstrap next values
        with torch.no_grad():
            next_values = self.critic(next_gobs)

        # ---- GAE ----
        T = rewards.shape[0]
        adv = torch.zeros(T, device=device)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_values[t] * nonterminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values

        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO epochs ----
        idx = torch.arange(T, device=device)

        for _ in range(cfg.epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]

                mb_obs = obs[mb]
                mb_gobs = gobs[mb]
                mb_actions = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_ret = returns[mb]
                mb_old_val = values[mb]

                logits = self.actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                val_pred = self.critic(mb_gobs)

                # value clipping (optionnel mais aide)
                val_clipped = mb_old_val + torch.clamp(val_pred - mb_old_val, -0.2, 0.2)
                vf1 = (val_pred - mb_ret).pow(2)
                vf2 = (val_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(vf1, vf2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
                self.opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, actor_path="predator_actor.pth", full_path="predator_full.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "cfg": self.cfg.__dict__},
            full_path
        )


# ======================
# Construction obs globale critic
# ======================
def get_pos_from_obs(obs_vec):
    # MPE: [vel(2), pos(2), ...] => pos = obs[2:4]
    return obs_vec[2:4]

def build_global_obs(obs_dict, predator_id, predators, prey_id):
    """
    global_obs = [pred_local_obs, prey_pos(2), other_pred_pos(2), other_pred_pos(2)]
    """
    pred_obs = obs_dict[predator_id]
    prey_pos = get_pos_from_obs(obs_dict[prey_id])

    other = [p for p in predators if p != predator_id]
    # for safety if ordering changes
    other = sorted(other)

    other_pos = []
    for p in other:
        other_pos.append(get_pos_from_obs(obs_dict[p]))
    other_pos = np.concatenate(other_pos, axis=0)  # (4,)

    gobs = np.concatenate([pred_obs, prey_pos, other_pos], axis=0)
    return gobs


# ======================
# Reward shaping coop
# ======================
def coop_shaping(next_obs, predators, prey_id):
    prey_pos = get_pos_from_obs(next_obs[prey_id])
    pred_pos = {p: get_pos_from_obs(next_obs[p]) for p in predators}

    dists = [np.linalg.norm(pred_pos[p] - prey_pos) for p in predators]

    # bonus: le prey proche de plusieurs preds
    coop_bonus = sum(d < 0.6 for d in dists)  # seuil à ajuster
    # spread: pénalise si preds trop collés
    spread_penalty = 0
    for p1, p2 in combinations(predators, 2):
        if np.linalg.norm(pred_pos[p1] - pred_pos[p2]) < 0.25:
            spread_penalty += 1

    return 0.6 * coop_bonus - 0.25 * spread_penalty


# ======================
# Entraînement principal
# ======================
def train_mappo_light(
    total_updates=2000,
    max_cycles=100,
    num_obstacles=2,
    seed=0,
    device="cpu",
):
    cfg = PPOConfig(seed=seed, device=device)
    set_seed(seed)

    # IMPORTANT: prey renvoie un int action => on entraîne en DISCRETE (continuous_actions=False)
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=num_obstacles,
        max_cycles=max_cycles,
        continuous_actions=False,
    )

    prey_agent = ReferencePreyAgent()

    obs, infos = env.reset(seed=seed)
    predators, _ = split_predators(obs.keys())

    # prey id selon version: "agent_0" ou "good_0"
    prey_id = "agent_0" if "agent_0" in obs else "good_0"

    actor_obs_dim = obs[predators[0]].shape[0]
    # global dim = pred_obs + prey_pos(2) + other_pred_pos(4)
    global_dim = actor_obs_dim + 2 + 4

    agent = MAPPOLight(actor_obs_dim, global_dim, action_dim=5, cfg=cfg)
    roll = Rollout()

    # logging
    running = []

    for upd in range(1, total_updates + 1):
        # collect rollout transitions (predator-transitions)
        roll.clear()
        obs, infos = env.reset(seed=seed + upd)
        done_any = False

        while len(roll) < cfg.rollout_steps:
            if not env.agents:
                obs, infos = env.reset(seed=seed + upd + 1000)

            predators_now, _ = split_predators(env.agents)
            if prey_id not in obs:
                # safety: if reset changed keys
                prey_id = "agent_0" if "agent_0" in obs else "good_0"

            actions = {}

            # Prey action from reference model (deterministic argmax)
            prey_action = prey_agent.get_action(obs[prey_id], prey_id)
            actions[prey_id] = prey_action

            # Predator actions from our actor
            cached = {}
            for p in predators_now:
                a, logp = agent.act(obs[p])
                gobs = build_global_obs(obs, p, predators_now, prey_id)
                v = agent.value(gobs)
                actions[p] = a
                cached[p] = (obs[p], gobs, a, logp, v)

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # Store predator transitions
            for p in predators_now:
                o, g, a, lp, v = cached[p]
                done = bool(terms[p] or truncs[p])

                # base reward env (capture) + shaping coop
                shaped = float(rewards[p]) + float(coop_shaping(next_obs, predators_now, prey_id))

                # next global obs for bootstrap
                next_g = build_global_obs(next_obs, p, predators_now, prey_id)

                roll.obs.append(o)
                roll.gobs.append(g)
                roll.actions.append(a)
                roll.logp.append(lp.detach().cpu())
                roll.val.append(float(v.detach().cpu().item()))
                roll.rew.append(shaped)
                roll.done.append(done)
                roll.next_gobs.append(next_g)

                running.append(shaped)

            obs = next_obs

        stats = agent.update(roll)

        if upd % 20 == 0:
            mean_r = float(np.mean(running[-5000:])) if len(running) > 0 else 0.0
            print(
                f"Update {upd}/{total_updates} | "
                f"mean_shaped(last5k)={mean_r:.3f} | "
                f"pi={stats['policy_loss']:.3f} vf={stats['value_loss']:.3f} ent={stats['entropy']:.3f}"
            )

        if upd % 200 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            agent.save(
                actor_path=f"checkpoints/predator_actor_{upd}.pth",
                full_path=f"checkpoints/predator_full_{upd}.pth",
            )

    # save final
    agent.save(actor_path="predator_actor.pth", full_path="predator_full.pth")
    print("✅ Saved: predator_actor.pth (soumission) and predator_full.pth (debug)")

    env.close()
    return agent


if __name__ == "__main__":
    train_mappo_light(
        total_updates=2000,
        max_cycles=100,
        num_obstacles=2,
        seed=0,
        device="cpu",  # mets "cuda" si dispo
    )
