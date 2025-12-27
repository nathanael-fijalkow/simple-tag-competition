import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
import os
import sys

from pettingzoo.mpe import simple_tag_v3

# --- Resolve project root ---
ROOT = Path(__file__).resolve().parents[2]

# --- Prey agent directory ---
PREY_DIR = ROOT / "reference_agents_source"
assert PREY_DIR.exists(), f"Prey dir not found: {PREY_DIR}"

# --- Add to Python path ---
sys.path.insert(0, str(PREY_DIR))

# --- Import prey agent ---
from prey_agent import StudentAgent as ReferencePreyAgent
print("✔ Prey agent loaded from:", PREY_DIR)

# =========================
# Utils
# =========================
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def split_predators(keys):
    preds = [k for k in keys if "adversary" in k]
    prey = [k for k in keys if "agent_0" in k or "good_0" in k]
    return preds, prey


def pos(o):
    return o[2:4]


def vel(o):
    return o[0:2]


# =========================
# Role assignment
# =========================
def assign_roles(obs, predators, prey_id):
    if prey_id not in obs or len(predators) < 3:
        # fallback si reset bizarre
        return {p: "flank" for p in predators}, np.array([0.0, 0.0])

    prey_v = vel(obs[prey_id])
    prey_p = pos(obs[prey_id])

    if np.linalg.norm(prey_v) < 1e-4:
        # fallback = direction du predator le plus proche
        closest = min(predators, key=lambda p: np.linalg.norm(pos(obs[p]) - prey_p))
        escape = prey_p - pos(obs[closest])
    else:
        escape = prey_v

    escape = escape / (np.linalg.norm(escape) + 1e-6)

    # projection angle pour choisir le bloqueur
    scores = []
    for p in predators:
        d = pos(obs[p]) - prey_p
        if np.linalg.norm(d) < 1e-4:
            s = 1
        else:
            d /= np.linalg.norm(d)
            s = -np.dot(d, escape)
        scores.append((s, p))

    scores.sort(reverse=True)
    blocker = scores[0][1]
    rem = [p for p in predators if p != blocker]
    inter1, inter2 = rem[0], rem[1]

    return {blocker: "blocker", inter1: "flank_l", inter2: "flank_r"}, escape


# =========================
# Target positions
# =========================
def rotate90(v):
    return np.array([-v[1], v[0]])


def role_target(role, prey_pos, escape):
    if role == "blocker":
        return prey_pos + escape * 0.6
    if role == "flank_l":
        return prey_pos + rotate90(escape) * 0.5
    if role == "flank_r":
        return prey_pos - rotate90(escape) * 0.5
    return prey_pos


# =========================
# Observation relative
# =========================
def build_rel_obs(obs, pid, prey_id, predators):
    p = obs[pid]
    prey = obs[prey_id]

    rel_pos = pos(p) - pos(prey)
    rel_vel = vel(p) - vel(prey)

    other = [x for x in predators if x != pid]
    other = sorted(other)
    other_rel = [pos(obs[x]) - pos(prey) for x in other]
    other_rel = np.concatenate(other_rel, axis=0)

    return np.concatenate([rel_pos, rel_vel, vel(prey), other_rel], axis=0)


# =========================
# Networks
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=256, action_dim=5):
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


class Critic(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class CFG:
    gamma: float = 0.99
    gae: float = 0.95
    clip: float = 0.2
    ent: float = 0.005
    vf: float = 0.5
    lr: float = 3e-4
    steps: int = 2048
    epochs: int = 6
    mb: int = 512
    device: str = "cpu"


# =========================
# Reward shaping
# =========================
def coop_reward(obs, next_obs, predators, prey_id, role_map, escape):
    prey_p = pos(next_obs[prey_id])
    pred_pos = {p: pos(next_obs[p]) for p in predators}

    r = 0
    for p in predators:
        t = role_target(role_map[p], prey_p, escape)
        d = np.linalg.norm(pred_pos[p] - t)
        r += -0.4 * d

    vecs = [pred_pos[p] - prey_p for p in predators]
    angles = []
    for v1, v2 in combinations(vecs, 2):
        c = np.dot(v1, v2) / ((np.linalg.norm(v1)*np.linalg.norm(v2))+1e-6)
        angles.append(c)
    r += -0.3 * sum(angles)

    r += 0.3 * np.linalg.norm(prey_p)

    for p1, p2 in combinations(predators, 2):
        if np.linalg.norm(pred_pos[p1]-pred_pos[p2]) < 0.2:
            r -= 0.5

    return float(r)


# =========================
# Train loop
# =========================
def train(total_updates=1200, seed=0, device="cpu"):
    set_seed(seed)
    cfg = CFG(device=device)

    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=100,
        continuous_actions=False,
    )

    prey_agent = ReferencePreyAgent()

    # --- Initial obs reset ---
    obs, infos = env.reset(seed=seed)
    predators, _ = split_predators(obs.keys())
    prey_id = "agent_0" if "agent_0" in obs else "good_0"

    sample = build_rel_obs(obs, predators[0], prey_id, predators)
    obs_dim = sample.shape[0]

    actor = Actor(obs_dim).to(device)
    critic = Critic(obs_dim).to(device)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)

    print("🚀 Launching cooperative PPO predator training...")

    for upd in range(1, total_updates+1):

        buf = dict(obs=[], act=[], logp=[], rew=[], val=[], done=[], next_obs=[])

        obs, infos = env.reset(seed=seed+upd)
        predators, _ = split_predators(obs.keys())
        prey_id = "agent_0" if "agent_0" in obs else "good_0"

        while len(buf["rew"]) < cfg.steps:

            predators, _ = split_predators(env.agents)
            roles, escape = assign_roles(obs, predators, prey_id)

            actions = {}

            # prey action
            if prey_id in obs:
                actions[prey_id] = prey_agent.get_action(obs[prey_id], prey_id)

            cached = {}
            for p in predators:
                x = build_rel_obs(obs, p, prey_id, predators)
                xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

                logits = actor(xt)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
                v = critic(xt)

                actions[p] = int(a.item())
                cached[p] = (x, a.item(), lp.detach(), v.detach(), escape, roles)

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for p in predators:
                if p in cached:
                    x, a, lp, v, escape, roles = cached[p]
                    shaped = float(rewards[p]) + coop_reward(obs, next_obs, predators, prey_id, roles, escape)

                    buf["obs"].append(x)
                    buf["act"].append(a)
                    buf["logp"].append(lp.item())
                    buf["val"].append(v.item())
                    buf["rew"].append(shaped)
                    buf["done"].append(float(terms[p] or truncs[p]))

            obs = next_obs

        # ===== PPO update =====
        O = torch.tensor(np.array(buf["obs"]), dtype=torch.float32, device=device)
        A = torch.tensor(np.array(buf["act"]), dtype=torch.long, device=device)
        OL = torch.tensor(np.array(buf["logp"]), dtype=torch.float32, device=device)
        V = torch.tensor(np.array(buf["val"]), dtype=torch.float32, device=device)
        R = torch.tensor(np.array(buf["rew"]), dtype=torch.float32, device=device)
        D = torch.tensor(np.array(buf["done"]), dtype=torch.float32, device=device)

        with torch.no_grad():
            values_next = critic(O)

        adv = torch.zeros_like(R)
        last = 0
        for t in reversed(range(len(R))):
            nonterm = 1-D[t]
            delta = R[t] + cfg.gamma*values_next[t]*nonterm - V[t]
            last = delta + cfg.gamma*cfg.gae*nonterm*last
            adv[t] = last
        ret = adv + V
        adv = (adv-adv.mean())/(adv.std()+1e-8)

        for _ in range(cfg.epochs):
            idx = torch.randperm(len(R), device=device)
            for i in range(0, len(R), cfg.mb):
                mb = idx[i:i+cfg.mb]

                logits = actor(O[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[mb])

                ratio = torch.exp(lp - OL[mb])
                s1 = ratio*adv[mb]
                s2 = torch.clamp(ratio, 1-cfg.clip, 1+cfg.clip)*adv[mb]
                pg = -torch.min(s1, s2).mean()

                vpred = critic(O[mb])
                vf = 0.5*(vpred - ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.vf*vf - cfg.ent*ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()), 0.5)
                opt.step()

        if upd % 50 == 0:
            print(f"[upd {upd}] shaped_mean={np.mean(buf['rew']):.3f}")

    torch.save(actor.state_dict(), "predator_actor_v2.pth")
    print("✅ Saved predator_actor_v2.pth")


if __name__ == "__main__":
    train(total_updates=1200, seed=0, device="cpu")
