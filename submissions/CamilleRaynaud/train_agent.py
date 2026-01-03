import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3

# ---------------------------
# Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBS_DIM = 16
ACT_DIM = 5
HIDDEN_DIM = 256
MAX_EPISODES = 2000
MAX_CYCLES = 50
LR = 3e-4
PPO_EPOCHS = 5
CLIP_EPS = 0.2

# ---------------------------
# Actor & Critic
# ---------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------
# GAE
# ---------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        returns.insert(0, gae + values[t])
    return returns

# ---------------------------
# Entraînement
# ---------------------------
def train():
    env = simple_tag_v3.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=MAX_CYCLES
    )

    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    for ep in range(MAX_EPISODES):
        env.reset()
        buffers = {aid: {"states": [], "actions": [], "logprobs": [], "rewards": [], "values": [], "dones": []}
                   for aid in env.agents if "adversary" in aid}

        for agent in env.agent_iter():
            obs, reward, done, trunc, info = env.last()
            is_predator = "adversary" in agent

            # ----------------- Dead agent -----------------
            if done or trunc:
                env.step(None)
                continue

            # ----------------- Prey -----------------
            if not is_predator:
                env.step(np.random.randint(0, ACT_DIM))
                continue

            # ----------------- Predator -----------------
            buf = buffers[agent]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Actor forward
            logits = actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            value = critic(obs_tensor)

            # Stocker les infos dans le buffer
            buf["states"].append(obs_tensor.cpu())
            buf["actions"].append(action.item())
            buf["logprobs"].append(dist.log_prob(action).item())
            buf["values"].append(value.item())
            buf["rewards"].append(reward)
            buf["dones"].append(0.0)

            env.step(action.item())

        # ----------------- PPO update -----------------
        all_states, all_actions, all_returns, all_values, all_logprobs = [], [], [], [], []
        for buf in buffers.values():
            R = compute_gae(buf["rewards"], buf["values"], buf["dones"])
            all_states.extend([s for s in buf["states"] if s is not None])
            all_actions.extend(buf["actions"])
            all_returns.extend(R)
            all_values.extend(buf["values"])
            all_logprobs.extend(buf["logprobs"])

        if len(all_states) == 0:
            continue

        states = torch.cat(all_states).to(DEVICE)
        actions = torch.tensor(all_actions).to(DEVICE)
        returns = torch.tensor(all_returns).to(DEVICE)
        values = torch.tensor(all_values).to(DEVICE)
        logprobs_old = torch.tensor(all_logprobs).to(DEVICE)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            logits = actor(states)
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)
            ratio = torch.exp(logprobs - logprobs_old)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (returns - critic(states)).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 50 == 0:
            print(f"Episode {ep} — PPO update done")

    # ----------------- Sauvegarde du modèle -----------------
    out = Path("submissions/CamilleRaynaud/predator_model.pth")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out)
    print("✔ Actor model saved to", out)


if __name__ == "__main__":
    train()
