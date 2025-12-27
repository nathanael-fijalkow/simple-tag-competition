# ===============================
# train_models_final.py
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3

# ---------- Réseau Actor-Critic ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)


# ---------- Agent PPO ----------
class PPOAgent:
    def __init__(self, obs_dim, action_dim=5):
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.entropy_coef = 0.001  # 🔧 réduit
        self.value_coef = 0.5

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.detach(),
            value.squeeze().detach(),
        )


# ---------- Buffer ----------
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()


# ---------- PPO Update ----------
def ppo_update(agent, buffer, epochs=4):
    if len(buffer.rewards) == 0:
        return

    obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32)
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values).squeeze()
    rewards = buffer.rewards
    dones = buffer.dones

    # ----- Returns -----
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0.0
        G = r + agent.gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 🔧

    for _ in range(epochs):
        logits, value_preds = agent.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-agent.clip_eps, 1+agent.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(value_preds.squeeze(), returns)
        entropy = dist.entropy().mean()

        loss = (
            policy_loss
            + agent.value_coef * value_loss
            - agent.entropy_coef * entropy
        )

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()


# ---------- Entraînement ----------
def train():
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=100,
        continuous_actions=True,  # 🔧 CRITIQUE
    )

    obs, _ = env.reset()
    predator_id = [a for a in obs if "adversary" in a][0]
    obs_dim = obs[predator_id].shape[0]
    print("Observation dimension:", obs_dim)

    agent = PPOAgent(obs_dim)
    buffer = RolloutBuffer()
    update_every = 5

    for episode in range(5000):
        obs, _ = env.reset()
        episode_reward = 0.0

        # distances initiales
        prev_dist = {
            a: np.linalg.norm(obs[a][:2] - obs["agent_0"][:2])
            for a in env.agents if "adversary" in a
        }

        while env.agents:
            actions = {}
            cache = {}

            for aid in env.agents:
                if "adversary" in aid:
                    act, logp, val = agent.act(obs[aid])
                    actions[aid] = act
                    cache[aid] = (obs[aid], act, logp, val)
                else:
                    actions[aid] = 0  # 🔧 prey fixe

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for aid in cache:
                curr_dist = np.linalg.norm(
                    next_obs[aid][:2] - next_obs["agent_0"][:2]
                )
                delta_dist = prev_dist[aid] - curr_dist
                shaped_reward = rewards[aid] + 0.1 * delta_dist  # 🔧

                o, a, lp, v = cache[aid]
                buffer.obs.append(o)
                buffer.actions.append(a)
                buffer.log_probs.append(lp)
                buffer.values.append(v)
                buffer.rewards.append(shaped_reward)
                buffer.dones.append(terms[aid] or truncs[aid])

                episode_reward += shaped_reward
                prev_dist[aid] = curr_dist

            obs = next_obs

        if (episode + 1) % update_every == 0:
            ppo_update(agent, buffer)
            buffer.clear()

        if episode % 100 == 0:
            print(f"Episode {episode}, avg reward: {episode_reward:.2f}")

    torch.save(agent.model.state_dict(), "ppo_predator.pth")
    print("✔ Model saved as ppo_predator.pth")


if __name__ == "__main__":
    train()
