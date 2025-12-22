# ===============================
# train_models_fixed.py
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
        self.entropy_coef = 0.01

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()


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


# ---------- Update PPO ----------
def ppo_update(agent, buffer, epochs=4):
    obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32)
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values)
    rewards = buffer.rewards
    dones = buffer.dones

    # Calcul des returns
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + agent.gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values.detach()

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

        loss = policy_loss + 0.5 * value_loss

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
        continuous_actions=False
    )

    sample_obs = env.reset()[0]
    obs_dim = sample_obs[[a for a in sample_obs if "adversary" in a][0]].shape[0]
    print("Observation dimension:", obs_dim)

    agent = PPOAgent(obs_dim=obs_dim)
    buffer = RolloutBuffer()
    update_every = 5

    for episode in range(5000):
        obs, _ = env.reset()
        episode_rewards = {aid: 0 for aid in env.agents}

        while env.agents:
            # Au lieu de stocker value/log_prob juste après act
            # -> on stocke tout dans le buffer seulement après le reward shaping

            # 1️⃣ Calcul des actions
            actions = {}
            tmp = {}
            for agent_id in env.agents:
                if "adversary" in agent_id:
                    action, logp, value = agent.act(obs[agent_id])
                    actions[agent_id] = action
                    tmp[agent_id] = (obs[agent_id], action, logp.detach(), value.detach())
                else:
                    actions[agent_id] = np.random.randint(5)

            # 2️⃣ Step de l'environnement
            next_obs, rewards, term, trunc, _ = env.step(actions)

            # 3️⃣ Shaped reward et stockage dans buffer
            for agent_id in env.agents:
                if "adversary" in agent_id:
                    dx, dy = obs[agent_id][4], obs[agent_id][5]
                    dist_to_prey = np.sqrt(dx**2 + dy**2)
                    shaped_reward = rewards[agent_id] - 0.01 * dist_to_prey
                    o, a, lp, v = tmp[agent_id]
                    buffer.obs.append(o)
                    buffer.actions.append(a)
                    buffer.log_probs.append(lp)
                    buffer.values.append(v)
                    buffer.rewards.append(shaped_reward)
                    buffer.dones.append(term[agent_id] or trunc[agent_id])
                    episode_rewards[agent_id] += shaped_reward

            obs = next_obs

        if (episode + 1) % update_every == 0:
            ppo_update(agent, buffer)
            buffer.clear()

        if episode % 100 == 0:
            avg_reward = np.mean(list(episode_rewards.values()))
            print(f"Episode {episode}, avg shaped reward: {avg_reward:.2f}")

    torch.save(agent.model.state_dict(), "ppo_predator.pth")
    print("✔ Model saved as ppo_predator.pth")


if __name__ == "__main__":
    train()
