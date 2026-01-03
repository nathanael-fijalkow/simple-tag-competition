import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from torch.distributions import Categorical
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim=14, act_dim=5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.body(x)
        return self.policy(x), self.value(x)


class AgentBuffer:
    """Separate trajectories per predator"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    values = values + [0]

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * (1-dones[step]) - values[step]
        gae = delta + gamma * lam * (1-dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns


def train():
    env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2)
    env.reset()

    policy = PPOPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    buffers = {aid: AgentBuffer() for aid in env.agents}

    batch_size = 4096
    update_steps = 5

    for episode in range(2000):
        env.reset()

        for agent in env.agent_iter():
            obs, reward, done, _ = env.last()

            if done:
                action = None
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                logits, value = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

                buffers[agent].states.append(obs_t.cpu())
                buffers[agent].actions.append(action.item())
                buffers[agent].logprobs.append(dist.log_prob(action).item())
                buffers[agent].values.append(value.item())

            env.step(action)

            # ---- Heuristic reward shaping ----
            shaped_reward = reward

            if obs is not None and len(obs) >= 4:
                # approx prey direction components are usually early in obs vector
                prey_dx, prey_dy = obs[0], obs[1]
                dist_to_prey = np.sqrt(prey_dx**2 + prey_dy**2)
                shaped_reward += -0.01 * dist_to_prey

            buffers[agent].rewards.append(shaped_reward)
            buffers[agent].dones.append(done)

        # ---- when batch is full: PPO update ----
        total_steps = sum(len(buf.rewards) for buf in buffers.values())
        if total_steps >= batch_size:
            optimize(policy, optimizer, buffers, update_steps)
            for buf in buffers.values():
                buf.clear()

        if episode % 50 == 0:
            print(f"Episode {episode} — collected {total_steps} steps")

    save_model(policy)


def optimize(policy, optimizer, buffers, update_steps):
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2

    states = []
    actions = []
    returns = []
    values = []
    logprobs_old = []

    for buf in buffers.values():
        R = compute_gae(buf.rewards, buf.values, buf.dones, gamma, lam)

        states.extend(buf.states)
        actions.extend(buf.actions)
        returns.extend(R)
        values.extend(buf.values)
        logprobs_old.extend(buf.logprobs)

    states = torch.cat(states).to(DEVICE)
    actions = torch.tensor(actions).to(DEVICE)
    returns = torch.tensor(returns).to(DEVICE)
    values = torch.tensor(values).to(DEVICE)
    logprobs_old = torch.tensor(logprobs_old).to(DEVICE)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(update_steps):
        logits, value = policy(states)
        dist = Categorical(logits=logits)

        logprobs = dist.log_prob(actions)
        ratio = torch.exp(logprobs - logprobs_old)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - value.squeeze()).pow(2).mean()

        loss = policy_loss + 0.5*value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_model(policy):
    out = Path("submissions/your_username/predator_model.pth")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)
    print("✔ Model saved to", out)


if __name__ == "__main__":
    train()
