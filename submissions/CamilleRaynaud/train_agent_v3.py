import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pathlib import Path
import random

# ===== Hyperparameters =====
HIDDEN_DIM = 512
LR_ACTOR = 1e-3
LR_CRITIC = 5e-4
GAMMA = 0.95
EPS_CLIP = 0.2
ENTROPY_COEF = 0.02
MAX_EPISODES = 10000
MAX_CYCLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
ACTION_DIM = 5
GRAD_CLIP = 0.5

# ===== Actor & Critic =====
class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x)

# ===== PPO update =====
def ppo_update(actor, critic, optimizer_actor, optimizer_critic, states, actions, returns, old_logits, states_critic=None):
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    old_logits = torch.tensor(old_logits, dtype=torch.float32, device=DEVICE)

    if states_critic is None:
        states_critic = states
    else:
        states_critic = torch.tensor(states_critic, dtype=torch.float32, device=DEVICE)

    for _ in range(EPOCHS):
        logits = actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        ratio = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
        advantage = returns - critic(states_critic).squeeze()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(critic(states_critic).squeeze(), returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
        optimizer_actor.step()
        optimizer_critic.step()

# ===== Training loop =====
def train():
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=MAX_CYCLES, continuous_actions=False
    )

    obs_dict, _ = env.reset()
    sample_agent = [a for a in obs_dict.keys() if "adversary" in a][0]
    obs_dim = len(obs_dict[sample_agent])
    actor = Actor(obs_dim=obs_dim).to(DEVICE)
    critic = Critic(obs_dim=obs_dim*3).to(DEVICE)  # central critic
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    for episode in range(1, MAX_EPISODES+1):
        obs_dict, _ = env.reset()
        memory = {'states_actor': [], 'states_critic': [], 'actions': [], 'rewards': [], 'logits': []}
        episode_reward = 0

        while env.agents:
            actions_dict = {}
            predator_obs_list = []

            # Collect predator obs
            for agent_id in env.agents:
                o = obs_dict[agent_id]
                if "adversary" in agent_id:
                    predator_obs_list.append(o)

            central_state = np.concatenate(predator_obs_list)

            for agent_id in env.agents:
                o = obs_dict[agent_id]
                if "adversary" in agent_id:
                    obs_tensor = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    logits = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                    memory['states_actor'].append(o)
                    memory['states_critic'].append(central_state)
                    memory['actions'].append(action)
                    memory['logits'].append(logits.detach().cpu().squeeze().numpy())
                    actions_dict[agent_id] = action
                else:
                    actions_dict[agent_id] = random.randint(0, ACTION_DIM-1)

            obs_dict, rewards, terminations, truncations, _ = env.step(actions_dict)

            for agent_id, r in rewards.items():
                if "adversary" in agent_id:
                    memory['rewards'].append(r)
                    episode_reward += r

            if all(list(terminations.values())) or all(list(truncations.values())):
                break

        # Compute returns
        returns = []
        R = 0
        for r in reversed(memory['rewards']):
            R = r + GAMMA * R
            returns.insert(0, R)

        # PPO update
        ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                   memory['states_actor'], memory['actions'], returns, memory['logits'],
                   states_critic=memory['states_critic'])

        # Logging
        if episode % 50 == 0:
            avg_reward = episode_reward / MAX_CYCLES
            print(f"Episode {episode}/{MAX_EPISODES} | Avg Predator Reward: {avg_reward:.2f}")

        # Save checkpoints
        if episode % 5000 == 0:
            torch.save(actor.state_dict(), f"predator_model_ep{episode}.pth")
            print(f"Saved model at episode {episode}")

    torch.save(actor.state_dict(), Path("predator_model_final.pth"))
    print("Training complete. Final model saved at predator_model_final.pth")

if __name__ == "__main__":
    train()
