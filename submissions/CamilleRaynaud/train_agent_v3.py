import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pathlib import Path
import random
from collections import deque

# ===== Hyperparameters =====
HIDDEN_DIM = 256
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
ENTROPY_COEF = 0.01
MAX_EPISODES = 10000  # Réaliste pour commencer
MAX_CYCLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
ACTION_DIM = 5
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
SLIDING_WINDOW = 100

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
    states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    old_logits = torch.tensor(np.array(old_logits), dtype=torch.float32, device=DEVICE)

    if states_critic is None:
        states_critic = states
    else:
        states_critic = torch.tensor(np.array(states_critic), dtype=torch.float32, device=DEVICE)

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
        optimizer_actor.step()
        optimizer_critic.step()

# ===== Training loop =====
def train():
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=MAX_CYCLES, continuous_actions=False
    )

    # Detect obs dim dynamically
    obs_dict, infos = env.reset()
    sample_agent = [a for a in obs_dict.keys() if "adversary" in a][0]
    obs_dim = len(obs_dict[sample_agent])
    actor = Actor(obs_dim=obs_dim).to(DEVICE)
    critic = Critic(obs_dim=obs_dim*3).to(DEVICE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)

    sliding_rewards = deque(maxlen=SLIDING_WINDOW)

    for episode in range(1, MAX_EPISODES + 1):
        obs_dict, infos = env.reset()
        memory = {'states_actor': [], 'states_critic': [], 'actions': [], 'rewards': [], 'logits': []}
        episode_reward = 0

        while env.agents:
            actions_dict = {}
            predator_obs_list = []

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

            obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)

            for agent_id, r in rewards.items():
                if "adversary" in agent_id:
                    memory['rewards'].append(r)
                    episode_reward += r

            if all(list(terminations.values())) or all(list(truncations.values())):
                break

        sliding_rewards.append(episode_reward)

        returns = []
        R = 0
        for r in reversed(memory['rewards']):
            R = r + GAMMA * R
            returns.insert(0, R)

        ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                   memory['states_actor'], memory['actions'], returns, memory['logits'],
                   states_critic=memory['states_critic'])

        # ===== Logging =====
        if episode % LOG_INTERVAL == 0:
            avg_reward = np.mean(sliding_rewards)
            print(f"Episode {episode}/{MAX_EPISODES} | Avg Reward (last {SLIDING_WINDOW}): {avg_reward:.2f}")

        # ===== Save model =====
        if episode % SAVE_INTERVAL == 0:
            save_path = Path("predator_model.pth")
            torch.save(actor.state_dict(), save_path)
            print(f"Model saved at episode {episode}")

    # Save final model
    save_path = Path("predator_model.pth")
    torch.save(actor.state_dict(), save_path)
    print("Training completed. Final model saved.")

if __name__ == "__main__":
    train()
