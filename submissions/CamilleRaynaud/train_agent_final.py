import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pathlib import Path
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEVICE)
ACTION_DIM = 5
GAMMA = 0.99
LR = 3e-4

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)

def train():
        env = simple_tag_v3.parallel_env(
            num_good=1, num_adversaries=3, num_obstacles=2,
            max_cycles=50, continuous_actions=False
        )

        obs, _ = env.reset()
        sample_agent = [a for a in obs if "adversary" in a][0]
        obs_dim = len(obs[sample_agent])

        actor = Actor(obs_dim).to(DEVICE)
        optimizer = optim.Adam(actor.parameters(), lr=LR)

        replay_states = []
        replay_actions = []
        replay_returns = []

        for episode in range(6000):
            obs, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []

            done = False
            while env.agents:
                actions = {}

                for agent_id in env.agents:
                    if "adversary" in agent_id:
                        s = torch.tensor(obs[agent_id], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        logits = actor(s)
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample().item()

                        actions[agent_id] = a
                        episode_states.append(obs[agent_id])
                        episode_actions.append(a)

                    else:
                        actions[agent_id] = np.random.randint(ACTION_DIM)

                obs, rewards, terms, truncs, _ = env.step(actions)

                for aid,r in rewards.items():
                    if "adversary" in aid:
                        episode_rewards.append(r)

                if any(terms.values()) or any(truncs.values()):
                    break

            # compute returns
            R = 0
            returns = []
            for r in reversed(episode_rewards):
                R = r + GAMMA * R
                returns.insert(0, R)

            replay_states += episode_states
            replay_actions += episode_actions
            replay_returns += returns

            states = torch.tensor(replay_states, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(replay_actions, dtype=torch.int64, device=DEVICE)
            returns = torch.tensor(replay_returns, dtype=torch.float32, device=DEVICE)

            logits = actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)

            loss = -(log_probs * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if episode % 200 == 0:
                print("Episode", episode, "| loss", loss.item())

        torch.save(actor.state_dict(), "predator_actor_final.pth")

        metadata = {
            "obs_dim": obs_dim,
            "normalize": False
        }
        Path("predator_metadata.json").write_text(json.dumps(metadata))

        print("Training complete")

if __name__ == "__main__":
    train()
