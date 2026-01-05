#fait par Nicolas Delaere
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    """
    REINFORCE-based agent for Simple Tag competition.

    This agent uses a trained policy network to control the predator.
    The agent handles only the "predator" type. The prey is provided publicly by the course.
    """

    def __init__(self):
        """
        Initialize your predator agent.
        """
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent

        # Load predator model
        self._num_actions = 5
        self._model = None
        self._weights_path = self.submission_dir / "predator_model.pth"
        self._device = torch.device("cpu")
        self._initialized = False

    def get_action(self, observation, agent_id):
        """
        Get action for the given observation.

        Args:
            observation: Agent's observation from the environment (numpy array)
                         - Predator (adversary): shape (14,)
            agent_id (str): Unique identifier for this agent instance

        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right
                    3 = move down
                    4 = move up
        """
        obs_flat = np.asarray(observation, dtype=np.float32).flatten()

        if not self._initialized:
            self.load_model(obs_flat.shape[0])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).to(self._device)
            logits = self._model.policy(obs_tensor)
            return int(logits.argmax(dim=-1).item())

    def load_model(self, obs_dim):
        """
        Helper method to load the trained PyTorch model.

        Args:
            obs_dim: Dimension of the observation space
        """
        self._model = ReinforcePolicy(obs_dim, self._num_actions)
        self._model.to(self._device)

        if self._weights_path.exists():
            state = torch.load(self._weights_path, map_location=self._device)
            actor_state = {k.replace('network.', ''): v for k, v in state.items()}
            self._model.actor.load_state_dict(actor_state)
            print("Loaded model weights")
        else:
            print("No model")

        self._model.eval()
        self._initialized = True


# ============================================================================
# Neural Network Architecture
# ============================================================================

def build_mlp(input_dim, output_dim, hidden_sizes):
    modules = []
    current_dim = input_dim
    for hidden_dim in hidden_sizes:
        modules.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU()
        ])
        current_dim = hidden_dim
    modules.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*modules)


class ReinforcePolicy(nn.Module):
    """
    Neural network for REINFORCE algorithm.

    Contains:
    - Actor network: outputs action logits
    - Baseline network: outputs state value for variance reduction
    """

    def __init__(self, obs_dim, num_actions=5, architecture=(128, 128)):
        super().__init__()
        self.actor = build_mlp(obs_dim, num_actions, architecture)
        self.baseline = build_mlp(obs_dim, 1, architecture)

    def policy(self, x):
        return self.actor(x)

    def baseline_value(self, x):
        return self.baseline(x).squeeze(-1)

    def forward(self, x):
        return self.policy(x), self.baseline_value(x)


# ============================================================================
# Training Code (only used during training, not evaluation)
# ============================================================================

import sys
from torch.distributions import Categorical


class HyperParameters:
    def __init__(self):
        self.max_steps = 500000
        self.rollout_horizon = 512
        self.gamma = 0.99
        self.lr = 1e-3
        self.entropy_coef = 0.01


def obs_to_tensor(obs, device):
    arr = np.asarray(obs, dtype=np.float32).flatten()
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def compute_returns(rewards, dones, gamma):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            G = 0.0
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


def extract_agent_names(agent_list):
    predators = [a for a in agent_list if "adversary" in a]
    preys = [a for a in agent_list if "agent" in a]
    return predators, preys[0] if preys else None


class TrajectoryBuffer:
    def __init__(self, agent_names):
        self.data = {
            agent: {'obs': [], 'act': [], 'logp': [], 'rew': [], 'done': []}
            for agent in agent_names
        }

    def add(self, agent, obs, act, logp):
        self.data[agent]['obs'].append(obs)
        self.data[agent]['act'].append(act)
        self.data[agent]['logp'].append(logp)

    def finalize_step(self, agent, reward, done):
        self.data[agent]['rew'].append(reward)
        self.data[agent]['done'].append(done)

    def size(self, agent):
        return len(self.data[agent]['rew'])

    def clear(self):
        for agent in self.data:
            for key in self.data[agent]:
                self.data[agent][key] = []

    def compile(self, gamma):
        all_obs, all_act, all_logp, all_ret = [], [], [], []

        for agent, traj in self.data.items():
            if not traj['rew']:
                continue

            rews = np.array(traj['rew'], dtype=np.float32)
            dones = np.array(traj['done'], dtype=np.float32)

            # Compute Monte Carlo returns
            ret = compute_returns(rews, dones, gamma)

            all_obs.append(np.stack(traj['obs']))
            all_act.append(np.array(traj['act'], dtype=np.int64))
            all_logp.append(np.array(traj['logp'], dtype=np.float32))
            all_ret.append(ret)

        return {
            'obs': np.concatenate(all_obs),
            'act': np.concatenate(all_act),
            'logp': np.concatenate(all_logp),
            'ret': np.concatenate(all_ret)
        }

    def mean_reward(self):
        all_rewards = []
        for traj in self.data.values():
            all_rewards.extend(traj['rew'])
        return float(np.mean(all_rewards)) if all_rewards else 0.0


class ReinforceTrainer:

    def __init__(self, obs_dim, num_actions, hp, device):
        self.hp = hp
        self.device = device
        self.network = ReinforcePolicy(obs_dim, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hp.lr)

    @torch.no_grad()
    def act(self, obs_numpy):
        obs_tensor = torch.as_tensor(obs_numpy, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.network.policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, batch):
        # Convert to tensors
        obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch['act'], dtype=torch.int64, device=self.device)
        ret = torch.as_tensor(batch['ret'], dtype=torch.float32, device=self.device)

        # Forward pass
        logits, baseline_vals = self.network(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act)
        entropy = dist.entropy().mean()

        advantage = ret - baseline_vals.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        baseline_loss = ((baseline_vals - ret) ** 2).mean()
        policy_loss = -(log_probs * advantage).mean()
        loss = policy_loss + 0.5 * baseline_loss - self.hp.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.network.actor.state_dict(), path)


# ============================================================================
# Main Training Loop
# ============================================================================

def run_training():
    from pettingzoo.mpe import simple_tag_v3
    root_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root_path))
    from reference_agents_source.prey_agent import StudentAgent as PreyAgent
    hp = HyperParameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training REINFORCE on: {device}")
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=25, continuous_actions=False
    )
    env.reset()

    predators, prey = extract_agent_names(env.possible_agents)
    sample_obs_space = env.observation_space(predators[0])
    obs_dim = int(np.prod(sample_obs_space.shape))
    num_actions = env.action_space(predators[0]).n

    trainer = ReinforceTrainer(obs_dim, num_actions, hp, device)
    prey_agent = PreyAgent()
    step_count = 0
    best_reward = float('-inf')
    current_dir = Path(__file__).parent

    print("Starting REINFORCE training\n")

    while step_count < hp.max_steps:
        obs_dict, _ = env.reset()
        buffer = TrajectoryBuffer(predators)
        rollout_steps = 0

        while rollout_steps < hp.rollout_horizon:
            if not env.agents:
                obs_dict, _ = env.reset()

            actions = {}

            for pred in predators:
                if pred not in obs_dict:
                    continue
                obs_flat = np.asarray(obs_dict[pred], dtype=np.float32).flatten()
                act, logp = trainer.act(obs_flat)
                actions[pred] = act
                buffer.add(pred, obs_flat, act, logp)

            if prey in obs_dict:
                prey_obs = np.asarray(obs_dict[prey], dtype=np.float32).flatten()
                actions[prey] = int(prey_agent.get_action(prey_obs, prey))

            # Environment step
            next_obs, rewards, terms, truncs, _ = env.step(actions)

            # Record rewards
            for pred in predators:
                if pred in obs_dict:
                    rew = float(rewards.get(pred, 0.0))
                    done = bool(terms.get(pred, False) or truncs.get(pred, False))
                    buffer.finalize_step(pred, rew, done)

            obs_dict = next_obs
            step_count += 1
            rollout_steps += 1

            if step_count >= hp.max_steps:
                break

        batch = buffer.compile(hp.gamma)
        trainer.update(batch)
        mean_rew = buffer.mean_reward()

        if mean_rew > best_reward:
            best_reward = mean_rew
            trainer.save(current_dir / "predator_model_best.pth")

        print(f"steps={step_count:>7,}  reward={mean_rew:>8.3f}  best={best_reward:>8.3f}")

    trainer.save(current_dir / "predator_model.pth")
    print("\nTraining completed\n")


if __name__ == "__main__":
    run_training()
