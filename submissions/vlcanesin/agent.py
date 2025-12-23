from collections import deque, defaultdict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd

try:
    from pettingzoo.mpe import simple_tag_v3
except ImportError:
    print("Error: pettingzoo not installed. Run: pip install pettingzoo[mpe]")
    sys.exit(1)


class AgentLoader:
    """Utility class to load agent implementations."""
    
    @staticmethod
    def load_agent_from_file(file_path: Path):
        """
        Dynamically load a StudentAgent from a Python file.
        
        Args:
            file_path: Path to the agent.py file
            
        Returns:
            Instantiated agent
        """
        try:
            spec = importlib.util.spec_from_file_location("student_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'StudentAgent'):
                raise AttributeError("Module must contain a 'StudentAgent' class")
            
            agent = module.StudentAgent()
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to load agent from {file_path}: {e}")


class ActorCritic(nn.Module):
    """
    The Actor decides which action to take given a state.
    The Critic estimates the value (expected return) of being in a given state.
    They both share a feature encoder.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_dim)  # latent -> action logits
        self.critic = nn.Linear(hidden_size, 1)          # latent -> expected returnf

    def forward(self, state):
        features = self.encoder(state)
        return self.actor(features), self.critic(features)


# This class combines the Actor and Critic, handles training, and action selection.
class PPOAgent():
    def __init__(
        self,
        state_dim=16,
        action_dim=5,
        agent_ids=["default"],
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        alpha_reward=0.1,
        epsilon_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_clip=0.2,
        max_grad_norm=0.5,
        batch_size=5*3,  # 3 agents, 5 full episodes
        minibatch_size=64,
        device="cuda"
    ):
        """
        PPO agent with additional improvements.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            agent_ids: List of ids of the agents controlled by PPO
            lr_actor: Learning rate for policy network
            lr_critic: Learning rate for value network
            gamma: Discount factor for future rewards
            alpha_reward: Shaping reward coefficient
            epsilon_clip: Clipping parameter for PPO objective (0.1-0.3)
            K_epochs: Number of optimization epochs per batch
            gae_lambda: Lambda for GAE (0=TD, 1=MC)
            entropy_coef: Coefficient for entropy bonus (exploration)
            value_clip: Clipping parameter for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Number of episodes to collect before updating
            minibatch_size: Size of mini-batches for SGD updates
            device: Device to run the training/inference
        """

        # Initialize actor and critic networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)

        # Define optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor_critic.encoder.parameters()) +
            list(self.actor_critic.actor.parameters()),
            lr=lr_actor
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.actor_critic.encoder.parameters()) +
            list(self.actor_critic.critic.parameters()),
            lr=lr_critic
        )

        self.gamma = gamma
        self.alpha_reward = alpha_reward
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        # Entropy coefficient for exploration bonus
        # Prevents premature convergence to deterministic policies
        self.entropy_coef = entropy_coef
        # Value clipping to prevent large value function updates
        self.value_clip = value_clip
        # Gradient clipping to prevent exploding gradients
        self.max_grad_norm = max_grad_norm
        # Batch size - collect multiple episodes before updating
        self.batch_size = batch_size
        # Mini-batch size for SGD updates during optimization
        self.minibatch_size = minibatch_size

        self.agent_ids = agent_ids

        # Buffers for current episode  (receives the id of the agent as key)
        self.episode_states = defaultdict(list)
        self.episode_actions = defaultdict(list)
        self.episode_log_probs = defaultdict(list)
        self.episode_rewards = defaultdict(list)
        self.episode_values = defaultdict(list)
        self.episode_dones = defaultdict(list)

        # Batch buffers for collecting multiple episodes before update
        # Previous implementation updated after every single episode
        self.batch_states = defaultdict(list)
        self.batch_actions = defaultdict(list)
        self.batch_log_probs = defaultdict(list)
        self.batch_rewards = defaultdict(list)
        self.batch_values = defaultdict(list)
        self.batch_dones = defaultdict(list)
        self.batch_next_values = defaultdict(list)

        self.episodes_collected = 0
        self.training_mode = True
        self.device = device

    def set_training(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        self.actor_critic.train(training)

    def select_action(self, state, agent_id, greedy=False):
        # Convert state (numpy array) to PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Get log probabilities from the actor (current policy)
        with torch.set_grad_enabled(self.training_mode):
            logits = self.actor_critic(state_tensor)[0]
            dist = torch.distributions.Categorical(logits=logits)

            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            # Store experience during training
            if self.training_mode:
                value = self.actor_critic(state_tensor)[1]
                self.episode_states[agent_id].append(state_tensor)
                self.episode_actions[agent_id].append(action)
                self.episode_log_probs[agent_id].append(dist.log_prob(action))
                self.episode_values[agent_id].append(value)

        return int(action.item())

    def phi(self, distance):
        """Reward potential function"""
        return -distance

    def store_reward(self, reward, agent_id, prey_distance=None, next_prey_distance=None):
        """Store reward for current step."""
        if self.training_mode:
            if prey_distance is not None and next_prey_distance is not None: 
                shaping_reward = self.gamma*self.phi(next_prey_distance) - self.phi(prey_distance)
                shaping_reward *= self.alpha_reward
                reward += shaping_reward

            self.episode_rewards[agent_id].append(reward)

    def end_episode(self, agent_id, next_state=None, done=True):
        """
        Called at end of episode to finalize episode data and add to batch.
        When batch is full, triggers learning. This enables collecting multiple
        episodes before updating, improving sample efficiency.
        
        Args:
            next_state: Final state (for bootstrapping value if not done)
            done: Whether episode terminated naturally
        """
        if not self.training_mode or len(self.episode_states[agent_id]) == 0:
            return

        # Calculate next state value for GAE
        if done:
            next_value = torch.tensor([[0.0]])
        elif next_state is not None:
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.actor_critic(next_state_tensor)[1]
        else:
            next_value = torch.tensor([[0.0]])

        # Add episode to batch
        self.batch_states[agent_id].extend(self.episode_states[agent_id])
        self.batch_actions[agent_id].extend(self.episode_actions[agent_id])
        self.batch_log_probs[agent_id].extend(self.episode_log_probs[agent_id])
        self.batch_rewards[agent_id].extend(self.episode_rewards[agent_id])
        self.batch_values[agent_id].extend(self.episode_values[agent_id])
        self.batch_next_values[agent_id].append(next_value)
        
        # Add done flags for each step in episode
        episode_len = len(self.episode_rewards[agent_id])
        self.batch_dones[agent_id].extend([False] * (episode_len - 1) + [done])

        # Clear episode buffers
        self.episode_states[agent_id] = []
        self.episode_actions[agent_id] = []
        self.episode_log_probs[agent_id] = []
        self.episode_rewards[agent_id] = []
        self.episode_values[agent_id] = []
        self.episode_dones[agent_id] = []

        # Episodes from different agents are counted here
        self.episodes_collected += 1

        # Learn only when batch is full (not after every episode)
        # This improves sample efficiency and training stability
        if self.episodes_collected >= self.batch_size:
            self.learn()
            self.episodes_collected = 0

    def calculate_gae(self, agent_id):
        """
        Calculate Generalized Advantage Estimation (GAE).
        More stable than simple TD error by balancing bias and variance.
        """
        advantages = []
        returns = []
        last_advantage = 0

        # Convert lists to tensors
        rewards = torch.tensor(self.batch_rewards[agent_id], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.batch_dones[agent_id], dtype=torch.float32, device=self.device)

        # Concatenate all next values (last value of each episode)
        values = torch.cat(self.batch_values[agent_id]).squeeze().detach().to(self.device)
        next_values = torch.cat(self.batch_next_values[agent_id]).squeeze().detach().to(self.device)
        
        # Build extended values array
        ext_values = torch.zeros(len(rewards) + 1)
        ext_values[:-1] = values
        
        # Set next values for episode boundaries
        episode_end_idx = 0
        for i, is_done in enumerate(dones):
            if is_done or i == len(dones) - 1:
                ext_values[i + 1] = next_values[episode_end_idx]
                episode_end_idx += 1
            else:
                ext_values[i + 1] = values[i + 1] if i + 1 < len(values) else 0

        # Calculate GAE in reverse
        for t in reversed(range(len(rewards))):
            # TD error for current step
            delta = (
                rewards[t]
                + self.gamma * ext_values[t + 1] * (1 - dones[t])
                - values[t]
            )
            # GAE formula: detach to prevent gradients through value
            last_advantage = (
                delta.detach()
                + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values
        return advantages, returns

    def learn_from_agent(self, agent_id):
        """Update policy and value networks using collected batch."""
        if len(self.batch_states[agent_id]) == 0:
            return

        # Calculate advantages and returns using GAE
        advantages, returns = self.calculate_gae(agent_id)

        # Convert stored lists to tensors for batch processing
        old_states = torch.cat(self.batch_states[agent_id]).squeeze(1)
        old_actions = torch.cat(self.batch_actions[agent_id])
        old_log_probs = torch.cat(self.batch_log_probs[agent_id])
        old_values = torch.cat(self.batch_values[agent_id]).squeeze()

        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get total batch size
        batch_size = old_states.size(0)

        # PPO Optimization with Mini-Batches
        # Standard PPO shuffles data and processes in mini-batches for each epoch
        for epoch in range(self.K_epochs):
            # Generate random permutation for shuffling
            indices = torch.randperm(batch_size)
            
            # Process data in mini-batches
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                minibatch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch
                mb_states = old_states[minibatch_indices]
                mb_actions = old_actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_old_values = old_values[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                
                # Get new log probabilities and values for mini-batch
                new_log_probs_dist = torch.distributions.Categorical(
                    logits=self.actor_critic(mb_states)[0]
                )
                new_log_probs = new_log_probs_dist.log_prob(mb_actions)
                new_values = self.actor_critic(mb_states)[1].squeeze(-1)
                
                # Entropy bonus encourages exploration and prevents
                # premature convergence to deterministic policies
                entropy = new_log_probs_dist.entropy().mean()

                # --- Critic Loss with Value Clipping ---
                # Value function clipping (similar to policy clipping)
                # Prevents destructively large updates to the value function
                # Takes the maximum of clipped and unclipped loss for conservatism
                value_pred_clipped = mb_old_values.detach() + torch.clamp(
                    new_values - mb_old_values.detach(),
                    -self.value_clip,
                    self.value_clip
                )
                value_loss_unclipped = F.mse_loss(new_values, mb_returns.detach())
                value_loss_clipped = F.mse_loss(
                    value_pred_clipped, mb_returns.detach()
                )
                critic_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                # --- Actor Loss (Clipped Surrogate Objective) ---
                # Ratio of new policy probability to old policy probability
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * mb_advantages
                )
                # Add entropy bonus to actor loss
                # Encourages exploration by rewarding policy diversity
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                # --- Update Networks with Gradient Clipping ---
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()

                critic_loss.backward(retain_graph=True)  # keep graph for shared encoder
                actor_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.critic_optimizer.step()
                self.actor_optimizer.step()

        # Clear batch buffers
        self.batch_states[agent_id] = []
        self.batch_actions[agent_id] = []
        self.batch_log_probs[agent_id] = []
        self.batch_rewards[agent_id] = []
        self.batch_values[agent_id] = []
        self.batch_dones[agent_id] = []
        self.batch_next_values[agent_id] = []

    def learn(self):
        for agent in self.agent_ids:
            self.learn_from_agent(agent)

    def save(self, path):
        torch.save(
            self.actor_critic.state_dict(),
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])


def train_parallel(num_episodes=2000, max_steps_per_episode=100, target_score=100*10*0.8, 
                   reward_mode="all", save_path=None):
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=max_steps_per_episode,  
        continuous_actions=False
    )

    predator_ids = [a for a in env.possible_agents if "adversary" in a]
    prey_id = [a for a in env.possible_agents if "adversary" not in a][0]
    num_agents = len(env.possible_agents)
    num_landmarks = len(env.unwrapped.world.landmarks)

    state_dim = 16
    action_dim = 5

    # One PPO agent per predator
    predator = PPOAgent(state_dim=state_dim, action_dim=action_dim, agent_ids=predator_ids)
    prey = AgentLoader.load_agent_from_file("../../reference_agents_source/prey_agent.py")

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting multi-agent PPO training (parallel_env)...")

    for episode in range(1, num_episodes + 1):
        # Seed all episodes
        episode_seed = episode
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)

        obs_dict, infos = env.reset(seed=episode)
        episode_rewards = {pid: 0.0 for pid in predator_ids}
        done_flags = {pid: False for pid in predator_ids}

        for step in range(max_steps_per_episode):
            # Select actions for all predators
            actions = {}
            for pid in predator_ids:
                if not done_flags[pid]:
                    actions[pid] = predator.select_action(obs_dict[pid], pid)
            
            # Select action for prey
            actions[prey_id] = prey.get_action(obs_dict[prey_id], prey_id)

            # Step the environment
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            # Get closest predator to prey
            start = 2 + 2 + 2*num_landmarks + 2*(num_agents-2)  # prey is last agent in ordering
            closest_pid = None
            closest_distance = float('inf')
            closest_next_distance = float('inf')
            for pid in predator_ids:
                prey_rel_pos = obs_dict[pid][start:start+2]
                distance = np.linalg.norm(prey_rel_pos)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pid = pid

                    prey_next_rel_pos = next_obs_dict[pid][start:start+2]
                    next_distance = np.linalg.norm(prey_next_rel_pos)
                    closest_next_distance = next_distance

            # Store rewards and track episode progress
            for pid in predator_ids:
                if not done_flags[pid]: 
                    if reward_mode == "min":
                        predator.store_reward(
                            rewards[pid], pid, 
                            closest_distance if pid == closest_pid else None, 
                            closest_next_distance if pid == closest_pid else None
                        )
                    elif reward_mode == "all":
                        prey_rel_pos = obs_dict[pid][start:start+2]
                        distance = np.linalg.norm(prey_rel_pos)
                        prey_next_rel_pos = next_obs_dict[pid][start:start+2]
                        next_distance = np.linalg.norm(prey_next_rel_pos)
                        predator.store_reward(
                            rewards[pid], pid, 
                            distance, next_distance
                        )

                    episode_rewards[pid] += rewards[pid]
                    done_flags[pid] = terminations[pid] or truncations[pid]

            obs_dict = next_obs_dict

            if all(done_flags.values()):
                break

        # End episode for each agent
        for pid in predator_ids:
            last_obs = obs_dict[pid] if not done_flags[pid] else None
            predator.end_episode(pid, next_state=last_obs, done=done_flags[pid])

        avg_episode_reward = np.mean(list(episode_rewards.values()))
        scores_deque.append(avg_episode_reward)
        scores.append(avg_episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(f"\nSolved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    print("\nTraining complete.")

    if save_path is not None:
        save_file = "predator_model.pth"
        predator.save(save_path / save_file)
        print(f"Saved shared model for all predators at {save_path / save_file}")

    return predator, scores


class StudentAgent:
    def __init__(self, agent_type: str = "predator"):
        """
        Initialize the predator agent.
        Loads the trained PPO Actor-Critic model.
        """
        self.submission_dir = Path(__file__).parent

        # Device: force CPU for evaluation (safe)
        self.device = torch.device("cpu")

        # Environment specs (simple_tag predator)
        self.state_dim = 16
        self.action_dim = 5

        # Initialize model
        self.model = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)

        # Load trained weights
        model_path = self.submission_dir / "predator_model.pth"
        self.load_model(model_path)

        # Evaluation mode
        self.model.eval()

    def get_action(self, observation, agent_id: str):
        """
        Select an action given an observation.

        Args:
            observation (np.ndarray): shape (16,)
            agent_id (str): unique agent identifier (not used here)

        Returns:
            int: discrete action in [0, 4]
        """

        # Convert observation to tensor
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            action = torch.argmax(logits, dim=-1)  # greedy!

        return int(action.item())

    def load_model(self, model_path: Path):
        """
        Load model weights from disk.
        """

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)


def plot_comparison(dirs, ma_window=50):
    plt.figure(figsize=(10, 5))

    for d in dirs:
        scores_df = pd.read_csv(join(d, "scores.csv"))
        scores_df = scores_df.loc[:, ~scores_df.columns.str.contains("^Unnamed")]

        # Assume single column of scores
        scores = scores_df.iloc[:, 0]

        # Moving average
        moving_avg = scores.rolling(window=ma_window).mean()

        # Plot
        plt.plot(scores, alpha=0.3, label=f"{d} (raw)")
        plt.plot(moving_avg, linewidth=2, label=f"{d} (MA {ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(train=True):
    if train:        
        _, scores = train_parallel(save_path=Path(__file__).parent)

        plt.plot(scores)
        plt.tight_layout()
        plt.savefig("scores.png")
        pd.DataFrame(data=scores).to_csv("scores.csv")

    dirs = [d for d in listdir(".") if not isfile(join(".", d))]
    dirs.remove("__pycache__")
    plot_comparison(dirs)
    

if __name__ == "__main__":
    main(True)