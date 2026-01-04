"""
Solution for the Simple Tag Competition.

Strategy Overview:
1. Algorithm: Proximal Policy Optimization (PPO) with a Shared Policy architecture.
   - We use a single Actor-Critic network shared across all 3 predator agents.
   - This allows agents to learn from each other's experiences and converge faster.

2. Reward Shaping:
   - To overcome sparse rewards, we added a dense distance-based bonus.
   - Predators receive a small reward for minimizing the distance to the prey.

3. Optimization:
   - Implemented linear decay for Learning Rate and Entropy Coefficient.
   - This helps in fine-tuning the policy in later episodes and avoiding local optima/plateaus.

4. Training:
   - Trained against a reference 'StudentAgent' prey.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from pettingzoo.mpe import simple_tag_v3
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path


# --- Actor Network (Policy) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        return F.log_softmax(self.network(state), dim=-1)


# --- Critic Network (Value Function) ---
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.network(state)


# --- PPO Agent for Simple Tag (Shared Policy Capable) ---
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_clip=0.2,
        max_grad_norm=0.5,
        batch_size=32,
        minibatch_size=256,
    ):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        # Multi-agent temporary buffers (agent_id -> lists)
        self.agent_buffers = {}

        # Main Batch buffers (aggregated for training)
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []
        self.batch_next_values = []

        self.episodes_collected = 0
        self.training_mode = True

    def update_hyperparameters(self, episode, total_episodes):
        """Decay learning rate and entropy coefficient linearly."""
        progress = episode / total_episodes
        
        # Entropy decay: 0.01 -> 0.001
        self.entropy_coef = 0.01 - (0.009 * progress)
        
        # LR decay: start -> min
        min_lr_actor = 3e-5
        min_lr_critic = 1e-4
        
        current_lr_actor = 3e-4 - ((3e-4 - min_lr_actor) * progress)
        current_lr_critic = 1e-3 - ((1e-3 - min_lr_critic) * progress)
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = current_lr_actor
            
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = current_lr_critic

    def set_training(self, training: bool):
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)

    def _ensure_agent_buffer(self, agent_id):
        if agent_id not in self.agent_buffers:
            self.agent_buffers[agent_id] = {
                'states': [], 'actions': [], 'log_probs': [], 
                'rewards': [], 'values': []
            }

    def select_action(self, state, agent_id, greedy=False):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        with torch.set_grad_enabled(self.training_mode):
            log_probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(logits=log_probs)

            if greedy:
                action = torch.argmax(log_probs, dim=-1)
            else:
                action = dist.sample()

            if self.training_mode:
                value = self.critic(state_tensor)
                
                # Store in specific agent's buffer
                self._ensure_agent_buffer(agent_id)
                buf = self.agent_buffers[agent_id]
                buf['states'].append(state_tensor)
                buf['actions'].append(action)
                buf['log_probs'].append(dist.log_prob(action))
                buf['values'].append(value)

        return int(action.item())

    def store_reward(self, reward, agent_id):
        if self.training_mode:
            self._ensure_agent_buffer(agent_id)
            self.agent_buffers[agent_id]['rewards'].append(reward)

    def end_episode(self, agent_id, next_state=None, done=True):
        if not self.training_mode:
            return
            
        if agent_id not in self.agent_buffers:
            return

        buf = self.agent_buffers[agent_id]
        if len(buf['states']) == 0:
            return

        # Calculate next value for GAE
        if done:
            next_value = torch.tensor([[0.0]])
        elif next_state is not None:
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor)
        else:
            next_value = torch.tensor([[0.0]])

        # Move this agent's trajectory to the main batch
        self.batch_states.extend(buf['states'])
        self.batch_actions.extend(buf['actions'])
        self.batch_log_probs.extend(buf['log_probs'])
        self.batch_rewards.extend(buf['rewards'])
        self.batch_values.extend(buf['values'])
        
        # We store next_value only once per trajectory segment for GAE, 
        # but here we are flattening everything. 
        # To keep GAE correct with the existing calculate_gae implementation,
        # we need to handle the 'dones' list carefully.
        # The existing calculate_gae uses batch_dones and batch_next_values 
        # to reconstruct trajectories.
        
        self.batch_next_values.append(next_value)
        
        episode_len = len(buf['rewards'])
        self.batch_dones.extend([False] * (episode_len - 1) + [done])

        # Clear agent buffer
        self.agent_buffers[agent_id] = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': []
        }

    def check_update(self):
        """Check if we should update (called once per episode cycle, not per agent)"""
        self.episodes_collected += 1
        if self.episodes_collected >= self.batch_size:
            self.learn()
            self.episodes_collected = 0

    def calculate_gae(self):
        advantages = []
        last_advantage = 0

        rewards = torch.tensor(self.batch_rewards, dtype=torch.float32)
        values = torch.cat(self.batch_values).squeeze()
        dones = torch.tensor(self.batch_dones, dtype=torch.float32)
        next_values = torch.cat(self.batch_next_values).squeeze()
        
        ext_values = torch.zeros(len(rewards) + 1)
        ext_values[:-1] = values
        
        episode_end_idx = 0
        for i, is_done in enumerate(dones):
            if is_done or i == len(dones) - 1:
                ext_values[i + 1] = next_values[episode_end_idx]
                episode_end_idx += 1
            else:
                ext_values[i + 1] = values[i + 1] if i + 1 < len(values) else 0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * ext_values[t + 1] * (1 - dones[t])
                - values[t]
            )
            last_advantage = (
                delta.detach()
                + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        return advantages, returns

    def learn(self):
        if len(self.batch_states) == 0:
            return

        advantages, returns = self.calculate_gae()

        old_states = torch.cat(self.batch_states).squeeze(1)
        old_actions = torch.cat(self.batch_actions)
        old_log_probs = torch.cat(self.batch_log_probs)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = old_states.size(0)

        for epoch in range(self.K_epochs):
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                minibatch_indices = indices[start_idx:end_idx]
                
                mb_states = old_states[minibatch_indices]
                mb_actions = old_actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                
                new_log_probs_dist = torch.distributions.Categorical(
                    logits=self.actor(mb_states)
                )
                new_log_probs = new_log_probs_dist.log_prob(mb_actions)
                new_values = self.critic(mb_states).squeeze()
                
                entropy = new_log_probs_dist.entropy().mean()

                # Critic loss
                critic_loss = F.mse_loss(new_values, mb_returns.detach())

                # Actor loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * mb_advantages
                )
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                # Update networks
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

        # Clear batch buffers
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []
        self.batch_next_values = []

    def save(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])


# --- Load Reference Prey Agent ---
def load_reference_prey():
    """Load the reference prey agent from reference_agents_source."""
    import sys
    import importlib.util
    
    prey_path = Path("reference_agents_source/prey_agent.py")
    if not prey_path.exists():
        raise FileNotFoundError(f"Reference prey not found at {prey_path}")
    
    spec = importlib.util.spec_from_file_location("prey_module", prey_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.StudentAgent()


# --- Training Function ---
def train_simple_tag(
    num_episodes=10000,
    max_steps=100,
    save_every=500,
    render=False,
    resume_path=None,
    start_episode=1
):
    """Train PPO agents on Simple Tag environment."""
    
    print("=" * 60)
    print("Training PPO on Simple Tag Environment")
    print("=" * 60)
    print("Training against reference prey (used for evaluation)")
    print("=" * 60)
    
    # Create environment
    env = simple_tag_v3.parallel_env(
        num_good=1,      # 1 prey
        num_adversaries=3,  # 3 predators
        num_obstacles=2,
        max_cycles=max_steps,
        continuous_actions=False,
        render_mode='human' if render else None
    )
    
    # Get observation and action dimensions
    env.reset()
    sample_agent = env.agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Agents: {env.agents}")
    
    # Create agents
    # SHARED POLICY: One agent for all predators
    shared_agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon_clip=0.2,
        K_epochs=4,
        batch_size=32,
        minibatch_size=256,
        entropy_coef=0.01,
        gae_lambda=0.95,
    )

    if resume_path:
        print(f"Loading checkpoint from {resume_path}...")
        shared_agent.load(resume_path)
        print(" Checkpoint loaded")

    # Load the reference prey agent (the one used for evaluation)
    print("\nLoading reference prey agent...")
    try:
        prey_agent = load_reference_prey()
        print(" Reference prey loaded successfully")
        print(" Training will use the REAL evaluation prey")
    except Exception as e:
        print(f" CRITICAL ERROR: Failed to load reference prey: {e}")
        raise RuntimeError("Cannot train without reference prey")
    
    # Training metrics
    episode_rewards = []
    predator_rewards_history = []
    prey_rewards_history = []
    
    print(f"\nStarting training for {num_episodes} episodes (starting from {start_episode})...")
    print("-" * 60)
    
    for episode in range(start_episode, num_episodes + 1):
        # Update hyperparameters (decay)
        shared_agent.update_hyperparameters(episode, num_episodes)

        observations, infos = env.reset()
        
        episode_predator_reward = 0
        episode_prey_reward = 0
        
        # Identify predators
        predators = [agent_id for agent_id in env.agents if "adversary" in agent_id]
        
        for step in range(max_steps):
            actions = {}
            
            # Get actions for all agents
            for agent_id in env.agents:
                obs = observations[agent_id]
                
                if "adversary" in agent_id:  # Predator
                    # Use shared agent
                    actions[agent_id] = shared_agent.select_action(obs, agent_id)
                else:  # Prey
                    actions[agent_id] = prey_agent.get_action(obs, agent_id)
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store rewards for predators with REWARD SHAPING
            for agent_id in predators:
                raw_reward = rewards[agent_id]
                
                # Reward shaping: bonus based on distance to prey
                # obs[2:4] contains relative position of prey
                prey_rel_pos = observations[agent_id][2:4]
                dist_to_prey = float(np.linalg.norm(prey_rel_pos))
                
                # Closer to prey = higher bonus (0.1 * (1.0 - normalized_distance))
                distance_bonus = 0.1 * (1.0 - min(dist_to_prey, 1.0))
                shaped_reward = raw_reward + distance_bonus
                
                shared_agent.store_reward(shaped_reward, agent_id)
                episode_predator_reward += shaped_reward
            
            # Track prey reward
            for agent_id in env.agents:
                if "agent" in agent_id:  # Prey
                    episode_prey_reward += rewards[agent_id]
            
            observations = next_observations
            
            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                break
        
        # End episode for all predator agents
        for agent_id in predators:
            final_obs = observations.get(agent_id, None)
            done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
            shared_agent.end_episode(
                agent_id,
                next_state=final_obs if final_obs is not None else None,
                done=done
            )
            
        # Check for update (once per episode)
        shared_agent.check_update()
        
        # Track rewards
        avg_predator_reward = episode_predator_reward / len(predators)
        predator_rewards_history.append(avg_predator_reward)
        prey_rewards_history.append(episode_prey_reward)
        
        # Print progress
        if episode % 100 == 0:
            recent_predator = np.mean(predator_rewards_history[-100:])
            recent_prey = np.mean(prey_rewards_history[-100:])
            print(f"Episode {episode:4d} | "
                  f"Predator Avg: {recent_predator:7.2f} | "
                  f"Prey: {recent_prey:7.2f}")
        
        # Save checkpoint
        if episode % save_every == 0:
            save_dir = Path("checkpoints_v5")
            save_dir.mkdir(exist_ok=True)
            
            save_path = save_dir / f"predator_ppo_episode_{episode}.pth"
            shared_agent.save(save_path)
            print(f"✓ Checkpoint saved: {save_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final save
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    final_path = save_dir / "predator_ppo_final.pth"
    shared_agent.save(final_path)
    print(f" Final model saved: {final_path}")
    
    # Plot training curves
    plot_training_results(predator_rewards_history, prey_rewards_history)
    
    env.close()
    
    return shared_agent, predator_rewards_history, prey_rewards_history


def plot_training_results(predator_rewards, prey_rewards):
    """Plot training curves."""
    plt.figure(figsize=(12, 5))
    
    # Predator rewards
    plt.subplot(1, 2, 1)
    plt.plot(predator_rewards, alpha=0.3, label='Episode Reward')
    window = 100
    if len(predator_rewards) >= window:
        moving_avg = np.convolve(predator_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(predator_rewards)), moving_avg, 
                label=f'{window}-episode Moving Avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Predator Reward')
    plt.title('Predator Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prey rewards
    plt.subplot(1, 2, 2)
    plt.plot(prey_rewards, alpha=0.3, label='Episode Reward')
    if len(prey_rewards) >= window:
        moving_avg = np.convolve(prey_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(prey_rewards)), moving_avg,
                label=f'{window}-episode Moving Avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Prey Reward')
    plt.title('Prey Performance (Reference Agent)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print(f" Training plot saved: training_results.png")
    plt.close()  # Close figure to free memory


if __name__ == "__main__":
    # Check for existing checkpoints to resume
    import glob
    import os
    
    checkpoint_dir = "checkpoints_v5"
    checkpoints = glob.glob(f"{checkpoint_dir}/predator_ppo_episode_*.pth")
    
    resume_path = None
    start_episode = 1
    
    if checkpoints:
        # Sort by episode number
        try:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print(f"Found existing checkpoint: {latest_checkpoint}")
            resume_path = latest_checkpoint
            start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
        except Exception as e:
            print(f"Could not parse checkpoint version: {e}")

    # Train the agents
    agents, predator_history, prey_history = train_simple_tag(
        num_episodes=30000,
        max_steps=100,
        save_every=500,
        render=False,  # Set to True to visualize training
        resume_path=resume_path,
        start_episode=start_episode
    )
    
    print("\nTraining completed! Model saved in 'models/predator_ppo_final.pth'")
