"""
PPO Agent

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs action probabilities, Critic outputs state value.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through both actor and critic."""
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        """Select action and return log probability."""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, state_value
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update."""
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values, entropy


class PPOMemory:
    """Memory buffer for storing trajectories."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def get_batches(self):
        """Convert lists to tensors."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)
        
        return states, actions, old_log_probs, rewards, values, dones


class PPOTrainer:
    """PPO training algorithm."""
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        mini_batch_size=64
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = PPOMemory()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        values = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        """Perform PPO update."""
        # Get all data from memory
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batches()
        
        # Compute advantages and returns
        with torch.no_grad():
            # For last value, we use current value estimate
            next_value = values[-1].item()
            advantages = self.compute_gae(rewards, values, dones, next_value)
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                # Compute ratio for PPO
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    actor_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear memory
        self.memory.clear()


class StudentAgent:
    """
    PPO-based predator agent for Simple Tag environment.
    """
    def __init__(self):
        """Initialize your predator agent."""
        # Observation dimensions for adversary agents
        # Based on simple_tag: [self_vel(2), self_pos(2), landmark_rel_pos(4), 
        #                       other_agent_rel_pos(6), other_agent_vel(6)]
        # Total: 2 + 2 + 4 + 6 + 6 = 20 (can vary based on num agents/obstacles)
        # For adversaries with standard config: obs_dim is typically 14 or 16
        self.obs_dim = 16  # Will be updated on first observation
        self.action_dim = 5  # [no_action, left, right, down, up]
        
        # Initialize PPO trainer
        self.trainer = None
        self.is_training = False  # Set to True if you want to train
        
        # For inference, we'll use a pre-initialized network
        self.policy = None
        self.initialized = False
    
    def _initialize(self, obs):
        """Initialize network with correct observation dimension."""
        if not self.initialized:
            self.obs_dim = len(obs)
            self.policy = ActorCritic(self.obs_dim, self.action_dim)
            
            # If you have a pre-trained model, load it here:
            # self.policy.load_state_dict(torch.load('ppo_predator.pth'))
            
            self.policy.eval()
            self.initialized = True
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment
            agent_id: Unique identifier for this agent instance
            
        Returns:
            action: Action to take in the environment (0-4)
        """
        # Initialize on first call
        if not self.initialized:
            self._initialize(observation)
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action_probs, _ = self.policy(obs_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item()
    
    def save_model(self, path='ppo_predator.pth'):
        """Save the trained model."""
        if self.policy is not None:
            torch.save(self.policy.state_dict(), path)
            print(f"Model saved to {path}")
    
    def load_model(self, path='ppo_predator.pth'):
        """Load a pre-trained model."""
        if self.policy is None:
            # Create policy with default obs_dim, will be reinitialized if needed
            self.policy = ActorCritic(self.obs_dim, self.action_dim)
        
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
        self.initialized = True
        print(f"Model loaded from {path}")


# Training script (optional - run separately to train the agent)
def train_ppo_agent(num_episodes=1000, save_path='ppo_predator.pth'):
    """
    Training function for PPO agent.
    Run this separately to train your agent before submission.
    """
    from pettingzoo.mpe import simple_tag_v3
    
    # Create environment
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False
    )
    
    # Initialize
    observations, infos = env.reset()
    first_adversary = [a for a in env.agents if "adversary" in a][0]
    obs_dim = len(observations[first_adversary])
    action_dim = 5
    
    # Create PPO trainers for each adversary
    trainers = {}
    for agent_id in env.agents:
        if "adversary" in agent_id:
            trainers[agent_id] = PPOTrainer(obs_dim, action_dim)
    
    # Training loop
    episode_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0
        
        for step in range(25):
            actions = {}
            
            for agent_id in env.agents:
                obs = observations[agent_id]
                
                if "adversary" in agent_id:
                    # Get action from PPO agent
                    trainer = trainers[agent_id]
                    action, log_prob, value = trainer.policy.act(
                        torch.FloatTensor(obs)
                    )
                    actions[agent_id] = action
                else:
                    # Random action for prey
                    actions[agent_id] = np.random.randint(0, 5)
            
            # Environment step
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store experience for each adversary
            for agent_id in env.agents:
                if "adversary" in agent_id:
                    trainer = trainers[agent_id]
                    
                    obs = observations[agent_id]
                    action = actions[agent_id]
                    reward = rewards.get(agent_id, 0)
                    done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    
                    # Get log_prob and value again for storage
                    with torch.no_grad():
                        _, log_prob, value = trainer.policy.act(torch.FloatTensor(obs))
                    
                    trainer.memory.add(
                        obs, action, log_prob.item(), reward, value.item(), done
                    )
                    
                    episode_reward += reward
            
            observations = next_observations
            
            if not env.agents:
                break
        
        # Update all adversary agents
        for agent_id, trainer in trainers.items():
            if len(trainer.memory.states) > 0:
                trainer.update()
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        env.close()
    
    # Save the trained model (save first adversary's model)
    first_adversary = list(trainers.keys())[0]
    torch.save(trainers[first_adversary].policy.state_dict(), save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    
    return trainers


if __name__ == "__main__":
    # Example: Train the agent
    print("Training PPO agent for Simple Tag...")
    train_ppo_agent(num_episodes=500, save_path='ppo_predator.pth')