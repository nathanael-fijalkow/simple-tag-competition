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
    actor-critic network for ppo.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """forward pass through both actor and critic."""
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        """select action and return log probability."""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, state_value
    
    def evaluate(self, states, actions):
        """evaluate actions for ppo update."""
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values, entropy


class PPOMemory:
    """memory buffer for storing trajectories."""
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
        """convert lists to tensors."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)
        
        return states, actions, old_log_probs, rewards, values, dones


class PPOTrainer:
    """ppo training algorithm."""
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
        """compute generalized advantage estimation."""
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
        """perform ppo update."""
        # get all data from memory
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batches()
        
        # compute advantages and returns
        with torch.no_grad():
            next_value = values[-1].item()
            advantages = self.compute_gae(rewards, values, dones, next_value)
            returns = advantages + values
            
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ppo epochs
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            # generate random mini-batches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # sample mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                # compute ratio for ppo
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # total loss
                loss = (
                    actor_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # clear memory
        self.memory.clear()


class StudentAgent:
    """
    ppo-based predator agent for simple tag environment.
    """
    def __init__(self):
        """initialize your predator agent."""
        self.obs_dim = 16
        self.action_dim = 5  # [no_action, left, right, down, up]
        
        # model initialization
        self.device = torch.device("cpu")
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        
        # load weights
        self.load_model()
    
    def load_model(self):
        """load trained model weights."""
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "ppo_predator.pth")
        
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy.eval()  # evaluation mode
            print(f"loaded model from {model_path}")
        else:
            print("warning: no model found, using random weights.")
    
    def get_action(self, observation, agent_id: str):
        """
        get action for the given observation.
        
        args:
            observation: agent's observation from the environment
            agent_id: unique identifier for this agent instance
            
        returns:
            action: action to take in the environment (0-4)
        """
        # handle dict observations if needed
        if isinstance(observation, dict):
            observation = observation['observation']
        
        # convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # get action from policy
        with torch.no_grad():
            action_probs, _ = self.policy(obs_tensor)
            # use argmax for deterministic behavior (or sample for stochastic)
            action = torch.argmax(action_probs).item()
        
        return action


# training script
def train_ppo_agent(num_episodes=1000, save_path='ppo_predator.pth'):
    """
    training function for ppo agent using shared model for all adversaries.
    """
    try:
        from pettingzoo.mpe import simple_tag_v3
    except ImportError:
        print("error: pettingzoo not installed. run: pip install pettingzoo[mpe]")
        return None
    
    print("="*60)
    print("training ppo agent for simple tag")
    print("="*60)
    
    # create environment
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False
    )
    
    # initialize
    observations, infos = env.reset()
    first_adversary = [a for a in env.agents if "adversary" in a][0]
    obs_dim = len(observations[first_adversary])
    action_dim = 5
    
    print(f"observation dimension: {obs_dim}")
    print(f"action dimension: {action_dim}")
    print(f"number of episodes: {num_episodes}")
    print(f"using shared model for all 3 adversaries")
    print("="*60)
    
    # create one shared ppo trainer for all adversaries
    trainer = PPOTrainer(obs_dim, action_dim)
    
    # training loop
    episode_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0
        
        # track actions and values for this episode
        episode_data = {}
        
        for step in range(25):
            actions = {}
            
            for agent_id in env.agents:
                obs = observations[agent_id]
                
                if "adversary" in agent_id:
                    # get action from shared ppo agent
                    action, log_prob, value = trainer.policy.act(
                        torch.FloatTensor(obs)
                    )
                    actions[agent_id] = action
                    episode_data[agent_id] = (action, log_prob, value)
                else:
                    # random action for prey
                    actions[agent_id] = np.random.randint(0, 5)
            
            # environment step
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # store experience for all adversaries in shared memory
            for agent_id in env.agents:
                if "adversary" in agent_id and agent_id in episode_data:
                    obs = observations[agent_id]
                    action, log_prob, value = episode_data[agent_id]
                    reward = rewards.get(agent_id, 0)
                    done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    
                    # add to shared memory
                    trainer.memory.add(
                        obs, action, log_prob.item(), reward, value.item(), done
                    )
                    
                    episode_reward += reward
            
            observations = next_observations
            
            if not env.agents:
                break
        
        # update the shared model (learns from all 3 adversaries)
        if len(trainer.memory.states) > 0:
            trainer.update()
        
        episode_rewards.append(episode_reward)
        
        # print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"episode {episode + 1}/{num_episodes} | avg reward (last 100): {avg_reward:.2f}")
        
        env.close()
    
    # save the trained model
    torch.save(trainer.policy.state_dict(), save_path)


    print(f"training complete! model saved to: {save_path}")

    
    return trainer


if __name__ == "__main__":
    # train the agent when running this file directly
    train_ppo_agent(num_episodes=1000, save_path='ppo_predator.pth')