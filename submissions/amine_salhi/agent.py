"""
PPO Implementation with Reward Shaping and Orthogonal Initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque
from pettingzoo.mpe import simple_tag_v3
import importlib.util

# Initialisation orthogonale des couches
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialisation orthogonale"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# L'implémentation du PPO utilisée ici est basé sur celle vu en tps RL.
# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        )

    def forward(self, state):
        return self.network(state)

# Critic Network (Value) 
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )

    def forward(self, state):
        return self.network(state)

# Template de l'agent prédateur implémenté via PPO
class StudentAgent:
    """Agent prédateur entraîné via PPO pour l'évaluation."""
    
    def __init__(self):
        self.submission_dir = Path(__file__).parent
        self.state_dim = 16
        self.action_dim = 5
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.load_models()
        self.actor.eval()
    
    def load_models(self):
        """Charge les poids du modèle entraîné."""
        model_path = self.submission_dir / "predator_model.pth"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.actor.load_state_dict(checkpoint["actor_state_dict"])
                print(f"Loaded trained predator model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        else:
            print(f"Warning: No trained model found at {model_path}")
    
    def get_action(self, observation, agent_id: str):
        """Sélectionne l'action optimale (greedy) pour l'évaluation."""
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state)
            action = torch.argmax(logits, dim=1).item()
        return action

# PPO Trainer: a class to handle training
class PPOTrainer:
    def __init__(
        self,
        state_dim=16,
        action_dim=5,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon_clip=0.2,
        K_epochs=5,      
        gae_lambda=0.95,
        entropy_coef=0.02,
    ):
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Buffers
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []
    
    # Sélection d'action
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        
        return action.item()

    def learn(self):
        """Mise à jour PPO."""
        if not self.states:
            return
            
        # Calcul des avantages via GAE
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        # Conversion en tenseurs
        s = torch.FloatTensor(np.array(self.states))
        a = torch.LongTensor(self.actions)
        lp = torch.FloatTensor(self.log_probs)
        adv = torch.FloatTensor(advantages)
        ret = torch.FloatTensor(returns)
        
        # Normalisation des avantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(self.K_epochs):
            logits = self.actor(s)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            entropy = dist.entropy().mean()
            
            values = self.critic(s).squeeze()
            
            ratio = torch.exp(new_lp - lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * adv
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(values, ret)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
        # Nettoyage
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

# Entraînement des prédateurs avec PPO et Reward Shaping
def train_predators(prey_agent_path, num_episodes=10000, max_steps=25, save_path="predator_model.pth", print_every=100):
    # Chargement de la proie
    spec = importlib.util.spec_from_file_location("prey_module", prey_agent_path)
    prey_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prey_module)
    prey_agent = prey_module.StudentAgent()

    trainer = PPOTrainer()
    scores = []
    scores_deque = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=max_steps, continuous_actions=False)
        env.reset(seed=episode)
        
        episode_reward = 0
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if done:
                action = None
            elif 'adversary' in agent_name:
                action = trainer.select_action(observation)
                
                # Le REWARD SHAPING sert à encourager la proximité avec la proie
                
                prey_rel_pos = observation[2:4]
                dist = np.linalg.norm(prey_rel_pos)
                
                # On ajoute un bonus inversement proportionnel à la distance
                shaped_reward = reward + (0.1 * (1.0 - dist))
                
                trainer.rewards.append(shaped_reward)
                trainer.dones.append(done)
                episode_reward += shaped_reward
            else:
                action = prey_agent.get_action(observation, agent_name)
            
            env.step(action)
        
        # Mise à jour après chaque batch 
        if episode % 32 == 0:
            trainer.learn()
            
        scores.append(episode_reward)
        scores_deque.append(episode_reward)
        
        if episode % print_every == 0:
            print(f"Ep {episode:5d} | Moyenne (100): {np.mean(scores_deque):7.2f}")
        
    torch.save({
        "actor_state_dict": trainer.actor.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
    }, save_path)
    
    return trainer, scores

if __name__ == "__main__":
    train_predators(prey_agent_path="prey_agent.py")