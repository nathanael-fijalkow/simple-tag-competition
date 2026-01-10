#!/usr/bin/env python3
"""
Script de rendu final - Entraînement du modèle optimisé pour soumission
Utilise les meilleurs hyperparamètres trouvés par tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from pathlib import Path
import sys
import importlib.util

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent))

try:
    from pettingzoo.mpe import simple_tag_v3
except ImportError:
    print("Erreur: pettingzoo n'est pas installé. Exécutez: pip install pettingzoo[mpe]")
    sys.exit(1)


def load_agent_class(agent_file: Path):
    """Charger la classe StudentAgent depuis un fichier."""
    spec = importlib.util.spec_from_file_location("agent_module", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StudentAgent


# Réseau de neurones Q optimisé
class QNetwork(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=512, output_dim=5):  # Meilleurs params
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# Agent RL utilisant DQN
class StudentAgent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_action(self, observation, agent_id: str):
        obs = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(obs)
            action = q_values.argmax().item()
        return action


# Classe pour l'entraînement DQN avec meilleurs hyperparamètres
class DQNAgent:
    def __init__(self, input_dim=14, hidden_dim=512, output_dim=5, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=100000, batch_size=32):  # Meilleurs params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = QNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        obs = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q values actuels
        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # Q values cibles
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_final_model(num_episodes=1000, max_steps=25, update_freq=2, save_freq=200):
    """
    Entraînement du modèle final avec les meilleurs hyperparamètres
    """
    print("🚀 Entraînement du modèle final avec hyperparamètres optimisés...")
    print("📊 Configuration: lr=0.0001, gamma=0.99, epsilon_decay=0.995, hidden_dim=512, batch_size=32")

    # Charger l'agent proie
    prey_agent_file = Path(__file__).parent / "reference_agents_source" / "prey_agent.py"
    PreyAgentClass = load_agent_class(prey_agent_file)
    prey_agent = PreyAgentClass()

    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=1,
        max_cycles=max_steps,
        continuous_actions=False
    )

    # Utiliser les meilleurs hyperparamètres trouvés
    agent = DQNAgent(
        lr=0.0001,
        gamma=0.99,
        epsilon_decay=0.995,
        hidden_dim=512,
        batch_size=32
    )

    # Pour le reward shaping
    prev_distances_to_prey = {}

    for episode in range(num_episodes):
        observations, infos = env.reset()
        total_reward = 0
        steps = 0
        prev_distances_to_prey = {}

        # Initialiser les distances
        for agent_id in observations:
            if agent_id.startswith("adversary"):
                agent_pos = observations[agent_id][:2]
                prey_pos = observations["agent_0"][:2]
                distance = np.linalg.norm(agent_pos - prey_pos)
                prev_distances_to_prey[agent_id] = distance

        while env.agents:
            actions = {}
            for agent_id in env.agents:
                if agent_id.startswith("adversary"):
                    obs = observations[agent_id]
                    action = agent.select_action(obs)
                    actions[agent_id] = action
                elif agent_id.startswith("agent"):
                    obs = observations[agent_id]
                    action = prey_agent.get_action(obs, agent_id)
                    actions[agent_id] = action

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Reward shaping
            shaped_rewards = {}
            for agent_id in env.agents:
                if agent_id.startswith("adversary"):
                    natural_reward = rewards[agent_id]
                    shaped_reward = natural_reward

                    agent_pos = observations[agent_id][:2]
                    prey_pos = observations["agent_0"][:2]
                    current_distance = np.linalg.norm(agent_pos - prey_pos)

                    if agent_id in prev_distances_to_prey:
                        prev_distance = prev_distances_to_prey[agent_id]
                        distance_change = prev_distance - current_distance
                        shaped_reward += distance_change * 0.2

                    prev_distances_to_prey[agent_id] = current_distance

                    # Pénalités
                    obstacle_positions = [(-0.3, 0.3), (0.3, -0.3)]
                    for obs_pos in obstacle_positions:
                        obs_distance = np.linalg.norm(agent_pos - np.array(obs_pos))
                        if obs_distance < 0.15:
                            shaped_reward -= 0.5

                    shaped_reward -= 0.01
                    shaped_rewards[agent_id] = shaped_reward

            # Stocker les transitions
            for agent_id in env.agents:
                if agent_id.startswith("adversary"):
                    obs = observations[agent_id]
                    action = actions[agent_id]
                    reward = shaped_rewards[agent_id]
                    next_obs = next_observations[agent_id]
                    done = terminations[agent_id] or truncations[agent_id]
                    agent.store_transition(obs, action, reward, next_obs, done)
                    total_reward += reward

            agent.update()
            observations = next_observations
            steps += 1

        # Mise à jour du réseau cible
        if episode % update_freq == 0:
            agent.update_target_network()

        # Sauvegarde périodique
        if episode % save_freq == 0:
            checkpoint_path = f"predator_model_ep{episode}.pth"
            torch.save(agent.q_network.state_dict(), checkpoint_path)
            print(f"💾 Modèle sauvegardé: {checkpoint_path}")

        if episode % 100 == 0:
            print(f"📈 Épisode {episode}/{num_episodes}, Récompense: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Sauvegarde finale
    final_model_path = "predator_model.pth"
    torch.save(agent.q_network.state_dict(), final_model_path)
    print(f"✅ Entraînement terminé! Modèle final sauvegardé: {final_model_path}")
    print("🎯 Le modèle est prêt pour la soumission selon les consignes du README")

    env.close()
    return final_model_path


def evaluate_final_model(model_path="predator_model.pth", num_episodes=20):
    """
    Évaluation du modèle final
    """
    print(f"🔍 Évaluation du modèle {model_path}...")

    prey_agent_file = Path(__file__).parent / "reference_agents_source" / "prey_agent.py"
    PreyAgentClass = load_agent_class(prey_agent_file)
    prey_agent = PreyAgentClass()

    agent = StudentAgent(model_path)

    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=1,
        max_cycles=25,
        continuous_actions=False
    )

    total_rewards = []

    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0

        while env.agents:
            actions = {}
            for agent_id in env.agents:
                if agent_id.startswith("adversary"):
                    obs = observations[agent_id]
                    action = agent.get_action(obs, agent_id)
                    actions[agent_id] = action
                elif agent_id.startswith("agent"):
                    obs = observations[agent_id]
                    action = prey_agent.get_action(obs, agent_id)
                    actions[agent_id] = action

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent_id in env.agents:
                if agent_id.startswith("adversary"):
                    episode_reward += rewards[agent_id]

            observations = next_observations

        total_rewards.append(episode_reward)
        print(f"Épisode {episode + 1}: Récompense = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"📊 Récompense moyenne sur {num_episodes} épisodes: {avg_reward:.2f}")
    env.close()
    return avg_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement du modèle final optimisé")
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement final")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle final")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement")

    args = parser.parse_args()

    if args.train:
        model_path = train_final_model(num_episodes=args.episodes)
        print(f"\n🎉 Modèle entraîné et sauvegardé: {model_path}")
        print("📝 Pour soumettre:")
        print("   1. Copiez predator_model.pth dans votre dossier de soumission")
        print("   2. Modifiez template/agent.py avec votre StudentAgent")
        print("   3. Testez localement puis créez une Pull Request")

    elif args.evaluate:
        evaluate_final_model()

    else:
        print("Utilisez --train pour entraîner ou --evaluate pour évaluer")
        print("Exemple: python rendu.py --train --episodes 1000")