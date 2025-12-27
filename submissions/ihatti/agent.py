import os
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """
    Actor-Critic network.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # actor (policy head)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic (value head)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class StudentAgent:
    """
    Predator agent (inference only).
    Shared model across all instances.
    """

    _policy = None
    _device = torch.device("cpu")

    def __init__(self):
        self.obs_dim = 16
        self.action_dim = 5

        # load model once
        if StudentAgent._policy is None:
            StudentAgent._policy = ActorCritic(self.obs_dim, self.action_dim).to(self._device)
            self._load_model()

        self.policy = StudentAgent._policy

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "ppo_predator.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("ppo_predator.pth not found next to agent.py")

        state_dict = torch.load(model_path, map_location=self._device)
        StudentAgent._policy.load_state_dict(state_dict)
        StudentAgent._policy.eval()
        print("[INFO] PPO model loaded successfully")

    def get_action(self, observation, agent_id):
        """
        Select an action for the predator agent.
        """

        if isinstance(observation, dict):
            observation = observation["observation"]

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = self.policy(obs_tensor)

        # deterministic action
        return torch.argmax(action_probs, dim=-1).item()
