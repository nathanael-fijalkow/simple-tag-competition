"""
Student predator submission for Simple Tag competition.


"""

from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class PredatorPolicyNetwork(nn.Module):
    """MLP policy netwo"""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, action_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)  # logits


class StudentAgent:
    

    def __init__(self):
        self.submission_dir = Path(__file__).parent

        # Deterministic RNG per predator id for fallback
        self._rng = {}

        self.use_model = False
        self.device = "cpu"
        self.model = None

        model_path = self.submission_dir / "predator_model.pth"
        if TORCH_AVAILABLE and model_path.exists():
            try:
                # Load weights on CPU first
                state = torch.load(model_path, map_location="cpu")

                # Infer obs_dim from first Linear layer weight: (hidden_dim, obs_dim)
                # Our network is: network.0 is the first Linear
                first_w = state["network.0.weight"]
                obs_dim = int(first_w.shape[1])

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = PredatorPolicyNetwork(obs_dim=obs_dim, hidden_dim=256, action_dim=5).to(self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.use_model = True
            except Exception:
                # Never crash evaluation; fall back safely
                self.use_model = False
                self.model = None
                self.device = "cpu"

    def _get_rng(self, agent_id: str):
        if agent_id not in self._rng:
            # stable per-agent seed
            seed = abs(hash(agent_id)) % (2**32)
            self._rng[agent_id] = np.random.default_rng(seed)
        return self._rng[agent_id]

    def get_action(self, observation, agent_id: str):
        """
        observation: numpy array (predator obs, typically shape (16,))
        agent_id: e.g., 'adversary_0'

        returns: int in [0..4]
        """
        if self.use_model and self.model is not None:
            try:
                obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(obs_t)
                    action = int(torch.argmax(logits, dim=1).item())
                # safety clamp
                if 0 <= action <= 4:
                    return action
            except Exception:
                pass  # fall through to fallback

        # Deterministic fallback (won't crash, consistent across calls)
        rng = self._get_rng(agent_id)
        return int(rng.integers(0, 5))


if __name__ == "__main__"
    agent = StudentAgent()
    dummy_obs = np.random.randn(16).astype(np.float32)
    print("Action:", agent.get_action(dummy_obs, "adversary_0"))
