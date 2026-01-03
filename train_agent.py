import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3

# =========================
# Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ENVS = 8
ROLLOUT_STEPS = 256
TOTAL_UPDATES = 2000
MAX_EPISODE_LENGTH = 200

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 7e-4
PPO_EPOCHS = 8
MINI_BATCH = 1024
ENTROPY_START = 0.01
ENTROPY_END = 0.001
GRAD_CLIP = 0.5

ACTION_DIM = 5
NUM_ADVERSARIES = 3
ADV_IDS = [f"adversary_{i}" for i in range(NUM_ADVERSARIES)]
PREY_ID = "agent_0"

# =========================
# Normalisation des observations
# =========================
class NormTracker:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, data):
        data = np.asarray(data, np.float64)
        mean_batch, var_batch, count_batch = data.mean(0), data.var(0), data.shape[0]
        delta = mean_batch - self.mean
        total_count = self.count + count_batch
        self.mean += delta * count_batch / total_count
        m_a, m_b = self.var * self.count, var_batch * count_batch
        M2 = m_a + m_b + (delta**2) * self.count * count_batch / total_count
        self.var, self.count = M2 / total_count, total_count

    def normalize(self, x, clip_value=10.0):
        normed = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normed, -clip_value, clip_value)

# =========================
# Agent proie (optionnel)
# =========================
class PreyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, action_dim=ACTION_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.model(obs)

class PreyAgent:
    def __init__(self, model_path: Path, obs_dim: int):
        self.use_model = False
        self.obs_dim = obs_dim
        if model_path.exists():
            try:
                self.net = PreyNet(obs_dim)
                self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.net.eval()
                self.use_model = True
                print(f"Prey model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load prey model ({e}), using random actions.")

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        if not self.use_model:
            return int(np.random.randint(0, ACTION_DIM))
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.net(x)
        return int(torch.argmax(logits, dim=1).item())

# =========================
# PPO Networks
# =========================
class PredatorActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM)
        )

    def forward(self, obs):
        return self.model(obs)

class PredatorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state).squeeze(-1)

# =========================
# GAE
# =========================
@torch.no_grad()
def calc_gae(rewards, dones, values, last_value, gamma=GAMMA, lam=LAMBDA):
    T, E = rewards.shape
    advantages = torch.zeros(T, E, device=rewards.device)
    last_adv = torch.zeros(E, device=rewards.device)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_val = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns

# =========================
# Création environnement
# =========================
def create_env():
    return simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=NUM_ADVERSARIES, num_obstacles=2,
        max_cycles=MAX_EPISODE_LENGTH, continuous_actions=False
    )

# =========================
# Fonction principale
# =========================
def train():
    torch.set_num_threads(1)
    device = torch.device(DEVICE)

    # Environnements
    envs = [create_env() for _ in range(NUM_ENVS)]
    obs_envs = [e.reset()[0] for e in envs]

    # Détecter dimensions
    OBS_DIM = obs_envs[0][ADV_IDS[0]].shape[0]
    PREY_OBS_DIM = obs_envs[0][PREY_ID].shape[0]
    STATE_DIM = OBS_DIM * NUM_ADVERSARIES
    print(f"OBS_DIM={OBS_DIM}, PREY_OBS_DIM={PREY_OBS_DIM}, STATE_DIM={STATE_DIM}")

    # Réseaux
    actor = PredatorActor(OBS_DIM).to(device)
    critic = PredatorCritic(STATE_DIM).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Normalizers
    obs_norm = NormTracker((OBS_DIM,))
    state_norm = NormTracker((STATE_DIM,))

    prey = PreyAgent(Path(__file__).parent / "prey_model.pth", obs_dim=PREY_OBS_DIM)

    def flatten_state(obs_dict):
        return np.concatenate([np.array(obs_dict[aid], dtype=np.float32) for aid in ADV_IDS])

    global_steps = 0
    rolling_mean = []

    for update in range(1, TOTAL_UPDATES + 1):
        # Entropy decay
        ent_coef = ENTROPY_START + min(1.0, update/(TOTAL_UPDATES*0.7)) * (ENTROPY_END - ENTROPY_START)

        # Update normalizers
        obs_samples, state_samples = [], []
        for env_obs in obs_envs:
            obs_samples.extend([np.array(env_obs[aid], dtype=np.float32) for aid in ADV_IDS])
            state_samples.append(flatten_state(env_obs))
        obs_norm.update(np.stack(obs_samples))
        state_norm.update(np.stack(state_samples))

        # Rollout buffers
        buf_obs = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, NUM_ADVERSARIES, OBS_DIM, device=device)
        buf_state = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, STATE_DIM, device=device)
        buf_act = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, NUM_ADVERSARIES, dtype=torch.long, device=device)
        buf_logp = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, NUM_ADVERSARIES, device=device)
        buf_rew = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, device=device)
        buf_done = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, device=device)
        buf_val = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, device=device)

        # Rollout
        for t in range(ROLLOUT_STEPS):
            obs_batch = np.zeros((NUM_ENVS, NUM_ADVERSARIES, OBS_DIM), dtype=np.float32)
            state_batch = np.zeros((NUM_ENVS, STATE_DIM), dtype=np.float32)

            for e, env_obs in enumerate(obs_envs):
                for i, aid in enumerate(ADV_IDS):
                    obs_batch[e, i] = obs_norm.normalize(np.array(env_obs[aid], dtype=np.float32))
                state_batch[e] = state_norm.normalize(flatten_state(env_obs))

            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_batch, dtype=torch.float32, device=device)

            # Sample actions
            with torch.no_grad():
                logits = actor(obs_tensor.reshape(-1, OBS_DIM))
                dist = torch.distributions.Categorical(logits=logits)
                acts_flat = dist.sample()
                log_probs = dist.log_prob(acts_flat).view(NUM_ENVS, NUM_ADVERSARIES)
                acts_env = acts_flat.view(NUM_ENVS, NUM_ADVERSARIES).cpu().numpy()
                vals = critic(state_tensor)

            next_obs_envs = []
            team_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
            dones_env = np.zeros(NUM_ENVS, dtype=np.float32)

            for e, env in enumerate(envs):
                actions = {aid: int(acts_env[e, i]) for i, aid in enumerate(ADV_IDS)}
                actions[PREY_ID] = prey.select_action(np.array(obs_envs[e][PREY_ID], dtype=np.float32))
                nxt_obs, rewards, terms, truncs, _ = env.step(actions)

                team_rewards[e] = np.mean([rewards.get(aid, 0.0) for aid in ADV_IDS])
                dones_env[e] = float(any(terms.values()) or any(truncs.values()))
                if dones_env[e]:
                    nxt_obs, _ = env.reset()
                next_obs_envs.append(nxt_obs)

            buf_obs[t] = obs_tensor
            buf_state[t] = state_tensor
            buf_act[t] = torch.tensor(acts_env, device=device)
            buf_logp[t] = log_probs
            buf_rew[t] = torch.tensor(team_rewards, device=device)
            buf_done[t] = torch.tensor(dones_env, device=device)
            buf_val[t] = vals

            obs_envs = next_obs_envs
            global_steps += NUM_ENVS

        # Bootstrap last value
        with torch.no_grad():
            last_states = torch.tensor([state_norm.normalize(flatten_state(obs_envs[e])) for e in range(NUM_ENVS)],
                                       dtype=torch.float32, device=device)
            last_val = critic(last_states)

        # Compute advantages
        advantages, returns = calc_gae(buf_rew, buf_done, buf_val, last_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten tensors
        B = ROLLOUT_STEPS * NUM_ENVS
        obs_flat = buf_obs.reshape(B, NUM_ADVERSARIES, OBS_DIM)
        act_flat = buf_act.reshape(B, NUM_ADVERSARIES)
        logp_old_flat = buf_logp.reshape(B, NUM_ADVERSARIES)
        state_flat = buf_state.reshape(B, STATE_DIM)
        ret_flat = returns.reshape(B)
        adv_expanded = advantages.reshape(B, 1).expand(B, NUM_ADVERSARIES)
        idx_perm = torch.randperm(B, device=device)

        # PPO update
        for _ in range(PPO_EPOCHS):
            for start in range(0, B, MINI_BATCH):
                batch_idx = idx_perm[start:start + MINI_BATCH]
                logits = actor(obs_flat[batch_idx].reshape(-1, OBS_DIM))
                dist = torch.distributions.Categorical(logits=logits)
                logp_new = dist.log_prob(act_flat[batch_idx].reshape(-1)).view(-1, NUM_ADVERSARIES)
                entropy = dist.entropy().mean()
                ratio = torch.exp(logp_new - logp_old_flat[batch_idx])
                pi_loss = -torch.min(ratio * adv_expanded[batch_idx],
                                     torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_expanded[batch_idx]).mean()
                pi_loss -= ent_coef * entropy

                opt_actor.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                opt_actor.step()

                v_pred = critic(state_flat[batch_idx])
                v_loss = F.mse_loss(v_pred, ret_flat[batch_idx])
                opt_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                opt_critic.step()

        # Logging
        mean_team = float(buf_rew.mean().item())
        rolling_mean.append(mean_team)
        if len(rolling_mean) > 100:
            rolling_mean.pop(0)

        if update % 25 == 0:
            print(f"Update {update}/{TOTAL_UPDATES} | Steps {global_steps} | "
                  f"Team mean {mean_team:+.4f} | Rolling100 {np.mean(rolling_mean):+.4f} | Ent {ent_coef:.4f}")

        # Save model periodically
        if update % 200 == 0:
            out_path = Path(__file__).parent / "predator_model.pth"
            torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out_path)
            print(f"Saved -> {out_path}")

    # Final save
    out_path = Path(__file__).parent / "predator_model.pth"
    torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out_path)
    print(f"Training done. Model saved -> {out_path}")


if __name__ == "__main__":
    train()
