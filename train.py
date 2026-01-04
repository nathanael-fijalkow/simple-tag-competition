import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import time
import random
import argparse
from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

try:
    from pettingzoo.mpe import simple_tag_v3 as simple_tag
except Exception:
    from pettingzoo.mpe import simple_tag_v2 as simple_tag


# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unwrap_reset(x):
    # reset() may return obs or (obs, infos)
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], dict):
        return x[0]
    return x


def as_np(obs):
    if isinstance(obs, dict):
        if "observation" in obs:
            obs = obs["observation"]
        elif "obs" in obs:
            obs = obs["obs"]
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def agent_index(agent_id: str, n=3):
    m = re.search(r"(\d+)$", agent_id or "")
    if not m:
        return 0
    return int(m.group(1)) % n


def build_actor_input(obs_vec: np.ndarray, aid: str):
    # obs + one-hot predator id (3 predators)
    onehot = np.zeros(3, dtype=np.float32)
    onehot[agent_index(aid, 3)] = 1.0
    return np.concatenate([obs_vec, onehot], axis=0).astype(np.float32)


def load_public_prey_agent():
    """
    Loads reference prey StudentAgent from reference_agent_source/*prey*.py (robust to filename).
    That module must load prey_model.pth via Path(__file__).parent / "prey_model.pth".
    We try both folder names: reference_agent_source and reference_agents_source.
    """
    root = Path(__file__).parent
    candidates = [root / "reference_agent_source", root / "reference_agents_source"]
    base = None
    for c in candidates:
        if c.exists():
            base = c
            break
    if base is None:
        raise FileNotFoundError(f"Cannot find reference prey folder. Tried: {candidates}")

    py_files = sorted(list(base.glob("*prey*.py"))) + sorted(list(base.glob("*.py")))
    last_err = None
    for f in py_files:
        try:
            spec = importlib.util.spec_from_file_location("public_prey_module", str(f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # sets __file__ so Path(__file__) works inside
            if hasattr(mod, "StudentAgent"):
                return mod.StudentAgent()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load public prey StudentAgent. Last error: {last_err}")


# ---------------- networks ----------------
# Actor must match your agent.py _MLPPolicy architecture if you want to load weights there
class ActorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CriticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------- training ----------------
def main():
    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    parser.add_argument("--max_cycles", type=int, default=200)

    # PPO batch/update
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=512)

    # PPO hyperparams
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.03)  # early stop epoch if too large

    # shaping (small! keep it safe)
    parser.add_argument("--shaping_dist_coef", type=float, default=0.01)     # reward += -c * dist(pred, prey)
    parser.add_argument("--shaping_sep_coef", type=float, default=0.02)      # penalty if predators too close
    parser.add_argument("--shaping_sep_thresh", type=float, default=0.20)    # distance threshold for separation penalty

    # saving
    parser.add_argument("--save_every", type=int, default=100_000)
    parser.add_argument("--out", type=str, default=os.path.join("submissions", "sialaoui", "predator_model.pth"))

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cuda available:", torch.cuda.is_available(), "| device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load public prey (same as evaluation)
    prey_agent = load_public_prey_agent()

    # Environment
    env = simple_tag.parallel_env(max_cycles=args.max_cycles, continuous_actions=False)
    obs = unwrap_reset(env.reset(seed=args.seed))

    predators = [a for a in env.possible_agents if "adversary" in a]
    prey = [a for a in env.possible_agents if "adversary" not in a][0]
    n_pred = len(predators)

    # Infer obs dims
    pred_obs_dim = as_np(obs[predators[0]]).shape[0]
    prey_obs_dim = as_np(obs[prey]).shape[0]

    actor_in_dim = pred_obs_dim + 3
    critic_in_dim = pred_obs_dim * n_pred + prey_obs_dim
    n_actions = 5

    actor = ActorMLP(actor_in_dim, hidden=128, out_dim=n_actions).to(device)
    critic = CriticMLP(critic_in_dim, hidden=256).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    def build_critic_input(obs_dict):
        parts = [as_np(obs_dict[a]) for a in predators]
        parts.append(as_np(obs_dict[prey]))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def prey_relative_from_pred_obs(pred_obs_vec: np.ndarray):
        # heuristic based on your agent.py: obs[-4], obs[-3] ~ (dx, dy) predator->prey
        if pred_obs_vec.shape[0] < 4:
            return 0.0, 0.0
        return float(pred_obs_vec[-4]), float(pred_obs_vec[-3])

    global_step = 0
    last_save = 0
    start_time = time.time()
    update_idx = 0

    while global_step < args.total_steps:
        T = args.rollout_steps

        # Buffers
        b_actor_x = np.zeros((T, n_pred, actor_in_dim), dtype=np.float32)
        b_actions = np.zeros((T, n_pred), dtype=np.int64)
        b_logp = np.zeros((T, n_pred), dtype=np.float32)

        b_critic_x = np.zeros((T, critic_in_dim), dtype=np.float32)
        b_value = np.zeros((T,), dtype=np.float32)

        b_reward = np.zeros((T,), dtype=np.float32)
        b_done = np.zeros((T,), dtype=np.float32)

        # Rollout
        for t in range(T):
            actions = {}

            # critic value V(s)
            cx = build_critic_input(obs)
            b_critic_x[t] = cx
            with torch.no_grad():
                v = critic(torch.from_numpy(cx).to(device).unsqueeze(0)).squeeze(0)
            b_value[t] = float(v.item())

            # predators act (shared actor)
            pred_dxdy = []
            for j, aid in enumerate(predators):
                o = as_np(obs[aid])
                ax = build_actor_input(o, aid)
                b_actor_x[t, j] = ax

                with torch.no_grad():
                    logits = actor(torch.from_numpy(ax).to(device).unsqueeze(0)).squeeze(0)
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)

                actions[aid] = int(a.item())
                b_actions[t, j] = int(a.item())
                b_logp[t, j] = float(lp.item())

                pred_dxdy.append(prey_relative_from_pred_obs(o))

            # public prey acts (fixed opponent)
            prey_obs = as_np(obs[prey])
            actions[prey] = int(prey_agent.get_action(prey_obs, prey))

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # --- base team reward (sum predator rewards) ---
            team_r = sum(float(rewards.get(a, 0.0)) for a in predators)

            # --- reward shaping: distance to prey (dense signal) ---
            if args.shaping_dist_coef != 0.0:
                # use next_obs predator dx,dy to avoid shaping on stale obs
                dist_bonus = 0.0
                for aid in predators:
                    o2 = as_np(next_obs[aid])
                    dx, dy = prey_relative_from_pred_obs(o2)
                    dist = (dx * dx + dy * dy) ** 0.5
                    dist_bonus += -args.shaping_dist_coef * dist
                team_r += dist_bonus

            # --- reward shaping: separation (avoid stacking) ---
            if args.shaping_sep_coef != 0.0:
                # approximate predator-predator distances from their relative prey vectors:
                # if dxdy are predator->prey, then predator positions in prey-centered frame are (-dx, -dy).
                pts = []
                for aid in predators:
                    o2 = as_np(next_obs[aid])
                    dx, dy = prey_relative_from_pred_obs(o2)
                    pts.append(np.array([-dx, -dy], dtype=np.float32))
                sep_pen = 0.0
                for i in range(n_pred):
                    for j in range(i + 1, n_pred):
                        d = float(np.linalg.norm(pts[i] - pts[j]))
                        if d < args.shaping_sep_thresh:
                            sep_pen += -(args.shaping_sep_coef * (args.shaping_sep_thresh - d))
                team_r += sep_pen

            b_reward[t] = team_r

            done = (len(next_obs) == 0) or (terms and all(terms.values())) or (truncs and all(truncs.values()))
            b_done[t] = 1.0 if done else 0.0

            global_step += 1
            obs = next_obs

            if done:
                obs = unwrap_reset(env.reset(seed=args.seed + global_step))

            if global_step >= args.total_steps:
                break

        # Bootstrap last value
        with torch.no_grad():
            v_last = critic(torch.from_numpy(build_critic_input(obs)).to(device).unsqueeze(0)).squeeze(0).item()

        # GAE-Lambda
        adv = np.zeros((T,), dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - (b_done[t] if t == T - 1 else b_done[t + 1])
            next_value = (v_last if t == T - 1 else b_value[t + 1])
            delta = b_reward[t] + args.gamma * next_value * next_nonterminal - b_value[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            adv[t] = lastgaelam

        ret = adv + b_value

        # Normalize advantage (standard)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Normalize returns for critic stability (key stability trick)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        # Flatten actor batch (T * n_pred)
        flat_actor_x = b_actor_x.reshape(-1, actor_in_dim)
        flat_actions = b_actions.reshape(-1)
        flat_oldlogp = b_logp.reshape(-1)
        flat_adv = np.repeat(adv, n_pred).astype(np.float32)

        # Critic batch (T)
        flat_critic_x = b_critic_x
        flat_ret = ret.astype(np.float32)

        batch_actor = flat_actor_x.shape[0]
        batch_critic = flat_critic_x.shape[0]
        actor_inds = np.arange(batch_actor)
        critic_inds = np.arange(batch_critic)

        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        n_updates = 0

        # PPO Update
        for epoch in range(args.update_epochs):
            np.random.shuffle(actor_inds)
            np.random.shuffle(critic_inds)

            epoch_kl = 0.0
            epoch_n = 0

            for start in range(0, batch_actor, args.minibatch_size):
                mb = actor_inds[start:start + args.minibatch_size]

                x = torch.from_numpy(flat_actor_x[mb]).to(device)
                a = torch.from_numpy(flat_actions[mb]).to(device)
                old_lp = torch.from_numpy(flat_oldlogp[mb]).to(device)
                A = torch.from_numpy(flat_adv[mb]).to(device)

                logits = actor(x)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a)
                ent = dist.entropy().mean()

                ratio = (lp - old_lp).exp()
                pg1 = ratio * A
                pg2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * A
                policy_loss = -torch.min(pg1, pg2).mean()

                approx_kl = (old_lp - lp).mean()

                # critic minibatch (aligned by time, not by predator)
                cstart = (start // n_pred) % batch_critic
                cmb = critic_inds[cstart:cstart + min(args.minibatch_size, batch_critic)]
                cx = torch.from_numpy(flat_critic_x[cmb]).to(device)
                target = torch.from_numpy(flat_ret[cmb]).to(device)

                v = critic(cx)
                value_loss = ((v - target) ** 2).mean()

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
                optimizer.step()

                mean_policy_loss += float(policy_loss.item())
                mean_value_loss += float(value_loss.item())
                mean_entropy += float(ent.item())
                mean_kl += float(approx_kl.item())
                n_updates += 1

                epoch_kl += float(approx_kl.item())
                epoch_n += 1

            # early stop if KL too large
            avg_epoch_kl = epoch_kl / max(1, epoch_n)
            if args.target_kl is not None and avg_epoch_kl > args.target_kl:
                break

        mean_policy_loss /= max(1, n_updates)
        mean_value_loss /= max(1, n_updates)
        mean_entropy /= max(1, n_updates)
        mean_kl /= max(1, n_updates)
        mean_reward = float(b_reward.mean())

        update_idx += 1
        elapsed = time.time() - start_time
        print(
            f"[update {update_idx:05d}] step={global_step:>9} | "
            f"reward={mean_reward:>8.2f} | "
            f"policy={mean_policy_loss:>9.6f} | "
            f"value={mean_value_loss:>9.6f} | "
            f"entropy={mean_entropy:>6.3f} | "
            f"kl={mean_kl:>7.5f} | "
            f"elapsed={elapsed:.1f}s"
        )

        if global_step - last_save >= args.save_every:
            torch.save({"state_dict": actor.state_dict()}, args.out)
            last_save = global_step
            print(f"[save] step={global_step} -> {args.out}")

    torch.save({"state_dict": actor.state_dict()}, args.out)
    print(f"[done] saved actor -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
