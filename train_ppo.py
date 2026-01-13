import torch
from pettingzoo.mpe import simple_tag_v3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from tqdm import tqdm

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")
    
    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

def make_env():
    env = simple_tag_v3.parallel_env(max_cycles=25, render_mode=None)
    env = ss.pad_observations_v0(env)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env

print("Début entraînement...")
env = make_env()

model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, 
            batch_size=64, n_epochs=10, verbose=0)  # verbose=0 pour cacher les logs SB3

# Entraînement avec barre de progression
total_steps = 500_000
callback = TqdmCallback(total_steps)
model.learn(total_timesteps=total_steps, callback=callback)

# Sauvegarde
torch.save({
    'policy_state_dict': model.policy.state_dict(),
    'observation_space': env.observation_space,
    'action_space': env.action_space
}, "submissions/Rachidbh/predator_model.pth")

print("\n✓ Modèle sauvegardé!")