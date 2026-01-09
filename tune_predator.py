"""
Usage:
    python tune_predator.py --n-trials 50 --episodes 2000
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import importlib.util

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pettingzoo.mpe import simple_tag_v3
    

# Import from your agent
sys.path.insert(0, 'submissions/test_student')
from agent import PPOTrainer 


def load_prey_agent(prey_path):
    """Load the reference prey agent."""
    spec = importlib.util.spec_from_file_location("prey_module", prey_path)
    prey_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prey_module)
    return prey_module


def train_with_hyperparameters(trial, num_episodes=2000, prey_agent_path="reference_agents_source/prey_agent.py"):
    lr_actor = trial.suggest_float('lr_actor', 1e-5, 1e-2, log=True)
    lr_critic = trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    epsilon_clip = trial.suggest_float('epsilon_clip', 0.1, 0.3)
    K_epochs = trial.suggest_int('K_epochs', 3, 15)
    gae_lambda = trial.suggest_float('gae_lambda', 0.90, 0.99)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    
    prey_module = load_prey_agent(prey_agent_path)
    
    trainer = PPOTrainer(
        num_predators=3,
        state_dim=16,
        action_dim=5,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        epsilon_clip=epsilon_clip,
        K_epochs=K_epochs,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size
    )
    
    scores = []
    recent_scores = []
    
    for episode in range(1, num_episodes + 1):
        env = simple_tag_v3.env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,
            continuous_actions=False
        )
        
        env.reset(seed=trial.number * 10000 + episode)
        prey_agent = prey_module.StudentAgent()
        
        episode_reward = 0
        current_actions = None
        timestep_data = {i: {'obs': None, 'reward': 0} for i in range(3)}
        
        for agent_name in env.agent_iter(25):
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                if 'adversary' in agent_name:
                    pred_idx = int(agent_name.split('_')[1])
                    timestep_data[pred_idx]['obs'] = observation
                    timestep_data[pred_idx]['reward'] = reward
                    episode_reward += reward
                    
                    if all(timestep_data[i]['obs'] is not None for i in range(3)):
                        obs_list = [timestep_data[i]['obs'] for i in range(3)]
                        rewards_list = [timestep_data[i]['reward'] for i in range(3)]
                        actions, log_probs = trainer.select_actions(obs_list)
                        trainer.store_transition(obs_list, actions, log_probs, rewards_list, done=False)
                        current_actions = actions
                        timestep_data = {i: {'obs': None, 'reward': 0} for i in range(3)}
                    
                    if current_actions is not None:
                        action = current_actions[pred_idx]
                    else:
                        action = env.action_space(agent_name).sample()
                else:
                    action = prey_agent.get_action(observation, agent_name)
            
            env.step(action)
        
        trainer.end_episode()
        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        if episode % 100 == 0 and len(recent_scores) >= 50:
            avg_score = np.mean(recent_scores)
            trial.report(avg_score, episode)
            
            # Prune unpromising trials
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()
        
        env.close()
    
    # Return average of last 100 episodes
    final_score = np.mean(scores[-100:])
    return final_score


def tune_hyperparameters(n_trials=50, num_episodes=2000, timeout=None, quick_mode=False):
    
    if quick_mode:
        num_episodes = min(num_episodes, 1000)
        print(f"⚡ Quick mode: Reduced to {num_episodes} episodes per trial")
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: PPO Predators vs Reference Prey")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of trials: {n_trials}")
    print(f"  - Episodes per trial: {num_episodes}")
    print(f"  - Timeout: {timeout if timeout else 'None'}")
    print(f"  - Quick mode: {quick_mode}")
    print("="*70 + "\n")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=200)
    )
    
    # Define objective
    def objective(trial):
        return train_with_hyperparameters(trial, num_episodes=num_episodes)
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout, 
        show_progress_bar=True,
        catch=(Exception,)  
    )
    
    return study


def print_results(study):
    """Print optimization results."""
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\nCompleted trials: {len(completed)}")
    print(f"Pruned trials: {len(pruned)}")
    print(f"Failed trials: {len(failed)}")
    
    if len(completed) == 0:
        print("\nNo trials completed successfully!")
        return
    
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.2f}")
    
    print(f"\n{'Best Hyperparameters:'}")
    print("-" * 50)
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param:20s} = {value:.6f}")
        else:
            print(f"  {param:20s} = {value}")
    
    if len(completed) > 0:
        first_score = completed[0].value
        print(f"\n{'Trial Progress:'}")
        print("-" * 50)
        print(f"  First trial score:  {first_score:.2f}")
        print(f"  Best trial score:   {study.best_value:.2f}")
        print(f"  Improvement:        {study.best_value - first_score:+.2f}")
    
    # Statistics
    if len(completed) > 1:
        scores = [t.value for t in completed]
        print(f"\n{'Score Statistics:'}")
        print("-" * 50)
        print(f"  Mean:   {np.mean(scores):7.2f}")
        print(f"  Std:    {np.std(scores):7.2f}")
        print(f"  Min:    {np.min(scores):7.2f}")
        print(f"  Max:    {np.max(scores):7.2f}")
        print(f"  Median: {np.median(scores):7.2f}")
    
    print("\n" + "="*70 + "\n")


def save_best_config(study, output_file="best_ppo_config.py"):
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("No completed trials to save")
        return
    
    with open(output_file, 'w') as f:
        f.write("# Best PPO hyperparameters found by Optuna\n")
        f.write(f"# Best score: {study.best_value:.2f}\n")
        f.write(f"# Trial #{study.best_trial.number}\n\n")
        
        f.write("BEST_HYPERPARAMETERS = {\n")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                f.write(f"    '{param}': {value:.6f},\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("}\n\n")
        
        f.write("# Usage in agent.py:\n")
        f.write("# trainer = PPOTrainer(\n")
        f.write("#     num_predators=3,\n")
        f.write("#     state_dim=16,\n")
        f.write("#     action_dim=5,\n")
        for param in study.best_params.keys():
            f.write(f"#     {param}=BEST_HYPERPARAMETERS['{param}'],\n")
        f.write("# )\n")
    
    print(f"Saved best configuration to: {output_file}")


def show_parameter_importance(study):
    """Show which parameters had the most impact."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed) < 10:
        print("\nNeed at least 10 completed trials for parameter importance analysis")
        return
    
    try:
        print("\nParameter Importance:")
        print("-" * 50)
        importances = optuna.importance.get_param_importances(study)
        
        # Sort by importance
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        for param, importance in sorted_params:
            bar = "█" * int(importance * 50)
            print(f"  {param:20s} {importance:6.3f} {bar}")
        
        print("\nHigher importance = more impact on performance")
    except ImportError:
        print("\nInstall scikit-learn for parameter importance: pip install scikit-learn")
    except Exception as e:
        print(f"\nCould not compute parameter importance: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Tune PPO hyperparameters using Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard tuning (50 trials, 2000 episodes each)
  python tune_predator.py --n-trials 50
  
  # Quick testing (fewer episodes)
  python tune_predator.py --n-trials 20 --episodes 1000 --quick

  # Long optimization with timeout
  python tune_predator.py --n-trials 100 --timeout 7200

  # Save results to custom file
  python tune_predator.py --n-trials 30 --save my_config.py
        """
    )
    
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials (default: 50)')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Episodes per trial (default: 2000)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: None)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer episodes for testing')
    parser.add_argument('--save', type=str, default='best_ppo_config.py',
                       help='File to save best config (default: best_ppo_config.py)')
    
    args = parser.parse_args()
    
    # Check that prey agent exists
    prey_path = Path("reference_agents_source/prey_agent.py")
    if not prey_path.exists():
        print(f"Error: {prey_path} not found!")
        sys.exit(1)
    
    print("Starting hyperparameter optimization")
    
    
    # Run optimization
    study = tune_hyperparameters(
        n_trials=args.n_trials,
        num_episodes=args.episodes,
        timeout=args.timeout,
        quick_mode=args.quick
    )
    
    print_results(study)
    
    save_best_config(study, args.save)
    
    show_parameter_importance(study)
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
    
    


if __name__ == "__main__":
    main()