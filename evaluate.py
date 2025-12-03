"""
Evaluation script for Simple Tag competition.

This script evaluates student submissions against private reference implementations.
It is designed to be run in the GitHub Actions workflow.
"""

import sys
import json
import importlib.util
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pettingzoo.mpe import simple_tag_v3
except ImportError:
    print("Error: pettingzoo is not installed. Run: pip install pettingzoo[mpe]")
    sys.exit(1)


class AgentLoader:
    """Utility class to load agent implementations."""
    
    @staticmethod
    def load_agent_from_file(file_path: Path):
        """
        Dynamically load a StudentAgent from a Python file.
        
        Args:
            file_path: Path to the agent.py file
            
        Returns:
            Instantiated agent
        """
        try:
            spec = importlib.util.spec_from_file_location("student_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'StudentAgent'):
                raise AttributeError("Module must contain a 'StudentAgent' class")
            
            agent = module.StudentAgent()
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to load agent from {file_path}: {e}")


class SimpleTagEvaluator:
    """Evaluator for Simple Tag environment (predator-only evaluation)."""
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        prey_agent_path: Path,
        predator_agent_path: Path,
        num_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate student predator against reference prey.
        
        Args:
            prey_agent_path: Path to reference prey agent.py (public)
            predator_agent_path: Path to student predator agent.py
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading prey agent from: {prey_agent_path}")
        print(f"Loading predator agent from: {predator_agent_path}")
        
        # Loaders
        prey_loader = lambda: AgentLoader.load_agent_from_file(prey_agent_path)
        predator_loader = lambda: AgentLoader.load_agent_from_file(predator_agent_path)
        
        # Run evaluation
        prey_rewards = []
        predator_rewards = []
        
        for episode in range(num_episodes):
            # Seed all RNGs deterministically for each episode
            episode_seed = episode
            np.random.seed(episode_seed)
            random.seed(episode_seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
            
            env = simple_tag_v3.parallel_env(
                num_good=1,  # Number of prey
                num_adversaries=3,  # Number of predators
                num_obstacles=2,
                max_cycles=max_steps,
                continuous_actions=False
            )
            
            # Seed each episode deterministically with the episode index
            observations, infos = env.reset(seed=episode)
            
            # Initialize agents for this episode
            prey_agents = {}
            predator_agents = {}
            
            episode_prey_reward = 0
            episode_predator_reward = 0
            steps = 0
            
            while env.agents:
                actions = {}
                
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    
                    # Determine if this is prey or predator
                    if "adversary" in agent_id:
                        # This is a predator
                        if agent_id not in predator_agents:
                            predator_agents[agent_id] = predator_loader()
                        action = predator_agents[agent_id].get_action(obs, agent_id)
                    else:
                        # This is prey
                        if agent_id not in prey_agents:
                            prey_agents[agent_id] = prey_loader()
                        action = prey_agents[agent_id].get_action(obs, agent_id)
                    
                    actions[agent_id] = action
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    if "adversary" in agent_id:
                        episode_predator_reward += reward
                    else:
                        episode_prey_reward += reward
                
                steps += 1
                
                if steps >= max_steps:
                    break
            
            prey_rewards.append(episode_prey_reward)
            predator_rewards.append(episode_predator_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")
            
            env.close()
        
        # Calculate statistics
        results = {
            "success": True,
            "prey_score": float(np.mean(prey_rewards)),
            "predator_score": float(np.mean(predator_rewards)),
            "num_episodes": num_episodes
        }
        
        return results


def evaluate_submission(
    student_submission_dir: Path,
    reference_agents_dir: Path,
    output_file: Path,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """
    Evaluate a student predator against public reference prey.
    
    Args:
        student_submission_dir: Directory containing student's agent.py
        reference_agents_dir: Directory containing reference agents
        output_file: Path to save evaluation results
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results dictionary
    """
    student_agent_path = student_submission_dir / "agent.py"
    private_prey_path = reference_agents_dir / "prey_agent.py"
    
    # Validate paths
    if not student_agent_path.exists():
        return {
            "success": False,
            "error": f"Student agent not found at {student_agent_path}"
        }
    
    if not private_prey_path.exists():
        return {
            "success": False,
            "error": "Reference prey agent not found"
        }
    
    print(f"\n{'='*60}")
    print(f"Evaluating submission: {student_submission_dir.name}")
    print(f"{'='*60}\n")
    
    evaluator = SimpleTagEvaluator()
    
    # Evaluate student predator vs reference prey
    print("\n--- Evaluating student PREDATOR vs reference PREY ---")
    predator_results = evaluator.evaluate(
        prey_agent_path=private_prey_path,
        predator_agent_path=student_agent_path,
        num_episodes=num_episodes
    )
    
    # Combine results
    if not predator_results["success"]:
        error_msg = predator_results.get("error", "")
        return {
            "success": False,
            "error": error_msg.strip()
        }
    
    results = {
        "success": True,
        "student": student_submission_dir.name,
        "timestamp": datetime.now().isoformat(),
        "predator_score": predator_results["predator_score"],
        "num_episodes": num_episodes
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Predator score: {results['predator_score']:.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point for evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Simple Tag competition submission")
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to student submission directory"
    )
    parser.add_argument(
        "--reference-agents-dir",
        type=Path,
        default=Path("reference_agents_source"),
        help="Path to reference agents directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_evaluation.json"),
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes"
    )
    
    args = parser.parse_args()
    
    results = evaluate_submission(
        student_submission_dir=args.submission_dir,
        reference_agents_dir=args.reference_agents_dir,
        output_file=args.output,
        num_episodes=args.episodes
    )
    
    if not results["success"]:
        print(f"ERROR: {results['error']}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
