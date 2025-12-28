"""
Local testing script for student agents.

Use this to test your agent locally before submitting.
"""

import sys
from pathlib import Path
import numpy as np

try:
    from pettingzoo.mpe import simple_tag_v3
except ImportError:
    print("Error: pettingzoo not installed. Run: pip install pettingzoo[mpe]")
    sys.exit(1)


def load_agent_class(agent_file: Path):
    """Load the StudentAgent class from a file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("test_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'StudentAgent'):
        raise ValueError("agent.py must contain a 'StudentAgent' class")
    
    return module.StudentAgent


def test_agent(agent_file: Path, num_episodes: int = 100):
    """
    Test a student agent locally.
    
    Args:
        agent_file: Path to agent.py
        num_episodes: Number of episodes to run
    """
    print(f"Loading agent from: {agent_file}")
    
    try:
        AgentClass = load_agent_class(agent_file)
    except Exception as e:
        print(f"Failed to load agent: {e}")
        return False
    
    print("Agent loaded successfully")
    
    # Test predator agent only (students only submit predator)
    print(f"\n--- Testing PREDATOR agent ---")
    
    try:
        agent = AgentClass()
        print(f"Predator agent initialized")
    except Exception as e:
        print(f"Failed to initialize predator agent: {e}")
        return False
    
    # Run a few episodes
    total_reward = 0
    
    for episode in range(num_episodes):
        env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,  # Short episodes for testing
            continuous_actions=False
        )
        
        observations, infos = env.reset()
        
        # Track agents for this episode
        episode_agents = {}
        episode_reward = 0
        steps = 0
        
        while env.agents:
            actions = {}
            
            for agent_id in env.agents:
                obs = observations[agent_id]
                
                # Determine if this is a predator (adversary)
                is_predator = "adversary" in agent_id
                
                if is_predator:
                    # Use student predator agent
                    if agent_id not in episode_agents:
                        episode_agents[agent_id] = AgentClass()
                    
                    try:
                        action = episode_agents[agent_id].get_action(obs, agent_id)
                    except Exception as e:
                        print(f"Error in get_action: {e}")
                        return False
                else:
                    # Random prey agent for testing
                    action = np.random.randint(0, 5)
                
                actions[agent_id] = action
            
            # Step environment
            try:
                observations, rewards, terminations, truncations, infos = env.step(actions)
            except Exception as e:
                print(f"Error during environment step: {e}")
                return False
            
            # Accumulate rewards for predators
            for agent_id, reward in rewards.items():
                if "adversary" in agent_id:
                    episode_reward += reward
            
            steps += 1
            
            if steps >= 25:
                break
        
        total_reward += episode_reward
        env.close()
    
    avg_reward = total_reward / num_episodes
    print(f"Average predator reward over {num_episodes} episodes: {avg_reward:.4f}")
    
    print("\n" + "="*60)
    print("All tests passed! Your agent is ready to submit.")
    print("="*60)
    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_agent.py <path_to_agent.py>")
        print("\nExample:")
        print("  python test_agent.py submissions/myusername/agent.py")
        sys.exit(1)
    
    agent_file = Path(sys.argv[1])
    
    if not agent_file.exists():
        print(f"File not found: {agent_file}")
        sys.exit(1)
    
    if agent_file.name != "agent.py":
        print(f"Warning: File should be named 'agent.py', got '{agent_file.name}'")
    
    success = test_agent(agent_file)
    
    if not success:
        print("\nTests failed. Please fix the errors and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
