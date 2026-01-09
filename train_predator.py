import sys
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt


def load_prey_agent(prey_path):
    spec = importlib.util.spec_from_file_location("prey_module", prey_path)
    prey_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prey_module)
    return prey_module


def main():
    submission_dir = Path("submissions/test_student")
    prey_agent_path = Path("reference_agents_source/prey_agent.py")
    
    if not prey_agent_path.exists():
        sys.exit(1)
    
    agent_path = submission_dir / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    print("ENTRAÎNEMENT: 3 Prédateurs PPO vs Proie de Référence")
    
    
    # Paramètres d'entraînement
    num_episodes = 50000

    
    
    # Lancer l'entraînement
    save_path = submission_dir / "predator_model.pth"
    trainer, scores = agent_module.train_predators(
        prey_agent_path=str(prey_agent_path),
        num_episodes=num_episodes,
        max_steps=25,
        save_path=str(save_path),
        print_every=100
    )
    
    
    # Statistiques finales
    import numpy as np
    if len(scores) > 0:
        print(f"\nStatistiques finales:")
        print(f"  - Score moyen (100 derniers): {np.mean(scores[-100:]):.2f}")
        print(f"  - Score max: {np.max(scores):.2f}")
        print(f"  - Score min: {np.min(scores):.2f}")
        print(f"  - Épisodes complétés: {len(scores)}")
    
    scores_file = submission_dir / "training_scores.txt"
    with open(scores_file, 'w') as f:
        for i, score in enumerate(scores):
            f.write(f"{i+1},{score}\n")
    
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.3, label='Score par épisode', color='blue')
    
    window = 100
    if len(scores) >= window:
        moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, label=f'Moyenne mobile ({window} épisodes)', 
                linewidth=2, color='red')
    
    plt.xlabel('Épisode')
    plt.ylabel('Score total des prédateurs')
    plt.title('Progression de l\'entraînement PPO (3 Prédateurs vs Proie)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = submission_dir / 'training_progress.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans: {plot_file}")
        
    
    
    


if __name__ == "__main__":
    main()
    