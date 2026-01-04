# Simple-Tag-Competition

## Approche : PPO à politique partagée pour les prédateurs

L’algorithme utilisé est Proximal Policy Optimization (PPO), implémenté à partir du code étudié en cours.
L’objectif est d’entraîner les trois agents prédateurs à coopérer efficacement afin de capturer la proie de référence.

## 1. Principe général

L’approche repose sur l’utilisation d’une politique partagée unique pour les trois prédateurs.
Les agents adverses étant symétriques (même espace d’observation et même espace d’action), ils partagent un réseau Actor–Critic commun.

Chaque prédateur agit de manière indépendante à partir de sa propre observation locale mais l’ensemble des expériences est utilisé pour entraîner une seule politique globale.

Cette approche permet d’exploiter la symétrie du problème, d’augmenter la diversité des expériences collectées et d’améliorer la stabilité et l’efficacité de l’apprentissage,
  sans communication explicite entre agents.

## 2. Apprentissage PPO centralisé avec politique partagée (version  retenue)

Dans l’approche retenue pour le rendu, les transitions générées par les trois prédateurs sont regroupées dans un buffer partagé unique et utilisées comme si elles provenaient d’un seul agent.

Cette approximation introduit un biais inductif favorable à la coordination, en agissant comme une forme implicite de centralisation de l’apprentissage.
Les gradients PPO sont ainsi calculés à partir d’un grand volume d’expériences hétérogènes, ce qui favorise l’émergence rapide de comportements coopératifs entre les prédateurs et améliore la stabilité empirique de l’entraînement.

## 3. Approche PPO multi-agent à trajectoires séparées

Une seconde version a également été développée reposant sur des trajectoires séparées pour chaque prédateur et une récompense d’équipe alignée avec la métrique d’évaluation.

Cette approche est théoriquement plus correcte et conforme au cadre multi-agent classique.
Cependant, dans ce contexte précis, elle conduit à une coordination plus difficile et à des performances empiriques inférieures à la première  version avec un score maximal de 1557.9


## 4. Entraînement
L’entraînement est réalisé contre l’agent proie de référence fourni.
Les hyperparamètres PPO ont d’abord été ajustés manuellement, puis affinés à l’aide d’Optuna afin d’améliorer la stabilité et les performances finales.

## 5. Résultats et choix final

Avec l’approche PPO à politique partagée et buffer partagé, le score obtenu atteint 1771.44
Cette approche a donc été retenue pour le rendu final car elle maximise la métrique officielle de la compétition.
## 6. Organisation du dépôt


* `/submissions/mayamahouachi/agent.py` : définition et chargement de l’agent utilisé pour l’évaluation.
* `train_ppo_predators.py` : script d’entraînement.
* `ppo.py` : implémentation de l’algorithme PPO.
* `tune.py` : optimisation des hyperparamètres avec Optuna.

