#fichier du projet de RL simple tag competition; fait par Nicolas Delaere

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Réseau pour la policy
class PolicyNetwork(nn.Module):
    def __init__(self, taille_obs: int, nombre_actions: int = 5, couches_cachees=(128, 128)):
        super().__init__()
        liste_couches = []
        taille_precedente = taille_obs
        i = 0
        while i < len(couches_cachees):
            taille_actuelle = couches_cachees[i]
            liste_couches.append(nn.Linear(taille_precedente, taille_actuelle))
            liste_couches.append(nn.ReLU())
            taille_precedente = taille_actuelle
            i = i + 1
        liste_couches.append(nn.Linear(taille_precedente, nombre_actions))
        self.reseau = nn.Sequential(*liste_couches)

    def forward(self, entree):
        sortie = self.reseau(entree)
        return sortie


# Réseau pour la valeur
class ValueNetwork(nn.Module):
    def __init__(self, taille_obs: int, couches_cachees=(128, 128)):
        super().__init__()
        liste_couches = []
        taille_precedente = taille_obs
        compteur = 0
        while compteur < len(couches_cachees):
            taille_actuelle = couches_cachees[compteur]
            liste_couches.append(nn.Linear(taille_precedente, taille_actuelle))
            liste_couches.append(nn.ReLU())
            taille_precedente = taille_actuelle
            compteur = compteur + 1
        liste_couches.append(nn.Linear(taille_precedente, 1))
        self.reseau = nn.Sequential(*liste_couches)

    def forward(self, entree):
        sortie = self.reseau(entree)
        sortie = sortie.squeeze(-1)
        return sortie


class StudentAgent:
    def __init__(self):
        self.nombre_actions = 5
        self.politique: Optional[PolicyNetwork] = None
        chemin_fichier = os.path.abspath(__file__)
        dossier = os.path.dirname(chemin_fichier)
        self.chemin_modele = os.path.join(dossier, "predator_model.pth")
        self.appareil = torch.device("cpu")

    def _initialiser_politique_paresseusement(self, observation_recue): # pour la première observation reçue
        observation_tableau = np.asarray(observation_recue, dtype=np.float32)
        observation_aplatie = observation_tableau.flatten()
        dimension_observation = observation_aplatie.shape[0]

        self.politique = PolicyNetwork(taille_obs=dimension_observation, nombre_actions=self.nombre_actions)
        self.politique.to(self.appareil)

        modele_existe = os.path.exists(self.chemin_modele)
        if not modele_existe:
            print("pas de modèle trouvé, utilisation d'une politique aléatoire\n")
        else:
            dictionnaire_etat = torch.load(self.chemin_modele, map_location=self.appareil)
            self.politique.load_state_dict(dictionnaire_etat)
            print("modèle chargé:", self.chemin_modele)

        self.politique.eval()

    def get_action(self, observation, agent_id):
        politique_non_initialisee = self.politique is None
        if politique_non_initialisee:
            self._initialiser_politique_paresseusement(observation)

        observation_tableau = np.asarray(observation, dtype=np.float32)
        observation_aplatie = observation_tableau.flatten()
        tenseur_observation = torch.from_numpy(observation_aplatie)
        tenseur_observation = tenseur_observation.unsqueeze(0)
        tenseur_observation = tenseur_observation.to(self.appareil)

        with torch.no_grad():
            logits_sortie = self.politique(tenseur_observation)
            action_choisie = torch.argmax(logits_sortie, dim=-1)
            action_valeur = action_choisie.item()

        action_finale = int(action_valeur)
        return action_finale


# ============================================================================
# PARTIE ENTRAÎNEMENT PPO
# ============================================================================

class ConfigurationPPO:
    nombre_total_pas: int = 500_000
    pas_par_collecte: int = 512
    facteur_discount: float = 0.99
    lambda_gae: float = 0.95
    coefficient_clip: float = 0.2
    taux_apprentissage: float = 3e-4
    epoques_mise_a_jour: int = 10
    taille_mini_batch: int = 1024
    coefficient_entropie: float = 0.01
    coefficient_valeur: float = 0.5
    norme_gradient_max: float = 0.5


class AgentPPOPredateur:
    """Agent PPO avec politique partagée pour tous les prédateurs"""
    def __init__(self, dimension_obs: int, nombre_actions: int, configuration: ConfigurationPPO, appareil: torch.device):
        self.configuration = configuration
        self.appareil = appareil
        self.reseau_politique = PolicyNetwork(dimension_obs, nombre_actions)
        self.reseau_politique = self.reseau_politique.to(appareil)
        self.reseau_valeur = ValueNetwork(dimension_obs)
        self.reseau_valeur = self.reseau_valeur.to(appareil)
        parametres_politique = list(self.reseau_politique.parameters())
        parametres_valeur = list(self.reseau_valeur.parameters())
        tous_parametres = parametres_politique + parametres_valeur
        self.optimiseur = torch.optim.Adam(tous_parametres, lr=configuration.taux_apprentissage)

    @torch.no_grad()
    def choisir_action(self, observation_numpy: np.ndarray) -> Tuple[int, float, float]:
        """Obtenir action, log probabilité, et valeur pour une observation"""
        tenseur_obs = torch.as_tensor(observation_numpy, dtype=torch.float32, device=self.appareil)
        tenseur_obs = tenseur_obs.unsqueeze(0)
        logits_actions = self.reseau_politique(tenseur_obs)
        distribution = Categorical(logits=logits_actions)
        action_echantillon = distribution.sample()
        log_probabilite = distribution.log_prob(action_echantillon)
        valeur_etat = self.reseau_valeur(tenseur_obs)
        action_int = int(action_echantillon.item())
        logp_float = float(log_probabilite.item())
        val_float = float(valeur_etat.item())
        return action_int, logp_float, val_float


def calculer_avantages_generalises(
    recompenses: np.ndarray,
    episodes_termines: np.ndarray,
    valeurs: np.ndarray,
    facteur_discount: float,
    facteur_lambda: float,
    derniere_valeur: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculer l'estimation des avantages généralisés"""
    nombre_pas = recompenses.shape[0]
    avantages = np.zeros(nombre_pas, dtype=np.float32)
    dernier_avantage = 0.0
    indice = nombre_pas - 1
    while indice >= 0:
        est_terminal = episodes_termines[indice]
        if est_terminal:
            facteur_non_terminal = 0.0
        else:
            facteur_non_terminal = 1.0

        est_dernier_pas = indice == nombre_pas - 1
        if est_dernier_pas:
            valeur_suivante = derniere_valeur
        else:
            valeur_suivante = valeurs[indice + 1]

        erreur_td = recompenses[indice] + facteur_discount * valeur_suivante * facteur_non_terminal - valeurs[indice]
        dernier_avantage = erreur_td + facteur_discount * facteur_lambda * facteur_non_terminal * dernier_avantage
        avantages[indice] = dernier_avantage
        indice = indice - 1

    retours = avantages + valeurs
    return avantages, retours


def entrainer():
    from pettingzoo.mpe import simple_tag_v3
    chemin_parent = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(chemin_parent))
    from reference_agents_source.prey_agent import StudentAgent as AgentProie

    configuration = ConfigurationPPO()
    cuda_disponible = torch.cuda.is_available()
    if cuda_disponible:
        appareil = torch.device("cuda")
    else:
        appareil = torch.device("cpu")
    print(f"Using device: {appareil}")

    # l'env selectionné
    environnement = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False
    )
    environnement.reset()
    liste_agents_possibles = environnement.possible_agents
    noms_predateurs = []
    for nom_agent in liste_agents_possibles:
        if "adversary" in nom_agent:
            noms_predateurs.append(nom_agent)

    liste_proies = []
    for nom_agent in liste_agents_possibles:
        if "agent" in nom_agent:
            liste_proies.append(nom_agent)
    nom_proie = liste_proies[0]
    premier_predateur = noms_predateurs[0]

    espace_obs = environnement.observation_space(premier_predateur)
    forme_obs = espace_obs.shape
    dimension_obs_predateur = int(np.prod(forme_obs))
    espace_action = environnement.action_space(premier_predateur)
    nombre_actions = espace_action.n
    print("début de l'entraînement\n")
    agent_predateur = AgentPPOPredateur(dimension_obs_predateur, nombre_actions, configuration, appareil)
    agent_proie = AgentProie()

    compteur_pas = 0
    meilleure_recompense_moyenne = float('-inf')

    pas_total_atteint = compteur_pas >= configuration.nombre_total_pas
    while not pas_total_atteint:
        observations_dictionnaire, _ = environnement.reset()

        # collections
        trajectoires = {}
        for nom_pred in noms_predateurs:
            trajectoires[nom_pred] = {
                'obs': [], 'act': [], 'logp': [], 'rew': [], 'done': [], 'val': []
            }

        compteur_rollout = 0
        while compteur_rollout < configuration.pas_par_collecte:
            nombre_agents_actifs = len(environnement.agents)
            if nombre_agents_actifs == 0:
                observations_dictionnaire, _ = environnement.reset()

            actions_a_faire = {}

            # actions des prédateurs
            for nom_predateur in noms_predateurs:
                predateur_present = nom_predateur in observations_dictionnaire
                if predateur_present:
                    observation_brute = observations_dictionnaire[nom_predateur]
                    observation_array = np.asarray(observation_brute, dtype=np.float32)
                    observation_plate = observation_array.flatten()
                    action_choisie, log_prob, valeur = agent_predateur.choisir_action(observation_plate)
                    actions_a_faire[nom_predateur] = action_choisie
                    trajectoires[nom_predateur]['obs'].append(observation_plate)
                    trajectoires[nom_predateur]['act'].append(action_choisie)
                    trajectoires[nom_predateur]['logp'].append(log_prob)
                    trajectoires[nom_predateur]['val'].append(valeur)

            # action de la proie
            proie_presente = nom_proie in observations_dictionnaire
            if proie_presente:
                observation_proie_brute = observations_dictionnaire[nom_proie]
                observation_proie_array = np.asarray(observation_proie_brute, dtype=np.float32)
                observation_proie_plate = observation_proie_array.flatten()
                action_proie = agent_proie.get_action(observation_proie_plate, nom_proie)
                actions_a_faire[nom_proie] = int(action_proie)

            observations_suivantes, recompenses_dict, terminations_dict, truncations_dict, _ = environnement.step(actions_a_faire)

            # Store transitions
            for nom_predateur in noms_predateurs:
                predateur_etait_present = nom_predateur in observations_dictionnaire
                if not predateur_etait_present:
                    continue

                recompense_obtenue = recompenses_dict.get(nom_predateur, 0.0)
                recompense_float = float(recompense_obtenue)
                est_termine = terminations_dict.get(nom_predateur, False)
                est_tronque = truncations_dict.get(nom_predateur, False)
                episode_fini = bool(est_termine or est_tronque)
                trajectoires[nom_predateur]['rew'].append(recompense_float)
                trajectoires[nom_predateur]['done'].append(episode_fini)

            observations_dictionnaire = observations_suivantes
            compteur_pas += 1
            compteur_rollout += 1

            pas_total_atteint = compteur_pas >= configuration.nombre_total_pas
            if pas_total_atteint:
                break

        # calculs des avantages et retours
        toutes_observations = []
        toutes_actions = []
        tous_anciens_logp = []
        tous_avantages = []
        tous_retours = []

        for nom_predateur in noms_predateurs:
            longueur_trajectoire = len(trajectoires[nom_predateur]['rew'])
            trajectoire_vide = longueur_trajectoire == 0
            if trajectoire_vide:
                continue

            recompenses_array = np.asarray(trajectoires[nom_predateur]['rew'], dtype=np.float32)
            termines_array = np.asarray(trajectoires[nom_predateur]['done'], dtype=np.float32)
            valeurs_array = np.asarray(trajectoires[nom_predateur]['val'], dtype=np.float32)
            valeur_finale = 0.0

            avantages_calcules, retours_calcules = calculer_avantages_generalises(
                recompenses_array,
                termines_array,
                valeurs_array,
                configuration.facteur_discount,
                configuration.lambda_gae,
                valeur_finale
            )

            observations_empilees = np.stack(trajectoires[nom_predateur]['obs'], axis=0)
            toutes_observations.append(observations_empilees)
            actions_array = np.asarray(trajectoires[nom_predateur]['act'], dtype=np.int64)
            toutes_actions.append(actions_array)
            logp_array = np.asarray(trajectoires[nom_predateur]['logp'], dtype=np.float32)
            tous_anciens_logp.append(logp_array)
            tous_avantages.append(avantages_calcules)
            tous_retours.append(retours_calcules)

        # on concatène les données pour faire les mises à jour
        observations_concatenees = np.concatenate(toutes_observations, axis=0)
        actions_concatenees = np.concatenate(toutes_actions, axis=0)
        anciens_logp_concatenes = np.concatenate(tous_anciens_logp, axis=0)
        avantages_concatenes = np.concatenate(tous_avantages, axis=0)
        retours_concatenes = np.concatenate(tous_retours, axis=0)

        tenseur_obs = torch.as_tensor(observations_concatenees, dtype=torch.float32, device=appareil)
        tenseur_actions = torch.as_tensor(actions_concatenees, dtype=torch.int64, device=appareil)
        tenseur_anciens_logp = torch.as_tensor(anciens_logp_concatenes, dtype=torch.float32, device=appareil)
        tenseur_avantages = torch.as_tensor(avantages_concatenes, dtype=torch.float32, device=appareil)
        tenseur_retours = torch.as_tensor(retours_concatenes, dtype=torch.float32, device=appareil)

        taille_batch_total = tenseur_obs.shape[0]
        indices_batch = np.arange(taille_batch_total)

        # PPO mis a jour
        numero_epoque = 0
        while numero_epoque < configuration.epoques_mise_a_jour:
            np.random.shuffle(indices_batch)
            position_debut = 0
            while position_debut < taille_batch_total:
                position_fin = position_debut + configuration.taille_mini_batch
                indices_minibatch = indices_batch[position_debut:position_fin]

                observations_mb = tenseur_obs[indices_minibatch]
                actions_mb = tenseur_actions[indices_minibatch]
                anciens_logp_mb = tenseur_anciens_logp[indices_minibatch]
                avantages_mb = tenseur_avantages[indices_minibatch]
                retours_mb = tenseur_retours[indices_minibatch]

                # Forward
                logits_politique = agent_predateur.reseau_politique(observations_mb)
                distribution_actions = Categorical(logits=logits_politique)
                nouveaux_logp = distribution_actions.log_prob(actions_mb)
                entropie_moyenne = distribution_actions.entropy()
                entropie_moyenne = entropie_moyenne.mean()
                nouvelles_valeurs = agent_predateur.reseau_valeur(observations_mb)
                difference_logp = nouveaux_logp - anciens_logp_mb
                ratio_importance = difference_logp.exp()

                # Losses
                terme_surrogate_1 = ratio_importance * avantages_mb
                ratio_clippe = torch.clamp(ratio_importance, 1.0 - configuration.coefficient_clip, 1.0 + configuration.coefficient_clip)
                terme_surrogate_2 = ratio_clippe * avantages_mb
                minimum_surrogate = torch.min(terme_surrogate_1, terme_surrogate_2)
                perte_politique = -minimum_surrogate.mean()
                difference_valeur = nouvelles_valeurs - retours_mb
                perte_valeur = difference_valeur.pow(2)
                perte_valeur = perte_valeur.mean()
                perte_totale = perte_politique + configuration.coefficient_valeur * perte_valeur - configuration.coefficient_entropie * entropie_moyenne

                # Backward
                agent_predateur.optimiseur.zero_grad()
                perte_totale.backward()
                agent_predateur.optimiseur.step()

                position_debut = position_debut + configuration.taille_mini_batch

            numero_epoque = numero_epoque + 1

        # regarde si la perf est meilleure, si oui on save le modèle
        liste_toutes_recompenses = []
        for nom_predateur in noms_predateurs:
            liste_recompenses_predateur = trajectoires[nom_predateur]['rew']
            for recompense in liste_recompenses_predateur:
                liste_toutes_recompenses.append(recompense)

        au_moins_une_recompense = False
        for nom_predateur in noms_predateurs:
            longueur = len(trajectoires[nom_predateur]['rew'])
            if longueur > 0:
                au_moins_une_recompense = True
                break

        if au_moins_une_recompense:
            recompense_moyenne = float(np.mean(liste_toutes_recompenses))
        else:
            recompense_moyenne = 0.0

        performance_amelioree = recompense_moyenne > meilleure_recompense_moyenne
        if performance_amelioree:
            meilleure_recompense_moyenne = recompense_moyenne
            fichier_actuel = os.path.abspath(__file__)
            dossier_actuel = os.path.dirname(fichier_actuel)
            chemin_meilleur_modele = os.path.join(dossier_actuel, "predator_model_best.pth")
            dictionnaire_parametres = agent_predateur.reseau_politique.state_dict()
            torch.save(dictionnaire_parametres, chemin_meilleur_modele)

        print(f"steps={compteur_pas:>7,}  reward={recompense_moyenne:>8.3f}  best={meilleure_recompense_moyenne:>8.3f}")

        pas_total_atteint = compteur_pas >= configuration.nombre_total_pas

    fichier_actuel = os.path.abspath(__file__)
    dossier_actuel = os.path.dirname(fichier_actuel)
    chemin_sauvegarde = os.path.join(dossier_actuel, "predator_model.pth")
    dictionnaire_parametres_final = agent_predateur.reseau_politique.state_dict()
    torch.save(dictionnaire_parametres_final, chemin_sauvegarde)
    print("\n")
    print("======================")
    print("entrainement terminé")
    print("======================")


if __name__ == "__main__":
    entrainer()
