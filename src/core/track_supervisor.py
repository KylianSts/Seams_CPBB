"""
track_supervisor.py
-------------------
Module de validation métier (Le Juge Temporel).
Applique un algorithme de Soft Voting Continu sur une fenêtre bidirectionnelle.
Pondère les probabilités GMM par le niveau d'occlusion physique du joueur.
"""

import logging

logger = logging.getLogger(__name__)

class TrackSupervisor:
    def __init__(self, gmm_veto_threshold: float = 0.65):
        pass

    def apply_team_veto(self, tracked_objects, frame, state, team_detector) -> list:
        # Fonction maintenue vide pour la compatibilité avec l'Étape 3 de run_pipeline
        return tracked_objects

    def resolve_team_color(self, player, target_frame_idx: int, window: int = 15) -> int:
        """
        LE JUGE TEMPOREL (Soft Voting Bidirectionnel).
        Calcule l'équipe d'un joueur en pondérant chaque preuve par sa certitude et son niveau d'occlusion.
        """
        if not hasattr(player, 'gmm_history') or not player.gmm_history:
            return player.team_id if player.team_id is not None else 0

        score_A = 0.0
        score_B = 0.0

        # EXAMEN DU CASIER DE PREUVES
        for frame_idx, prob_A, prob_B, occlusion_ratio in player.gmm_history:
            
            # On ne prend en compte que les preuves dans la fenêtre [-15, +15]
            if abs(frame_idx - target_frame_idx) <= window:
                
                # 1. Le Bourreau d'Occlusion : (1 - O)^3
                # Si occlusion = 0.0 -> Facteur 1.0 (Pleine puissance)
                # Si occlusion = 0.5 -> Facteur 0.125 (Preuve presque ignorée)
                occlusion_factor = (1.0 - occlusion_ratio) ** 3
                
                # 2. L'Amplificateur de Certitude : P^2
                # Le poids final est la probabilité au carré multipliée par le facteur d'occlusion
                weight_A = (prob_A ** 2) * occlusion_factor
                weight_B = (prob_B ** 2) * occlusion_factor

                score_A += weight_A
                score_B += weight_B

        # S'il n'y a eu aucune preuve valide dans la fenêtre (cas rarissime)
        if score_A == 0.0 and score_B == 0.0:
            return player.team_id if player.team_id is not None else 0

        # LE VERDICT
        # L'équipe gagnante est celle qui a accumulé la plus grande "masse" de probabilités pures
        return 0 if score_A > score_B else 1