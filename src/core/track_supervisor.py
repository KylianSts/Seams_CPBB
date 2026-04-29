"""
track_supervisor.py
-------------------
Module de validation métier (Le Juge Temporel).
Examine l'historique bidirectionnel (gmm_history) pour classifier
l'équipe de chaque joueur de manière parfaite en ignorant les occlusions.
"""

import logging

logger = logging.getLogger(__name__)

class TrackSupervisor:
    def __init__(self, gmm_veto_threshold: float = 0.65):
        # Le superviseur n'a plus besoin d'état interne ni de mémoire de mapping,
        # car toutes les preuves temporelles sont stockées directement dans les joueurs.
        pass

    def apply_team_veto(self, tracked_objects, frame, state, team_detector) -> list:
        """
        Ancienne méthode de correction d'ID maintenue (mais vidée de son code).
        POURQUOI ? Pour éviter de faire crasher ton run_pipeline.py à l'Étape 3
        avant que l'on passe à l'Étape 4. Elle ne fait plus rien.
        """
        return tracked_objects

    def resolve_team_color(self, player, target_frame_idx: int, window: int = 15) -> int:
        """
        LE JUGE TEMPOREL.
        Calcule l'équipe d'un joueur à l'instant cible (T-15) en utilisant les preuves 
        du passé (T-30 à T-16) et du futur (T-14 à T).
        """
        # 1. LA SENTENCE EST IRRÉVOCABLE (Hystérésis)
        # Si le joueur a déjà été certifié, on renvoie son équipe sans aucun calcul.
        if getattr(player, 'is_team_locked', False) and player.locked_team_id is not None:
            return player.locked_team_id

        # Sécurité : Si le joueur vient d'apparaître et n'a pas encore de casier
        if not hasattr(player, 'gmm_history') or not player.gmm_history:
            return player.team_id if player.team_id is not None else 0

        past_A, past_B = [], []
        future_A, future_B = [], []

        # 2. EXAMEN DU CASIER DE PREUVES
        for frame_idx, prob_A, prob_B, is_isolated in player.gmm_history:
            
            # 🚫 RÈGLE D'OR : On ignore complètement les probabilités 
            # des frames où le joueur était partiellement caché !
            if not is_isolated:
                continue
            
            dist = frame_idx - target_frame_idx
            
            # Tri des preuves pures entre le passé et le futur de la frame cible
            if -window <= dist <= 0:
                past_A.append(prob_A)
                past_B.append(prob_B)
            elif 0 < dist <= window:
                future_A.append(prob_A)
                future_B.append(prob_B)

        all_A = past_A + future_A
        all_B = past_B + future_B

        # Si le joueur a été occlusé pendant TOUTE la fenêtre de 30 frames
        # (ex: énorme mêlée sous le panier), on utilise sa dernière couleur connue.
        if not all_A:
            return player.team_id if player.team_id is not None else 0

        # 3. LE VOTE (Moyenne Bidirectionnelle Pure)
        mean_A = sum(all_A) / len(all_A)
        mean_B = sum(all_B) / len(all_B)
        
        final_team = 0 if mean_A > mean_B else 1
        confidence = max(mean_A, mean_B)

        # 4. DÉTECTION DU SHIFT (Le détecteur de mensonge de BotSort)
        # On vérifie qu'on a assez de données des deux côtés pour faire une comparaison
        if len(past_A) > 3 and len(future_A) > 3:
            past_mean_A = sum(past_A) / len(past_A)
            fut_mean_A = sum(future_A) / len(future_A)
            
            # Si l'identité visuelle s'inverse violemment (> 60% de différence)
            # Exemple : 90% Rouge dans le passé, 15% Rouge dans le futur.
            if abs(past_mean_A - fut_mean_A) > 0.60:
                logger.warning(f"⚠️ [SHIFT DÉTECTÉ] ID {player.track_id} : BotSort a inversé deux joueurs.")
                # C'est un ID corrompu. On refuse de le verrouiller ! 
                # On retournera juste l'équipe majoritaire autour de cette frame précise.
                return final_team

        # 5. VERROUILLAGE (Hystérésis)
        # Si le joueur a accumulé plus de 45 frames pures depuis son entrée sur le terrain,
        # et que sa certitude est supérieure à 85%.
        if getattr(player, 'pure_frames_count', 0) > 45 and confidence > 0.85:
            player.is_team_locked = True
            player.locked_team_id = final_team
            logger.info(f"🔒 [VERROU] ID:{player.track_id} certifié Équipe {final_team} (Confiance: {confidence*100:.1f}%)")

        return final_team