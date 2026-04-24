"""
track_supervisor.py
-------------------
Module de validation métier.
Examine les résultats du tracker générique et annule les associations
qui violent les règles de la physique ou les probabilités du GMM.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

class TrackSupervisor:
    def __init__(self, gmm_veto_threshold: float = 0.70):
        """
        Args:
            gmm_veto_threshold: Confiance minimale (0.0 à 1.0) que le GMM 
            doit avoir pour oser poser un Veto contre BotSort.
        """
        self.gmm_veto_threshold = gmm_veto_threshold

    def apply_team_veto(self, tracked_objects, frame: np.ndarray, state, team_detector) -> list:
        """
        Annule un ID Switch si la boîte assignée par BotSort a une couleur d'équipe
        qui contredit avec certitude l'historique du joueur.
        """
        # Si le GMM n'est pas encore calibré (Pre-flight), on ne fait rien
        if not team_detector.is_calibrated or team_detector.gmm is None:
            return tracked_objects

        valid_tracks = []

        for t in tracked_objects:
            x1, y1, x2, y2 = t[:4]
            track_id = int(t[4])

            # 1. Le joueur existe-t-il déjà dans notre mémoire avec une équipe fixe ?
            if track_id in state.players and state.players[track_id].team_id is not None:
                historical_team = state.players[track_id].team_id

                # 2. On extrait les couleurs du torse de la boîte que BotSort vient de lui donner
                hist = team_detector._get_torso_histogram(frame, (x1, y1, x2, y2))
                
                if hist is not None:
                    # Probabilités du GMM (ex: [0.95, 0.05])
                    probs = team_detector.gmm.predict_proba([hist])[0]
                    predicted_team = int(np.argmax(probs))
                    confidence = probs[predicted_team]

                    # 3. LE VETO : Le GMM est sûr que c'est un adversaire
                    if predicted_team != historical_team and confidence >= self.gmm_veto_threshold:
                        logger.debug(f"ID Switch évité ! Veto posé sur l'ID {track_id}.")
                        # On NE l'ajoute PAS aux valid_tracks.
                        # Le joueur passera automatiquement en "Freeze" dans le Gap Filling
                        continue 

            # Si tout est normal, on valide la détection
            valid_tracks.append(t)

        return valid_tracks