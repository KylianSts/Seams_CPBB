"""
track_supervisor.py
-------------------
Module de validation métier.
Examine les résultats du tracker générique et corrige les associations (ID Switches)
qui violent l'apparence des équipes (GMM). Ne supprime AUCUNE boîte.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def boxes_intersect(b1, b2, margin=15):
    """
    Vérifie si deux BBoxes se chevauchent ou sont extrêmement proches.
    margin: tolérance en pixels pour capter la frame juste après la séparation.
    """
    return not (b1[2] < b2[0] - margin or 
                b1[0] > b2[2] + margin or 
                b1[3] < b2[1] - margin or 
                b1[1] > b2[3] + margin)


class TrackSupervisor:
    def __init__(self, gmm_veto_threshold: float = 0.65):
        """
        Args:
            gmm_veto_threshold: Confiance minimale (ex: 65%) requise par le GMM 
                                pour autoriser un échange d'ID.
        """
        self.gmm_veto_threshold = gmm_veto_threshold
        
        # Mémoire persistante des corrections du Superviseur
        # Dictionnaire { original_botSort_id : corrected_id }
        self.track_mapping = {}

    def apply_team_veto(self, tracked_objects, frame: np.ndarray, state, team_detector) -> list:
        if not team_detector.is_calibrated or team_detector.gmm is None:
            return tracked_objects

        # =======================================================================
        # 1. EXTRACTION ET TRADUCTION DES IDs (Application de la mémoire)
        # =======================================================================
        tracks_data = []
        for t in tracked_objects:
            raw_t = t.copy()
            orig_id = int(raw_t[4])
            
            # Application de la mémoire des Swaps précédents
            mapped_id = self.track_mapping.get(orig_id, orig_id)
            raw_t[4] = mapped_id 

            x1, y1, x2, y2 = raw_t[:4]
            
            # Évaluation GMM
            predicted_team = None
            confidence = 0.0
            hist = team_detector._get_torso_histogram(frame, (x1, y1, x2, y2))
            if hist is not None:
                probs = team_detector.gmm.predict_proba([hist])[0]
                predicted_team = int(np.argmax(probs))
                confidence = probs[predicted_team]

            tracks_data.append({
                'raw': raw_t,
                'orig_id': orig_id,
                'mapped_id': mapped_id, 
                'box': (x1, y1, x2, y2),
                'pred_team': predicted_team,
                'conf': confidence
            })

        # =======================================================================
        # 2. LE SWAP POST-OCCLUSION (L'Arbitre des couleurs)
        # =======================================================================
        n = len(tracks_data)
        swapped_indices = set()

        for i in range(n):
            if i in swapped_indices: continue
            
            td1 = tracks_data[i]
            id1 = td1['mapped_id']
            
            if id1 not in state.players or state.players[id1].team_id is None:
                continue
            hist_team1 = state.players[id1].team_id

            for j in range(i + 1, n):
                if j in swapped_indices: continue
                
                td2 = tracks_data[j]
                id2 = td2['mapped_id']

                if id2 not in state.players or state.players[id2].team_id is None:
                    continue
                hist_team2 = state.players[id2].team_id

                # RÈGLE 1 : Équipes adverses uniquement
                if hist_team1 == hist_team2:
                    continue
                
                # RÈGLE 2 : Proximité. Les boîtes DOIVENT se croiser (ou se frôler à 15px près)
                if not boxes_intersect(td1['box'], td2['box'], margin=15):
                    continue
                
                # RÈGLE 3 : Le GMM détecte une inversion avec certitude
                cond_swap1 = (td1['pred_team'] == hist_team2) and (td1['conf'] >= self.gmm_veto_threshold)
                cond_swap2 = (td2['pred_team'] == hist_team1) and (td2['conf'] >= self.gmm_veto_threshold)

                if cond_swap1 and cond_swap2:
                    logger.info(f"✅ [ID SWAP] Croisement corrigé : Échange de {id1} et {id2}.")
                    
                    # Mise à jour de la mémoire pour TOUTES les frames futures
                    orig1, orig2 = td1['orig_id'], td2['orig_id']
                    self.track_mapping[orig1] = id2
                    self.track_mapping[orig2] = id1
                    
                    # Correction immédiate des variables pour l'affichage de la frame courante
                    td1['mapped_id'], td2['mapped_id'] = id2, id1
                    td1['raw'][4], td2['raw'][4] = id2, id1
                    
                    swapped_indices.add(i)
                    swapped_indices.add(j)
                    break

        # =======================================================================
        # 3. RECONSTRUCTION (Aucune suppression)
        # =======================================================================
        # On renvoie TOUTES les boîtes, exactement comme BotSort les a fournies, 
        # avec uniquement l'ID (index 4) potentiellement modifié.
        final_tracks = [td['raw'] for td in tracks_data]
        return final_tracks