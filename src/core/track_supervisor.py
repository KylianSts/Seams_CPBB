"""
track_supervisor.py
-------------------
Module de validation métier.
Examine les résultats du tracker générique et corrige les associations (ID Switches)
qui violent les règles de la cinématique ou l'apparence des équipes (GMM).
"""

import logging
import math
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def project_to_court(foot_x: float, foot_y: float, H_matrix: np.ndarray):
    """Utilitaire local pour la Kinematic Gate : projette le pixel en mètres."""
    if H_matrix is None:
        return None
    pt = np.array([[[float(foot_x), float(foot_y)]]], dtype=np.float32)
    try:
        res = cv2.perspectiveTransform(pt, H_matrix)
        return (float(res[0, 0, 0]), float(res[0, 0, 1]))
    except Exception:
        return None


class TrackSupervisor:
    def __init__(self, gmm_veto_threshold: float = 0.65, max_jump_m: float = 1.5):
        """
        Args:
            gmm_veto_threshold: Confiance minimale (ex: 65%) requise par le GMM 
                                pour autoriser un échange d'ID (Swap).
            max_jump_m: Déplacement maximum autorisé en mètres entre 2 frames (Kinematic Gate).
                        1.5m en 1/30e sec = 45 m/s (très large, empêche les téléportations).
        """
        self.gmm_veto_threshold = gmm_veto_threshold
        self.max_jump_m = max_jump_m

    def apply_team_veto(self, tracked_objects, frame: np.ndarray, state, team_detector) -> list:
        """
        Applique la "Kinematic Gate" et le "Swap Post-Occlusion" sur les détections de BotSort.
        """
        if not team_detector.is_calibrated or team_detector.gmm is None:
            return tracked_objects

        # =======================================================================
        # 1. EXTRACTION ET PRÉPARATION DES DONNÉES DE LA FRAME
        # =======================================================================
        tracks_data = []
        for t in tracked_objects:
            x1, y1, x2, y2 = t[:4]
            track_id = int(t[4])
            
            # --- Évaluation GMM ---
            predicted_team = None
            confidence = 0.0
            hist = team_detector._get_torso_histogram(frame, (x1, y1, x2, y2))
            if hist is not None:
                probs = team_detector.gmm.predict_proba([hist])[0]
                predicted_team = int(np.argmax(probs))
                confidence = probs[predicted_team]
            
            # --- Évaluation Spatiale ---
            foot_x, foot_y = (x1 + x2) / 2.0, float(y2)
            court_pos = project_to_court(foot_x, foot_y, state.camera.H_matrix)

            tracks_data.append({
                'raw': t.copy(),  # On copie pour pouvoir inverser les IDs en toute sécurité
                'id': track_id,
                'box': (x1, y1, x2, y2),
                'pred_team': predicted_team,
                'conf': confidence,
                'court_pos': court_pos,
                'height': y2 - y1
            })

        # =======================================================================
        # 2. KINEMATIC GATE (La barrière physique anti-téléportation)
        # =======================================================================
        valid_tracks = []
        for td in tracks_data:
            tid = td['id']
            
            # Si le joueur est connu et que l'on peut comparer sa position
            if tid in state.players and state.players[tid].court_pos_m is not None and td['court_pos'] is not None:
                old_pos = state.players[tid].court_pos_m
                new_pos = td['court_pos']
                jump_dist = math.hypot(new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
                
                # Si le joueur se "téléporte" à l'autre bout du terrain, c'est un faux positif de BotSort
                if jump_dist > self.max_jump_m:
                    logger.debug(f"[KINEMATIC GATE] ID {tid} rejeté (Saut impossible de {jump_dist:.1f}m).")
                    continue # On le rejette. Le tracker fera du Gap Filling sur son ancienne position.
            
            valid_tracks.append(td)

        # =======================================================================
        # 3. LE SWAP POST-OCCLUSION (L'Arbitre des croisements)
        # =======================================================================
        n = len(valid_tracks)
        swapped_indices = set()

        for i in range(n):
            if i in swapped_indices: continue
            
            td1 = valid_tracks[i]
            id1 = td1['id']
            
            # On vérifie si ce joueur a un historique d'équipe
            if id1 not in state.players or state.players[id1].team_id is None:
                continue
            hist_team1 = state.players[id1].team_id

            for j in range(i + 1, n):
                if j in swapped_indices: continue
                
                td2 = valid_tracks[j]
                id2 = td2['id']

                if id2 not in state.players or state.players[id2].team_id is None:
                    continue
                hist_team2 = state.players[id2].team_id

                # RÈGLE 1 : On ne s'intéresse qu'aux switchs entre équipes adverses
                if hist_team1 == hist_team2:
                    continue
                
                # RÈGLE 2 : Les deux boîtes doivent être proches (Zone de croisement)
                # On utilise la hauteur des joueurs comme seuil de distance dynamique
                cx1, cy1 = (td1['box'][0]+td1['box'][2])/2, (td1['box'][1]+td1['box'][3])/2
                cx2, cy2 = (td2['box'][0]+td2['box'][2])/2, (td2['box'][1]+td2['box'][3])/2
                pixel_dist = math.hypot(cx1-cx2, cy1-cy2)
                
                max_dist_for_swap = max(td1['height'], td2['height']) * 1.5
                if pixel_dist > max_dist_for_swap:
                    continue # Trop loin l'un de l'autre pour s'être croisés
                
                # RÈGLE 3 : Le GMM est-il convaincu qu'ils sont inversés ?
                # La boîte 1 ressemble à l'équipe 2 ET la boîte 2 ressemble à l'équipe 1
                cond_swap1 = (td1['pred_team'] == hist_team2) and (td1['conf'] >= self.gmm_veto_threshold)
                cond_swap2 = (td2['pred_team'] == hist_team1) and (td2['conf'] >= self.gmm_veto_threshold)

                if cond_swap1 and cond_swap2:
                    logger.info(f"🔄 [ID SWAP] Crossover corrigé : Échange de {id1} et {id2}.")
                    
                    # On échange mathématiquement les IDs dans les données brutes (index 4)
                    td1['raw'][4], td2['raw'][4] = id2, id1
                    
                    # On marque ces index pour ne pas les ré-évaluer
                    swapped_indices.add(i)
                    swapped_indices.add(j)
                    break # On passe au joueur (i) suivant

        # =======================================================================
        # 4. RECONSTRUCTION DE LA SORTIE
        # =======================================================================
        final_tracks = [td['raw'] for td in valid_tracks]
        return final_tracks