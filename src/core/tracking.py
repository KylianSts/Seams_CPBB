"""
tracking.py
-----------
Module de suivi (Tracking) avec Gap Filling.
Prend les détections brutes de joueurs, utilise BotSort pour leur assigner
un ID unique, projette leur position sur le terrain FIBA, et comble les
disparitions temporaires (occlusion) via un buffer interne.
"""

import logging
from pathlib import Path
from typing import List, Tuple
from collections import deque 

import cv2
import numpy as np
import torch

from core.state import MatchState, PlayerState
from core.filters import apply_ema_2d
from core.track_supervisor import TrackSupervisor
from boxmot.trackers.botsort.botsort import BotSort

logger = logging.getLogger(__name__)


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_tracker(device: int = 0) -> BotSort:
    """
    Initialise le tracker BotSort en mémoire.
    À appeler une seule fois au démarrage dans run_pipeline.py.
    """
    logger.info("Chargement du tracker BotSort (ReID)...")

    reid_weights = Path("models/weights/osnet_x0_25_msmt17.pt")
    torch_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    tracker = BotSort(
        reid_weights=reid_weights,
        device=torch_device,
        half=False,
        track_high_thresh=0.45,
        track_low_thresh=0.15,
        new_track_thresh=0.55,
        track_buffer=60,
        match_thresh=0.80
    )

    # Buffer personnalisé pour le Gap Filling
    # Structure : { track_id: {"player": PlayerState, "lost_frames": int} }
    tracker.custom_lost_buffer = {}

    logger.info("Tracker prêt.")
    return tracker


# ===========================================================================
# UTILITAIRE DE PROJECTION
# ===========================================================================

def project_to_court(foot_x: float, foot_y: float, H_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Transforme des coordonnées pixels (image) en mètres (terrain FIBA)
    via la matrice d'homographie. Retourne None si H est invalide.
    """
    if H_matrix is None:
        return None

    point_px = np.array([[[foot_x, foot_y]]], dtype=np.float32)

    try:
        point_m = cv2.perspectiveTransform(point_px, H_matrix)
        return (float(point_m[0, 0, 0]), float(point_m[0, 0, 1]))
    except Exception as e:
        logger.debug(f"Erreur de projection : {e}")
        return None


# ===========================================================================
# MISE À JOUR DE L'ÉTAT (Par Frame)
# ===========================================================================

def update_players_tracking(
    tracker: BotSort,
    player_detections: List[Tuple[float, float, float, float, float]],
    frame: np.ndarray,
    state: MatchState,
    team_detector = None,       # NOUVEAU : Référence au modèle GMM
    supervisor = None,          # NOUVEAU : Instance du TrackSupervisor
    max_lost_frames: int = 20   # ~200ms à 30fps
) -> MatchState:
    """
    Met à jour state.players avec les IDs et positions des joueurs.
    Inclut le Gap Filling : un joueur perdu depuis moins de max_lost_frames
    est conservé à sa dernière position connue (fantôme).
    Conserve également l'historique cinématique des joueurs.
    """
    # 1. Formatage pour BotSort : (N, 6) → [x1, y1, x2, y2, conf, class_id]
    if player_detections:
        dets_array = np.array(
            [[d[0], d[1], d[2], d[3], d[4], 0.0] for d in player_detections],
            dtype=np.float32
        )
    else:
        dets_array = np.empty((0, 6), dtype=np.float32)

    # 2. Mise à jour BotSort brute
    tracked_objects = tracker.update(dets_array, frame)

    # =======================================================================
    # NOUVEAU : SUPERVISION (Le Veto GMM)
    # On filtre les résultats de BotSort avant de les injecter dans le State
    # =======================================================================
    if supervisor is not None and team_detector is not None and tracked_objects is not None:
        tracked_objects = supervisor.apply_team_veto(tracked_objects, frame, state, team_detector)
    # =======================================================================

    new_players_dict = {}
    active_ids = set()

    # 3. Joueurs DÉTECTÉS cette frame
    if tracked_objects is not None and len(tracked_objects) > 0:
        for t in tracked_objects:
            x1, y1, x2, y2 = t[:4]
            track_id = int(t[4])

            if track_id < 0:
                continue

            active_ids.add(track_id)

            foot_x = (x1 + x2) / 2.0
            foot_y = float(y2)
            raw_court_pos = project_to_court(foot_x, foot_y, state.camera.H_matrix)

            # --- RÉCUPÉRATION DE LA MÉMOIRE ---
            if track_id in state.players:
                old_history = state.players[track_id].pos_history_m
                old_speed = state.players[track_id].speed_kmh
                old_court_pos = state.players[track_id].court_pos_m 
            else:
                old_history = deque(maxlen=15)
                old_speed = 0.0
                old_court_pos = None 
            # ------------------------------------------------

            # --- LISSAGE EMA (Exponential Moving Average) ---
            if raw_court_pos is not None:
                if old_court_pos is not None:
                    # Formule du lissage
                    ALPHA = 0.10  
                    smooth_x = (ALPHA * raw_court_pos[0]) + ((1.0 - ALPHA) * old_court_pos[0])
                    smooth_y = (ALPHA * raw_court_pos[1]) + ((1.0 - ALPHA) * old_court_pos[1])
                    final_court_pos = (smooth_x, smooth_y)
                else:
                    # Première frame où on voit le joueur : pas de lissage possible
                    final_court_pos = raw_court_pos
            else:
                final_court_pos = None
            # ------------------------------------------------

            player = PlayerState(
                track_id=track_id,
                bbox_px=(float(x1), float(y1), float(x2), float(y2)),
                foot_pos_px=(foot_x, foot_y),
                court_pos_m=final_court_pos,
                pos_history_m=old_history,  
                speed_kmh=old_speed         
            )

            new_players_dict[track_id] = player

            # Mise à jour du buffer (joueur visible → lost_frames remis à 0)
            tracker.custom_lost_buffer[track_id] = {
                "player": player,
                "lost_frames": 0
            }

    # 4. Joueurs PERDUS → Gap Filling
    ids_to_remove = []

    for tid, data in tracker.custom_lost_buffer.items():
        if tid in active_ids:
            continue  # Déjà traité ci-dessus

        data["lost_frames"] += 1

        if data["lost_frames"] <= max_lost_frames:
            # On réinjecte le joueur à sa dernière position connue (fantôme)
            new_players_dict[tid] = data["player"]
        else:
            ids_to_remove.append(tid)

    for tid in ids_to_remove:
        del tracker.custom_lost_buffer[tid]

    # 5. Mise à jour de l'état global
    state.players = new_players_dict

    return state