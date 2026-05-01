"""
tracking.py
-----------
Module de suivi spatial (Tracking) et de ré-identification (ReID).
Gère l'association temporelle des Bounding Boxes (BotSort), la projection 
orthographique sur le terrain (Homographie), et le comblement des 
disparitions courtes (Gap Filling).
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch
from boxmot.trackers.botsort.botsort import BotSort

from core.state import MatchState, PlayerState

if TYPE_CHECKING:
    # Typage conditionnel pour éviter les imports circulaires si nécessaire à l'avenir
    pass

logger = logging.getLogger(__name__)

# --- Alias de type ---
BBox = Tuple[float, float, float, float]
Detection = Tuple[float, float, float, float, float]


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_tracker(device: int = 0) -> BotSort:
    """
    Initialise le tracker BotSort (avec extracteur de features OSNet).
    Configure un buffer personnalisé pour le Gap Filling métier.
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

    # Buffer mémoire métier pour pallier les occlusions sévères (ex: poteau, arbitre)
    # Format : { track_id: {"player": PlayerState, "lost_frames": int} }
    tracker.custom_lost_buffer = {}

    logger.info("Tracker prêt.")
    return tracker


# ===========================================================================
# GÉOMÉTRIE ET PROJECTION
# ===========================================================================

def project_to_court(foot_x: float, foot_y: float, H_matrix: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Transforme une coordonnée pixel (Image) en coordonnée métrique (Terrain FIBA).
    Retourne None si la matrice d'homographie est invalide ou absente.
    """
    if H_matrix is None:
        return None

    point_px = np.array([[[foot_x, foot_y]]], dtype=np.float32)

    try:
        point_m = cv2.perspectiveTransform(point_px, H_matrix)
        return (float(point_m[0, 0, 0]), float(point_m[0, 0, 1]))
    except Exception as e:
        logger.debug(f"Échec de la projection homographique : {e}")
        return None


# ===========================================================================
# MOTEUR DE TRACKING (FRAME PAR FRAME)
# ===========================================================================

def update_players_tracking(
    tracker: BotSort,
    player_detections: List[Detection],
    frame: np.ndarray,
    state: MatchState,
    current_t: float,
    max_lost_frames: int = 20
) -> MatchState:
    """
    Associe les détections de la frame courante aux trajectoires existantes.
    Maintient les joueurs disparus temporairement via le Gap Filling.
    """
    # 1. Formatage des entrées pour BotSort : [x1, y1, x2, y2, conf, class_id]
    if player_detections:
        dets_array = np.array(
            [[d[0], d[1], d[2], d[3], d[4], 0.0] for d in player_detections],
            dtype=np.float32
        )
    else:
        dets_array = np.empty((0, 6), dtype=np.float32)

    # 2. Inférence de l'association (Kalman + OSNet)
    tracked_objects = tracker.update(dets_array, frame)

    new_players_dict: Dict[int, PlayerState] = {}
    active_ids = set()

    # 3. Traitement des joueurs activement détectés
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

            # --- Transfert de la mémoire (État T-1 vers État T) ---
            if track_id in state.players:
                old_player = state.players[track_id]
                pos_history_m = old_player.pos_history_m
                raw_history = old_player.raw_history
                speed_kmh = old_player.speed_kmh
                
                # Héritage des attributs gérés par les autres modules
                team_id = old_player.team_id
                gmm_history = old_player.gmm_history
            else:
                pos_history_m = deque(maxlen=30)
                raw_history = deque(maxlen=40)
                speed_kmh = 0.0
                team_id = None
                gmm_history = deque(maxlen=80)

            # Enregistrement pour le lissage bidirectionnel (Look-Ahead)
            if raw_court_pos is not None:
                raw_history.append((state.frame_idx, raw_court_pos[0], raw_court_pos[1]))

            # Instanciation stricte du nouvel état du joueur
            player = PlayerState(
                track_id=track_id,
                bbox_px=(float(x1), float(y1), float(x2), float(y2)),
                foot_pos_px=(foot_x, foot_y),
                court_pos_m=raw_court_pos,
                team_id=team_id,
                pos_history_m=pos_history_m,
                raw_history=raw_history,
                speed_kmh=speed_kmh,
                gmm_history=gmm_history,
                is_lost=False
            )

            new_players_dict[track_id] = player

            # Remise à zéro du compteur d'occlusion dans le buffer
            tracker.custom_lost_buffer[track_id] = {
                "player": player,
                "lost_frames": 0
            }

    # 4. Traitement des disparitions (Gap Filling)
    ids_to_remove = []

    for tid, data in tracker.custom_lost_buffer.items():
        if tid in active_ids:
            continue

        data["lost_frames"] += 1

        if data["lost_frames"] <= max_lost_frames:
            # Le joueur est "Fantôme" : on maintient sa dernière position connue
            ghost_player = data["player"]
            ghost_player.is_lost = True
            new_players_dict[tid] = ghost_player
        else:
            ids_to_remove.append(tid)

    # Nettoyage du buffer pour les disparitions définitives
    for tid in ids_to_remove:
        del tracker.custom_lost_buffer[tid]

    # 5. Validation de la nouvelle topologie de la frame
    state.players = new_players_dict

    return state