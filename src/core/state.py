"""
state.py
--------
Contient les structures de données (Dataclasses) qui représentent 
l'état du match à un instant T.
Agit comme la mémoire centrale ("Blackboard") partagée entre tous les modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np


@dataclass
class PlayerState:
    """État d'un joueur unique sur la frame courante."""
    track_id: int                                       # ID unique du joueur (BotSort)
    bbox_px: Tuple[float, float, float, float]          # [x1, y1, x2, y2] en pixels
    foot_pos_px: Tuple[float, float]                    # (x, y) des pieds en pixels
    court_pos_m: Optional[Tuple[float, float]] = None   # (X, Y) en mètres sur terrain FIBA
    team_id: Optional[int] = None
    has_ball: bool = False
    is_lost: bool = False
    closest_defender_dist_m: Optional[float] = None
    is_open: bool = False

    # --- Cinématique ---
    # File d'attente pour lisser la trajectoire (fenêtre glissante)
    pos_history_m: deque = field(default_factory=lambda: deque(maxlen=30))
    speed_kmh: float = 0.0
    accel_ms2: float = 0.0
    pos_filter: Any = None


@dataclass
class CameraState:
    """État de la caméra et de l'homographie."""
    H_matrix: Optional[np.ndarray] = None
    is_stable: bool = False
    stable_frames_count: int = 0
    kp_filter: Any = None


@dataclass
class MatchState:
    """Mémoire centrale du pipeline."""

    # ===========================================================================
    # 1. DONNÉES DE BASE
    # ===========================================================================
    frame_idx: int = 0
    camera: CameraState = field(default_factory=CameraState)
    players: Dict[int, PlayerState] = field(default_factory=dict)

    ball_bbox_px: Optional[Tuple[float, float, float, float]] = None
    hoop_bbox_px: Optional[Tuple[float, float, float, float]] = None

    player_masks: List[np.ndarray] = field(default_factory=list)
    net_mask: Optional[np.ndarray] = None

    events: List[str] = field(default_factory=list)

    # ===========================================================================
    # 2. HISTORIQUES (Pour la détection de tir)
    # ===========================================================================
    # ~1.3 sec à 30fps — suffisant pour check_geometric_crossing
    ball_history: deque = field(default_factory=lambda: deque(maxlen=40))

    # Fenêtre courte : on compare l'aire actuelle à la moyenne récente
    net_area_history: deque = field(default_factory=lambda: deque(maxlen=40))

    # Frame précédente en BGR pour get_hoop_optical_flow
    prev_frame_bgr: Optional[np.ndarray] = None

    optical_flow_history: deque = field(default_factory=lambda: deque(maxlen=40))

    # ===========================================================================
    # 4. KEYPOINTS DU TERRAIN (YOLO-Pose)
    # ===========================================================================
    court_keypoints_px: Optional[np.ndarray] = None    # Tableau (N, 2)
    court_keypoints_conf: Optional[np.ndarray] = None  # Tableau (N,) — pour filtrer l'affichage

    # ===========================================================================
    # 5. DASHBOARD & HUD (Pour render.py)
    # ===========================================================================
    active_triggers: dict = field(default_factory=dict)
    shot_scores: dict = field(default_factory=dict)
    is_whistle_active: bool = False
    is_crowd_active: bool = False

    # ===========================================================================
    # 6. GESTION DE L'OCCLUSION DE LA BALLE
    # ===========================================================================
    is_ball_near_hoop_sticky: bool = False
    last_ball_near_hoop_frame: int = -9999

    # ===========================================================================
    # 7. MÉTRIQUES GLOBALES (Cinématique)
    # ===========================================================================
    # 7. MÉTRIQUES GLOBALES (Cinématique)
    avg_speed_kmh: float = 0.0
    std_speed_kmh: float = 0.0
    min_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0

    avg_accel_ms2: float = 0.0
    std_accel_ms2: float = 0.0
    min_accel_ms2: float = 0.0
    max_accel_ms2: float = 0.0

    team_metrics: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        0: {
            "avg_speed": 0.0, "std_speed": 0.0, "min_speed": 0.0, "max_speed": 0.0,
            "avg_accel": 0.0, "std_accel": 0.0, "min_accel": 0.0, "max_accel": 0.0,
            "spacing": 0.0, "paint_count": 0.0
        },
        1: {
            "avg_speed": 0.0, "std_speed": 0.0, "min_speed": 0.0, "max_speed": 0.0,
            "avg_accel": 0.0, "std_accel": 0.0, "min_accel": 0.0, "max_accel": 0.0,
            "spacing": 0.0, "paint_count": 0.0
        }
    })

    # ===========================================================================
    # 8. TACTIQUE & POSSESSION
    # ===========================================================================
    attacking_team_id: Optional[int] = None
    target_hoop: Optional[Tuple[float, float]] = None