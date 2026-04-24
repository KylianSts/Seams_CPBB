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

    # --- Cinématique ---
    # File d'attente pour lisser la trajectoire (fenêtre glissante)
    pos_history_m: deque = field(default_factory=lambda: deque(maxlen=15))
    speed_kmh: float = 0.0


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
    net_area_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Frame précédente en BGR pour get_hoop_optical_flow
    prev_frame_bgr: Optional[np.ndarray] = None

    # ===========================================================================
    # 3. CALIBRATION DU PANIER (Pour check_hoop_deformation)
    # ===========================================================================
    hoop_dims_history: list = field(default_factory=list)
    is_hoop_calibrated: bool = False
    hoop_ref_w: Optional[float] = None
    hoop_ref_h: Optional[float] = None

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

    # ===========================================================================
    # 6. GESTION DE L'OCCLUSION DE LA BALLE
    # ===========================================================================
    is_ball_near_hoop_sticky: bool = False
    last_ball_near_hoop_frame: int = -9999

    # ===========================================================================
    # 7. MÉTRIQUES GLOBALES (Cinématique)
    # ===========================================================================
    avg_speed_kmh: float = 0.0
    std_speed_kmh: float = 0.0