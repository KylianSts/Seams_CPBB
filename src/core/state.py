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
import copy

@dataclass
class FrameSnapshot:
    """
    Capture 'congelée' de l'état à la frame T.
    Utilisée par le Look-Ahead Buffer pour dessiner le passé sans subir
    les modifications du futur.
    """
    frame_idx: int
    frame_bgr: np.ndarray  # L'image brute (doit être un .copy() strict)
    
    # --- Detections & Tracking ---
    players: dict          # Copie profonde pour figer les coordonnées et les IDs
    ball_bbox_px: Optional[Tuple[float, float, float, float]]
    hoop_bbox_px: Optional[Tuple[float, float, float, float]]

    court_keypoints_px: Optional[np.ndarray]
    court_keypoints_conf: Optional[np.ndarray]
    
    # --- Masques ---
    player_masks: list
    net_mask: Optional[np.ndarray]
    
    # --- UI & Triggers ---
    active_triggers: dict
    shot_scores: dict
    
    # --- Réalité Augmentée ---
    ar_alpha_multiplier: float
    camera_matrix: Optional[np.ndarray]
    camera_stable: bool
    
    # --- Audio ---
    is_whistle_active: bool
    is_crowd_active: bool
    
    # --- Stats Sidebar (Cinématique & Tactique) ---
    team_metrics: list
    avg_speed_kmh: float
    std_speed_kmh: float
    min_speed_kmh: float
    max_speed_kmh: float
    avg_accel_ms2: float
    std_accel_ms2: float
    min_accel_ms2: float
    max_accel_ms2: float
    
    target_hoop: Optional[Tuple[float, float]]
    attacking_team_id: Optional[int]

    optical_flow_history: list
    net_area_history: list

    is_perfect_shot: bool = False

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

    # --- NOUVEAU : Classification d'Équipe (GMM Bidirectionnel) ---
    # Stocke les preuves : (frame_idx, proba_A, proba_B, is_isolated)
    gmm_history: deque = field(default_factory=lambda: deque(maxlen=80))
    is_team_locked: bool = False
    locked_team_id: Optional[int] = None
    pure_frames_count: int = 0

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

    ar_stable_frames: int = 0          # Compteur de frames depuis que la cam est stable
    ar_alpha_multiplier: float = 0.0   # Multiplicateur d'opacité (0.0 = invisible, 1.0 = 100%)

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

    def take_snapshot(self, current_frame: np.ndarray) -> FrameSnapshot:
        """
        Crée une copie déconnectée de la mémoire pour le rendu différé.
        Crucial : On utilise copy() et deepcopy() pour empêcher le buffer
        d'être modifié par les frames futures.
        """
        return FrameSnapshot(
            frame_idx=self.frame_idx,
            frame_bgr=current_frame.copy(),  # TRÈS IMPORTANT : fige les pixels
            
            # deepcopy fige l'état de chaque objet Player (position, couleur, id)
            players=copy.deepcopy(self.players), 
            
            ball_bbox_px=self.ball_bbox_px,
            hoop_bbox_px=self.hoop_bbox_px,

            court_keypoints_px=self.court_keypoints_px.copy() if self.court_keypoints_px is not None else None,
            court_keypoints_conf=self.court_keypoints_conf.copy() if self.court_keypoints_conf is not None else None,
            
            # Sécurisation des masques numpy
            player_masks=[m.copy() if isinstance(m, np.ndarray) else m for m in self.player_masks],
            net_mask=self.net_mask.copy() if self.net_mask is not None else None,
            
            # Sécurisation des dictionnaires
            active_triggers=self.active_triggers.copy(),
            shot_scores=self.shot_scores.copy(),
            
            # Valeurs scalaires (copiées par valeur automatiquement)
            ar_alpha_multiplier=self.ar_alpha_multiplier,
            camera_matrix=self.camera.H_matrix.copy() if self.camera.H_matrix is not None else None,
            camera_stable=getattr(self.camera, 'is_stable_strict', getattr(self.camera, 'is_stable', False)),
            
            is_whistle_active=self.is_whistle_active,
            is_crowd_active=self.is_crowd_active,
            
            team_metrics=copy.deepcopy(self.team_metrics),
            avg_speed_kmh=self.avg_speed_kmh,
            std_speed_kmh=self.std_speed_kmh,
            min_speed_kmh=self.min_speed_kmh,
            max_speed_kmh=self.max_speed_kmh,
            avg_accel_ms2=self.avg_accel_ms2,
            std_accel_ms2=self.std_accel_ms2,
            min_accel_ms2=self.min_accel_ms2,
            max_accel_ms2=self.max_accel_ms2,
            
            target_hoop=self.target_hoop,
            attacking_team_id=self.attacking_team_id,

            optical_flow_history=list(self.optical_flow_history),
            net_area_history=list(self.net_area_history),
        )