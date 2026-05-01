"""
state.py
--------
Structures de données centrales (Blackboard) pour l'analyse du match.
Définit l'état immuable (Snapshot) et l'état mutatif (MatchState) utilisé
tout au long du pipeline de traitement vidéo.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque, TYPE_CHECKING
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from core.filters import OneEuroFilter

@dataclass
class FrameSnapshot:
    """
    Représentation immuable de l'état du match à une frame précise (T).
    Utilisé par le Look-Ahead Buffer pour le rendu différé sans risque de mutation.
    """
    frame_idx: int
    frame_bgr: np.ndarray 
    
    # --- Détections & Tracking ---
    players: Dict[int, 'PlayerState']
    ball_bbox_px: Optional[Tuple[float, float, float, float]]
    hoop_bbox_px: Optional[Tuple[float, float, float, float]]

    court_keypoints_px: Optional[np.ndarray]
    court_keypoints_conf: Optional[np.ndarray]
    
    # --- Masques ---
    player_masks: List[np.ndarray]
    net_mask: Optional[np.ndarray]
    
    # --- UI & Triggers ---
    active_triggers: Dict[str, bool]
    shot_scores: Dict[str, float]
    
    # --- Réalité Augmentée ---
    ar_alpha_multiplier: float
    camera_matrix: Optional[np.ndarray]
    camera_stable: bool
    
    # --- Audio ---
    is_whistle_active: bool
    is_crowd_active: bool
    
    # --- Analytique & Cinématique ---
    team_metrics: Dict[int, Dict[str, float]]
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

    optical_flow_history: List[float]
    net_area_history: List[float]

    is_perfect_shot: bool = False


@dataclass
class PlayerState:
    """État d'un joueur unique sur la frame courante."""
    track_id: int
    bbox_px: Tuple[float, float, float, float]
    foot_pos_px: Tuple[float, float]
    court_pos_m: Optional[Tuple[float, float]] = None
    team_id: Optional[int] = None
    has_ball: bool = False
    is_lost: bool = False
    closest_defender_dist_m: Optional[float] = None
    is_open: bool = False

    # --- Cinématique & Historique ---
    pos_history_m: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    raw_history: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=40))
    speed_kmh: float = 0.0
    accel_ms2: float = 0.0
    
    # Référence vers l'instance OneEuroFilter (Typage textuel pour éviter l'import circulaire)
    pos_filter: Optional['OneEuroFilter'] = None

    # --- Classification d'Équipe (GMM) ---
    # Structure : (frame_idx, proba_A, proba_B, occlusion_ratio)
    gmm_history: Deque[Tuple[int, float, float, float]] = field(default_factory=lambda: deque(maxlen=80))

    def clone(self) -> 'PlayerState':
        """
        Crée une copie profonde optimisée de l'état du joueur.
        Remplace avantageusement `copy.deepcopy` pour les performances de la boucle temps réel.
        """
        return PlayerState(
            track_id=self.track_id,
            bbox_px=self.bbox_px,
            foot_pos_px=self.foot_pos_px,
            court_pos_m=self.court_pos_m,
            team_id=self.team_id,
            has_ball=self.has_ball,
            is_lost=self.is_lost,
            closest_defender_dist_m=self.closest_defender_dist_m,
            is_open=self.is_open,
            pos_history_m=deque(self.pos_history_m, maxlen=self.pos_history_m.maxlen),
            raw_history=deque(self.raw_history, maxlen=self.raw_history.maxlen),
            speed_kmh=self.speed_kmh,
            accel_ms2=self.accel_ms2,
            pos_filter=self.pos_filter,
            gmm_history=deque(self.gmm_history, maxlen=self.gmm_history.maxlen)
        )


@dataclass
class CameraState:
    """Représentation géométrique de la caméra par rapport au terrain FIBA."""
    H_matrix: Optional[np.ndarray] = None
    is_stable: bool = False
    stable_frames_count: int = 0
    kp_filter: Optional['OneEuroFilter'] = None

    def clone(self) -> 'CameraState':
        """Copie optimisée de l'état de la caméra."""
        return CameraState(
            H_matrix=self.H_matrix.copy() if self.H_matrix is not None else None,
            is_stable=self.is_stable,
            stable_frames_count=self.stable_frames_count,
            kp_filter=self.kp_filter
        )


@dataclass
class MatchState:
    """
    Blackboard applicatif : mémoire centrale du pipeline de traitement.
    Contient l'intégralité du contexte nécessaire pour traiter la frame T.
    """

    frame_idx: int = 0
    camera: CameraState = field(default_factory=CameraState)
    players: Dict[int, PlayerState] = field(default_factory=dict)

    ball_bbox_px: Optional[Tuple[float, float, float, float]] = None
    hoop_bbox_px: Optional[Tuple[float, float, float, float]] = None

    player_masks: List[np.ndarray] = field(default_factory=list)
    net_mask: Optional[np.ndarray] = None

    events: List[str] = field(default_factory=list)

    # --- Historiques courts (Validations temporelles) ---
    ball_history: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=40))
    net_area_history: Deque[float] = field(default_factory=lambda: deque(maxlen=40))
    optical_flow_history: Deque[float] = field(default_factory=lambda: deque(maxlen=40))

    prev_frame_bgr: Optional[np.ndarray] = None

    # --- Inférence spatiale (YOLO-Pose) ---
    court_keypoints_px: Optional[np.ndarray] = None
    court_keypoints_conf: Optional[np.ndarray] = None

    # --- UI & Triggers ---
    active_triggers: Dict[str, bool] = field(default_factory=dict)
    shot_scores: Dict[str, float] = field(default_factory=dict)
    is_whistle_active: bool = False
    is_crowd_active: bool = False

    ar_stable_frames: int = 0
    ar_alpha_multiplier: float = 0.0

    is_ball_near_hoop_sticky: bool = False
    last_ball_near_hoop_frame: int = -9999

    # --- Analytique Globale ---
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

    attacking_team_id: Optional[int] = None
    target_hoop: Optional[Tuple[float, float]] = None

    def take_snapshot(self, current_frame: np.ndarray) -> FrameSnapshot:
        """
        Génère une copie déconnectée (Snapshot) de la mémoire courante.
        Garantit l'immuabilité des données nécessaires au rendu différé (Look-Ahead).
        """
        return FrameSnapshot(
            frame_idx=self.frame_idx,
            frame_bgr=current_frame.copy(),
            
            # Utilisation de la méthode de clonage native (O(N) vs copy.deepcopy)
            players={tid: player.clone() for tid, player in self.players.items()},
            
            ball_bbox_px=self.ball_bbox_px,
            hoop_bbox_px=self.hoop_bbox_px,

            court_keypoints_px=self.court_keypoints_px.copy() if self.court_keypoints_px is not None else None,
            court_keypoints_conf=self.court_keypoints_conf.copy() if self.court_keypoints_conf is not None else None,
            
            player_masks=[m.copy() if isinstance(m, np.ndarray) else m for m in self.player_masks],
            net_mask=self.net_mask.copy() if self.net_mask is not None else None,
            
            active_triggers=self.active_triggers.copy(),
            shot_scores=self.shot_scores.copy(),
            
            ar_alpha_multiplier=self.ar_alpha_multiplier,
            camera_matrix=self.camera.H_matrix.copy() if self.camera.H_matrix is not None else None,
            camera_stable=self.camera.is_stable,
            
            is_whistle_active=self.is_whistle_active,
            is_crowd_active=self.is_crowd_active,
            
            # Copie de premier niveau suffisante pour les métriques scalaires
            team_metrics={k: v.copy() for k, v in self.team_metrics.items()},
            
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