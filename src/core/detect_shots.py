"""
detect_shots.py
---------------
Module probabiliste de validation de tirs (Shot Detection).
Combine trois heuristiques indépendantes (Géométrie spatiale, Déformation du filet via SAM,
et Flux Optique) pour générer un score de confiance composite.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Alias de types ---
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
BallHistory = List[Tuple[int, float, float]]  # [(frame_idx, cx, cy), ...]


@dataclass
@dataclass
class ShotConfig:
    """Configuration centralisée des seuils probabilistes de tir."""

    # --- Physique du Filet (SAM) ---
    net_min_history_frames: int = 10
    net_recent_window: int = 5
    net_baseline_percentile: int = 20      # Centile pour déterminer l'aire "au repos"
    net_expansion_target_ratio: float = 0.50 # On donne un score de 1.0 à partir de 50% d'expansion

    # --- Gestion Temporelle ---
    shot_cooldown_frames: int = 60         # Délai de repos après un tir réussi

    # --- Flux Optique ---
    flow_margin_px: int = 20               # Marge autour de la BBox du panier pour le crop
    flow_normalization_max: float = 5.0    # Un mouvement moyen de 5 pixels par frame donne un score de 1.0
    flow_activation_threshold: float = 0.25
    flow_min_duration_frames: int = 3

class ShotState(Enum):
    """Les 4 états possibles de notre machine temporelle."""
    IDLE = 0         # Rien à signaler, on attend
    TRACKING = 1     # Mode Sticky ON ! La balle est dans la zone, on analyse
    SCORED = 2       # Un tir vient d'être marqué (Cooldown long en cours)
    MISSED = 3       # Le tir a échoué / Timeout (Cooldown court en cours)

class ShotManager:
    """
    Machine à états finis gérant le cycle de vie d'un tir.
    """
    def __init__(self, cfg: ShotConfig):
        self.cfg = cfg
        self.state = ShotState.IDLE
        
        # Compteurs internes (invisibles pour le pipeline)
        self._cooldown = 0
        self._consecutive_near = 0
        self._frames_lost = 0

    def update(
        self, 
        ball_detected: bool, 
        is_near: bool, 
        is_falling: bool, 
        came_from_above: bool
    ) -> ShotState:
        """Mise à jour à appeler à chaque frame."""
        
        # 1. Gestion des Cooldowns (Temps mort après une action)
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self.state = ShotState.IDLE
            return self.state

        # 2. Mise à jour des compteurs de présence
        if is_near:
            self._consecutive_near += 1
            self._frames_lost = 0
        else:
            self._consecutive_near = 0
            if self.state == ShotState.TRACKING:
                self._frames_lost += 1

        # 3. Transitions d'États
        if self.state == ShotState.IDLE:
            # Condition d'entrée (Le Triple Filtre !)
            if self._consecutive_near >= 3 and is_falling and came_from_above:
                self.state = ShotState.TRACKING
                self._frames_lost = 0

        elif self.state == ShotState.TRACKING:
            # Conditions de sortie par Timeout (Coupure du Sticky)
            if ball_detected and self._frames_lost > 15:
                self._trigger_miss()
            elif self._frames_lost > 30: # Timeout complet
                self._trigger_miss()

        return self.state

    def register_success(self):
        """À appeler par le pipeline quand la physique (SAM/Flux optique) valide le tir."""
        self.state = ShotState.SCORED
        self._cooldown = self.cfg.shot_cooldown_frames
        self._consecutive_near = 0
        self._frames_lost = 0

    def _trigger_miss(self):
        """Action interne quand le Sticky Timeout est atteint."""
        self.state = ShotState.MISSED
        self._cooldown = 15  # Cooldown court après un échec
        self._consecutive_near = 0
        self._frames_lost = 0
        
    @property
    def is_tracking(self) -> bool:
        """Propriété de confort pour le pipeline."""
        return self.state == ShotState.TRACKING


# ===========================================================================
# 1. VALIDATION GÉOMÉTRIQUE (Trajectoire de la balle)
# ===========================================================================

def check_geometric_crossing(ball_history: BallHistory, hoop_bbox: Optional[BBox]) -> float:
    """
    Analyse l'historique récent de la balle pour valider qu'elle a traversé l'arceau
    strictement de HAUT en BAS.
    Retourne un score de 0.0 à 1.0 basé sur la centralité du passage.
    """
    if len(ball_history) < 2 or hoop_bbox is None:
        return 0.0

    hx1, hy1, hx2, hy2 = hoop_bbox
    hoop_width = hx2 - hx1
    hoop_center_x = (hx1 + hx2) / 2.0

    p_above = None
    p_below = None

    # Analyse rétrochronologique (du présent vers le passé)
    for _, bx, by in reversed(ball_history):
        
        # 1. On cherche d'abord la position BASSE (fin de la chute)
        if by > hy2 and p_below is None:
            p_below = (bx, by)
        
        # 2. On cherche ensuite la position HAUTE (avant la chute)
        elif by < hy1:
            if p_below is not None:
                # Validation chronologique : la balle était en haut AVANT d'être en bas
                p_above = (bx, by)
                break
            else:
                # VETO : La balle est en haut mais n'est jamais redescendue (Tir en l'air)
                return 0.0

    if not p_above or not p_below:
        return 0.0

    # Interpolation mathématique de l'intersection avec l'axe Y du panier
    try:
        intersect_x = p_above[0] + (p_below[0] - p_above[0]) * (hy1 - p_above[1]) / (p_below[1] - p_above[1])
    except ZeroDivisionError:
        return 0.0

    # Évaluation de la précision (Swish = 1.0, Bricque = 0.0)
    dist_to_center = abs(intersect_x - hoop_center_x)
    max_dist = hoop_width / 2.0

    if dist_to_center > max_dist:
        return 0.0 
    
    return float(max(0.0, 1.0 - (dist_to_center / max_dist)))


# ===========================================================================
# 2. VALIDATION PHYSIQUE (Gonflement du filet via SAM)
# ===========================================================================

def check_net_area_variation(net_area_history: List[float], cfg: ShotConfig = ShotConfig()) -> float:
    """
    Analyse la courbe d'aire du masque SAM du filet.
    Recherche une expansion ponctuelle (courbe en cloche) caractéristique d'un swish.
    """
    if len(net_area_history) < cfg.net_min_history_frames:
        return 0.0

    # La "Baseline" (filet au repos) ignore les micro-mouvements et le bruit de segmentation
    baseline_area = np.percentile(net_area_history, cfg.net_baseline_percentile)
    
    if baseline_area < 100:  # Sécurité si le masque est aberrant ou vide
        return 0.0
        
    recent_max_area = max(net_area_history[-cfg.net_recent_window:])
    expansion_ratio = (recent_max_area - baseline_area) / baseline_area
    
    # Normalisation linéaire : expansion cible = score maximal (1.0)
    score = expansion_ratio / cfg.net_expansion_target_ratio
    
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# 3. VALIDATION PAR FLUX OPTIQUE (Mouvement interne)
# ===========================================================================

def get_hoop_optical_flow(
    prev_frame: Optional[np.ndarray], 
    curr_frame: Optional[np.ndarray], 
    hoop_bbox: Optional[BBox], 
    net_mask: Optional[np.ndarray],
    cfg: ShotConfig = ShotConfig()
) -> float:
    """
    Calcule la magnitude du flux optique (Farneback) restreinte strictement
    aux pixels appartenant au masque du filet.
    """
    if prev_frame is None or curr_frame is None or hoop_bbox is None or net_mask is None:
        return 0.0

    x1, y1, x2, y2 = map(int, hoop_bbox)
    h_img, w_img = curr_frame.shape[:2]
    
    # Découpe sécurisée
    y1_m = max(0, y1 - cfg.flow_margin_px)
    y2_m = min(h_img, y2 + cfg.flow_margin_px)
    x1_m = max(0, x1 - cfg.flow_margin_px)
    x2_m = min(w_img, x2 + cfg.flow_margin_px)

    roi_prev = prev_frame[y1_m:y2_m, x1_m:x2_m]
    roi_curr = curr_frame[y1_m:y2_m, x1_m:x2_m]
    mask_crop = net_mask[y1_m:y2_m, x1_m:x2_m]
    
    if roi_prev.size == 0 or not np.any(mask_crop): 
        return 0.0

    gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Isoler le mouvement du filet via une multiplication matricielle
    masked_mag = mag * mask_crop.astype(np.float32)
    net_pixels_count = np.sum(mask_crop)
    
    if net_pixels_count == 0:
        return 0.0
        
    avg_net_movement = np.sum(masked_mag) / net_pixels_count
    
    return float(np.clip(avg_net_movement / cfg.flow_normalization_max, 0.0, 1.0))


def check_optical_flow_signature(flow_history: List[float], cfg: ShotConfig = ShotConfig()) -> float:
    """
    Analyse temporelle du flux optique.
    Un vrai tir provoque un mouvement continu et soutenu, contrairement à un glitch visuel.
    """
    if len(flow_history) < cfg.flow_min_duration_frames:
        return 0.0
        
    consecutive_frames = 0
    max_consecutive = 0
    
    for val in flow_history:
        if val >= cfg.flow_activation_threshold:
            consecutive_frames += 1
            if consecutive_frames > max_consecutive:
                max_consecutive = consecutive_frames
        else:
            consecutive_frames = 0
            
    return 1.0 if max_consecutive >= cfg.flow_min_duration_frames else 0.0