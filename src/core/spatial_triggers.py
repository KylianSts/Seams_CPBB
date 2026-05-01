"""
spatial_triggers.py
-------------------
Module de "Culling" et de déclencheurs géométriques.
Fournit des tests mathématiques ultra-rapides (AABB) pour décider 
quand activer les modèles d'IA lourds (comme SAM 2 ou le lissage complexe).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.state import PlayerState

# --- Alias de types ---
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class TriggersConfig:
    """Configuration centralisée des déclencheurs spatiaux."""
    
    # --- Trigger Panier ---
    # Élargissement de la zone de détection autour du panier.
    # Un ratio de 1.50 signifie que la zone est agrandie de 150% de sa propre largeur/hauteur.
    hoop_margin_ratio: float = 1.50 
    
    # --- Trigger Caméra (Validation Croisée) ---
    # Déplacement minimum du panier (en % de la largeur vidéo) pour soupçonner un mouvement de caméra
    cam_hoop_threshold_ratio: float = 0.05 
    # Déplacement minimum du sol (en % de la largeur vidéo) pour CONFIRMER un mouvement de caméra
    cam_floor_threshold_ratio: float = 0.002
    
    # --- Trigger Balle (Chute) ---
    falling_window_frames: int = 5
    falling_min_dy_px: float = 2.0


# ===========================================================================
# 1. DÉCLENCHEURS DE ZONES (AABB)
# ===========================================================================

def is_ball_near_hoop(
    ball_bbox: Optional[BBox], 
    hoop_bbox: Optional[BBox], 
    cfg: TriggersConfig = TriggersConfig()
) -> bool:
    """
    Vérifie si la balle se trouve dans la zone d'influence de l'arceau.
    Déclencheur principal pour activer la segmentation du filet (SAM 2)
    et les heuristiques de tir.
    """
    if ball_bbox is None or hoop_bbox is None:
        return False

    bx1, by1, bx2, by2 = ball_bbox
    hx1, hy1, hx2, hy2 = hoop_bbox

    # Centre de masse de la balle
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    w_h = hx2 - hx1
    h_h = hy2 - hy1

    # Élargissement absolu basé sur la dimension courante du panier (résiste au zoom)
    margin_x = w_h * cfg.hoop_margin_ratio
    margin_y = h_h * cfg.hoop_margin_ratio

    safe_x1 = hx1 - margin_x
    safe_x2 = hx2 + margin_x
    safe_y1 = hy1 - margin_y
    safe_y2 = hy2 + margin_y

    return (safe_x1 <= bcx <= safe_x2) and (safe_y1 <= bcy <= safe_y2)


def get_players_in_ar_zone(
    players: Dict[int, 'PlayerState'], 
    ar_zone_bbox: BBox
) -> List[int]:
    """
    Retourne les IDs des joueurs qui chevauchent physiquement une zone 2D (ex: Logo AR).
    Déclencheur pour allumer la segmentation corporelle des joueurs (Occlusion).
    """
    intersecting_ids = []
    zx1, zy1, zx2, zy2 = ar_zone_bbox

    for track_id, player in players.items():
        px1, py1, px2, py2 = player.bbox_px
        
        # Test d'intersection AABB (Axis-Aligned Bounding Box) optimisé
        overlap_x = (px1 <= zx2) and (px2 >= zx1)
        overlap_y = (py1 <= zy2) and (py2 >= zy1)

        if overlap_x and overlap_y:
            intersecting_ids.append(track_id)

    return intersecting_ids


# ===========================================================================
# 2. DÉCLENCHEURS DE COMPORTEMENT
# ===========================================================================

def is_camera_stable(
    current_hoop_bbox: Optional[BBox],
    prev_hoop_bbox: Optional[BBox],
    current_court_kp: Optional[np.ndarray],
    prev_court_kp: Optional[np.ndarray],
    vid_w: int,
    cfg: TriggersConfig = TriggersConfig()
) -> bool:
    """
    Détermine si la caméra est fixe via une Validation Croisée à 2 niveaux :
    1. Rapide : Vérifie si le panier (objet statique par excellence) a bougé.
    2. Profond : Si le panier a bougé (ex: tremblement dû à un dunk), 
       vérifie si le sol confirme ce mouvement.
    """
    hoop_stable = False
    
    # --- NIVEAU 1 : Le Panier ---
    if current_hoop_bbox is not None and prev_hoop_bbox is not None:
        cx_curr = (current_hoop_bbox[0] + current_hoop_bbox[2]) / 2.0
        cy_curr = (current_hoop_bbox[1] + current_hoop_bbox[3]) / 2.0
        cx_prev = (prev_hoop_bbox[0] + prev_hoop_bbox[2]) / 2.0
        cy_prev = (prev_hoop_bbox[1] + prev_hoop_bbox[3]) / 2.0

        dist_ratio = math.hypot(cx_curr - cx_prev, cy_curr - cy_prev) / vid_w
        
        if dist_ratio <= cfg.cam_hoop_threshold_ratio:
            hoop_stable = True
    
    if hoop_stable:
        return True
        
    # --- NIVEAU 2 : Le Sol (Validation croisée) ---
    # Le panier indique un mouvement. Est-ce un artefact (vibration physique) ou un vrai pan de caméra ?
    if current_court_kp is not None and prev_court_kp is not None:
        valid_mask = (current_court_kp[:, 0] > 0) & (prev_court_kp[:, 0] > 0)
        
        if np.any(valid_mask):
            pts_curr = current_court_kp[valid_mask]
            pts_prev = prev_court_kp[valid_mask]
            
            distances = np.linalg.norm(pts_curr - pts_prev, axis=1)
            mean_dist_ratio = float(np.mean(distances)) / vid_w
            
            # Si le sol n'a presque pas bougé, la caméra est considérée comme stable.
            if mean_dist_ratio <= cfg.cam_floor_threshold_ratio:
                return True 

    # Si les deux niveaux confirment le mouvement, la caméra bouge.
    return False


def is_ball_falling(
    ball_history: List[Tuple[int, float, float]], 
    cfg: TriggersConfig = TriggersConfig()
) -> bool:
    """
    Analyse le vecteur vertical récent de la balle.
    Utile pour distinguer un tir (balle qui descend vers l'arceau) 
    d'une passe lobée (balle qui monte).
    """
    if len(ball_history) < cfg.falling_window_frames:
        return False
        
    # Extraction des coordonnées Y (l'axe Y augmente vers le bas en OpenCV)
    recent_y = [pos[2] for pos in ball_history[-cfg.falling_window_frames:]]
    
    # Delta Y entre la position la plus ancienne de la fenêtre et la position courante
    dy = recent_y[-1] - recent_y[0]
    
    # Si dy est positif et supérieur au seuil, la balle tombe physiquement sur l'écran.
    return dy > cfg.falling_min_dy_px