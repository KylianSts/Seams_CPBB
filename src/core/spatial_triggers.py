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
    cam_hoop_threshold_ratio: float = 0.01 
    
    # --- Trigger Balle (Chute) ---
    falling_window_frames: int = 5
    falling_min_dy_px: float = 2.0

    # --- Trigger Balle (Altitude) ---
    altitude_window_frames: int = 20  # On regarde jusqu'à 20 frames en arrière pour trouver le pic


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
    prev_hoop: Optional[BBox], 
    curr_hoop: Optional[BBox], 
    vid_w: int, 
    cfg: TriggersConfig = TriggersConfig()
) -> bool:
    """
    Détermine si la caméra est stable en se basant UNIQUEMENT 
    sur le déplacement du panier entre deux frames.
    """
    # S'il n'y a pas de panier visible sur l'une des deux frames, 
    # on suppose par sécurité que la caméra est stable pour ne pas briser l'AR.
    if prev_hoop is None or curr_hoop is None:
        return True 

    # 1. Calcul du centre du panier (Frame N-1)
    px_c = (prev_hoop[0] + prev_hoop[2]) / 2.0
    py_c = (prev_hoop[1] + prev_hoop[3]) / 2.0
    
    # 2. Calcul du centre du panier (Frame N)
    cx_c = (curr_hoop[0] + curr_hoop[2]) / 2.0
    cy_c = (curr_hoop[1] + curr_hoop[3]) / 2.0

    # 3. Calcul de la distance parcourue en pixels
    distance = math.hypot(cx_c - px_c, cy_c - py_c)
    dist_ratio = distance / vid_w

    # La caméra est considérée comme stable si le déplacement est inférieur au seuil (ex: 5%)
    return dist_ratio <= cfg.cam_hoop_threshold_ratio


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


def has_ball_passed_above_hoop(
    ball_history: List[Tuple[int, float, float]], 
    hoop_bbox: Optional[BBox], 
    cfg: TriggersConfig = TriggersConfig()
) -> bool:
    """
    Filtre d'Altitude (Spatial Gate).
    Vérifie si la balle s'est trouvée physiquement plus haut que l'arceau 
    dans son historique récent. Empêche le déclenchement d'un tir si 
    le ballon est manipulé en dessous du filet (ex: arbitre, rebond bas).
    """
    if not ball_history or hoop_bbox is None:
        return False

    hx1, hy1, hx2, hy2 = hoop_bbox

    # On isole uniquement la fenêtre temporelle qui nous intéresse
    recent_history = ball_history[-cfg.altitude_window_frames:]

    for _, _, cy in recent_history:
        # Rappel OpenCV : l'axe Y part de 0 en haut et augmente vers le bas.
        # Si le centre Y de la balle (cy) est plus petit que le haut du panier (hy1),
        # alors la balle était "au-dessus" du panier.
        if cy < hy1:
            return True

    return False