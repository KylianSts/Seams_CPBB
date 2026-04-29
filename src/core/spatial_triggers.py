"""
spatial_triggers.py
-------------------
Module de "Culling" et de déclencheurs géométriques.
Fournit des tests mathématiques ultra-rapides (AABB) pour décider 
quand activer les modèles d'IA lourds (comme SAM 2).
"""

import logging
from typing import Dict, Optional, Tuple, List

from core.state import PlayerState

logger = logging.getLogger(__name__)


def is_ball_near_hoop(
    ball_bbox: Optional[Tuple[float, float, float, float]], 
    hoop_bbox: Optional[Tuple[float, float, float, float]], 
    margin_ratio: float = 1.50  # 150% d'élargissement par défaut
) -> bool:
    """
    Vérifie si la balle se trouve dans la Bounding Box de l'arceau, 
    élargie de X% proportionnellement de TOUS les côtés.
    Déclencheur pour allumer la segmentation du filet (SAM 2).
    """
    if ball_bbox is None or hoop_bbox is None:
        return False

    bx1, by1, bx2, by2 = ball_bbox
    hx1, hy1, hx2, hy2 = hoop_bbox

    # Centre de la balle
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    # Dimensions natives de la boîte du panier
    w_h = hx2 - hx1
    h_h = hy2 - hy1

    # Élargissement proportionnel de tous les côtés
    safe_x1 = hx1 - (w_h * margin_ratio)
    safe_x2 = hx2 + (w_h * margin_ratio)
    safe_y1 = hy1 - (h_h * margin_ratio)
    safe_y2 = hy2 + (h_h * margin_ratio)

    # Test d'inclusion géométrique
    return (safe_x1 <= bcx <= safe_x2) and (safe_y1 <= bcy <= safe_y2)


def get_players_in_ar_zone(
    players: Dict[int, PlayerState], 
    ar_zone_bbox: Tuple[float, float, float, float]
) -> List[int]:
    """
    Retourne les IDs des joueurs qui chevauchent la Bounding Box d'un élément AR.
    (Valable pour un logo virtuel, une zone peinte (raquette), la ligne à 3 points, etc.)
    Déclencheur pour allumer la segmentation corporelle des joueurs (SAM 2).
    """
    intersecting_ids = []
    zx1, zy1, zx2, zy2 = ar_zone_bbox

    for track_id, player in players.items():
        px1, py1, px2, py2 = player.bbox_px
        
        # Test d'intersection AABB (Axis-Aligned Bounding Box) standard
        # Les rectangles se croisent si l'un ne finit pas avant que l'autre commence
        overlap_x = (px1 <= zx2) and (px2 >= zx1)
        overlap_y = (py1 <= zy2) and (py2 >= zy1)

        if overlap_x and overlap_y:
            intersecting_ids.append(track_id)

    return intersecting_ids


def is_camera_stable(
    current_hoop_bbox: Optional[Tuple[float, float, float, float]], 
    prev_hoop_bbox: Optional[Tuple[float, float, float, float]], 
    threshold_ratio: float = 0.02  # 2% de la largeur du panier
) -> bool:
    """
    Détermine si la caméra est stable en mesurant le déplacement relatif du panier.
    Indispensable pour gérer différentes résolutions vidéo (720p, 1080p, 4K).
    
    Args:
        current_hoop_bbox: [x1, y1, x2, y2] à T
        prev_hoop_bbox: [x1, y1, x2, y2] à T-1
        threshold_ratio: Déplacement maximal autorisé (en % de la largeur de la BBox).
    """
    if current_hoop_bbox is None or prev_hoop_bbox is None:
        return False

    # 1. On calcule le centre du panier actuel
    curr_w = current_hoop_bbox[2] - current_hoop_bbox[0]
    if curr_w <= 0:
        return False
        
    curr_cx = current_hoop_bbox[0] + (curr_w / 2.0)
    curr_cy = (current_hoop_bbox[1] + current_hoop_bbox[3]) / 2.0
    
    # 2. On calcule le centre du panier précédent
    prev_cx = (prev_hoop_bbox[0] + prev_hoop_bbox[2]) / 2.0
    prev_cy = (prev_hoop_bbox[1] + prev_hoop_bbox[3]) / 2.0

    # 3. Calcul de la distance de déplacement (Euclidienne)
    movement_px = ((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)**0.5

    # 4. Conversion en pourcentage de la taille de l'objet
    movement_ratio = movement_px / curr_w

    # 5. La caméra est stable si le mouvement relatif est inférieur au seuil
    return float(movement_ratio) < threshold_ratio


def is_ball_falling(ball_history: list, window: int = 5, min_dy_px: float = 2.0) -> bool:
    """
    Analyse l'historique récent de la balle pour déterminer si elle est en phase 
    descendante (chute vers le panier). L'axe Y augmente vers le bas de l'image.
    
    Args:
        ball_history: Liste de tuples (frame_idx, cx, cy)
        window: Nombre de frames à regarder en arrière pour lisser le bruit
        min_dy_px: Seuil de descente minimum en pixels pour valider la chute
    """
    # S'il n'y a pas assez d'historique, on accepte par défaut pour ne pas bloquer
    if len(ball_history) < window:
        return False
        
    # On extrait les coordonnées Y des X dernières frames
    recent_y = [pos[2] for pos in ball_history[-window:]]
    
    # Tendance : Y_actuel - Y_ancien
    # Si dy > 0, la balle est plus basse sur l'écran qu'il y a 5 frames -> Elle tombe.
    dy = recent_y[-1] - recent_y[0]
    
    return dy > min_dy_px