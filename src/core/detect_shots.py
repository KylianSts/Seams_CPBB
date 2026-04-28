"""
detect_shots.py
---------------
Module de diagnostic pour la validation de tirs.
Fournit un ensemble de fonctions indépendantes (probabilistes) pour 
analyser le comportement de la balle et du panier.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from core.state import MatchState

# ---------------------------------------------------------------------------
# 1. VALIDATION GÉOMÉTRIQUE (Trajectoire)
# ---------------------------------------------------------------------------

def check_geometric_crossing(ball_history: list, hoop_bbox: tuple) -> float:
    """
    Trace une ligne entre la dernière position au-dessus et la première en-dessous.
    Score 1.0 si la ligne passe au centre du panier, 0.0 si elle rate ou effleure.
    """
    if len(ball_history) < 2 or hoop_bbox is None:
        return 0.0

    hx1, hy1, hx2, hy2 = hoop_bbox
    hoop_width = hx2 - hx1
    hoop_center_x = (hx1 + hx2) / 2.0

    p_above = None
    p_below = None

    # 1. Recherche des points pivots (en partant de la fin pour être au plus proche)
    for _, bx, by in reversed(ball_history):
        if by < hy1 and p_above is None:
            p_above = (bx, by)
        if by > hy2 and p_below is None:
            p_below = (bx, by)
        
        if p_above and p_below:
            break

    if not p_above or not p_below:
        return 0.0

    # 2. Calcul de l'intersection sur la ligne Y = hy1 (le haut de la boîte)
    # Formule de l'interpolation linéaire : x = x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)
    try:
        intersect_x = p_above[0] + (p_below[0] - p_above[0]) * (hy1 - p_above[1]) / (p_below[1] - p_above[1])
    except ZeroDivisionError:
        return 0.0

    # 3. Calcul du score selon la proximité du centre
    # On regarde la distance entre l'intersection et le centre du panier
    dist_to_center = abs(intersect_x - hoop_center_x)
    max_dist = hoop_width / 2.0

    if dist_to_center > max_dist:
        return 0.0  # La ligne passe en dehors des bords gauche/droite
    
    # Score linéaire : 1.0 au centre, 0.0 au bord de l'arceau
    score = 1.0 - (dist_to_center / max_dist)
    
    return float(max(0.0, score))

# ---------------------------------------------------------------------------
# 2. VALIDATION PHYSIQUE (SAM2 - Aire du filet)
# ---------------------------------------------------------------------------

def check_net_area_variation(current_net_mask: np.ndarray, net_area_history: List[float]) -> float:
    """
    Compare l'aire actuelle du filet à la moyenne historique (stabilité).
    Retourne le ratio de variation.
    """
    if current_net_mask is None or len(net_area_history) < 5:
        return 0.0

    current_area = np.sum(current_net_mask)
    avg_area = sum(net_area_history) / len(net_area_history)
    
    if avg_area == 0: return 0.0
    
    variation = (current_area - avg_area) / avg_area
    # On retourne la variation positive (gonflement du filet)
    return max(0.0, variation)


# ---------------------------------------------------------------------------
# 3. VALIDATION PAR FLUX OPTIQUE (Mouvement interne)
# ---------------------------------------------------------------------------

def get_hoop_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray, hoop_bbox: Tuple[float, float, float, float]) -> float:
    """
    Calcule le mouvement moyen à l'intérieur de la zone du panier.
    Utile pour détecter un filet qui bouge même si SAM2 ne l'isole pas parfaitement.
    """
    if prev_frame is None or curr_frame is None or hoop_bbox is None:
        return 0.0

    # Crop de la zone du panier (avec une petite marge)
    x1, y1, x2, y2 = map(int, hoop_bbox)
    margin = 20
    roi_prev = prev_frame[max(0, y1-margin):y2+margin, max(0, x1-margin):x2+margin]
    roi_curr = curr_frame[max(0, y1-margin):y2+margin, max(0, x1-margin):x2+margin]
    
    if roi_prev.size == 0 or roi_curr.size == 0: return 0.0

    # Conversion en gris
    gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)

    # Calcul du flux optique (Farneback est robuste)
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # On calcule la magnitude moyenne du mouvement
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))