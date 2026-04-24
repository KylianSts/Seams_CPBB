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
# 3. VALIDATION MORPHOLOGIQUE (Déformation de la BBox)
# ---------------------------------------------------------------------------

def check_hoop_deformation(current_hoop_bbox: Tuple[float, float, float, float], hoop_ref_dims: Tuple[float, float]) -> Tuple[float, float]:
    """
    Regarde si la Bounding Box du panier s'étire (souvent vers le bas lors d'un swish).
    Retourne (variation_largeur, variation_hauteur).
    """
    if current_hoop_bbox is None or hoop_ref_dims is None:
        return 0.0, 0.0

    x1, y1, x2, y2 = current_hoop_bbox
    w, h = x2 - x1, y2 - y1
    ref_w, ref_h = hoop_ref_dims

    v_w = (w - ref_w) / ref_w
    v_h = (h - ref_h) / ref_h
    
    return v_w, v_h

# ---------------------------------------------------------------------------
# 4. VALIDATION PAR FLUX OPTIQUE (Mouvement interne)
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

# ---------------------------------------------------------------------------
# 5. VALIDATION D'OCCLUSION (Indice de pénétration)
# ---------------------------------------------------------------------------

def check_ball_occlusion(ball_detected: bool, last_ball_pos: Tuple[float, float], hoop_bbox: Tuple[float, float, float, float]) -> float:
    """
    Si la balle n'est plus détectée alors qu'elle était juste au-dessus du panier,
    c'est un fort indice qu'elle est à l'intérieur du filet.
    """
    if ball_detected: return 0.0
    if last_ball_pos is None or hoop_bbox is None: return 0.0
    
    # Si la dernière position connue était dans la moitié haute du panier
    hx1, hy1, hx2, hy2 = hoop_bbox
    lx, ly = last_ball_pos
    
    if (hx1 <= lx <= hx2) and (hy1 <= ly <= hy1 + (hy2-hy1)*0.5):
        return 1.0 # Probabilité maximale d'entrée dans le filet
        
    return 0.0


def check_ball_velocity_profile(ball_history: List[Tuple[int, float, float]], hoop_bbox: Tuple[float, float, float, float], fps: float = 30.0) -> float:
    """
    Analyse la signature de décélération de la balle lors de son passage dans l'arceau.
    Un panier réussi présente une courbe de ralentissement progressive (friction du filet).
    """
    if len(ball_history) < 6 or hoop_bbox is None:
        return 0.0

    hx1, hy1, hx2, hy2 = hoop_bbox
    
    # 1. Extraction des vitesses verticales (Vy) sur les dernières frames
    # Vy = (y_actuel - y_precedent)
    vy_list = []
    for i in range(1, len(ball_history)):
        vy = ball_history[i][2] - ball_history[i-1][2]
        vy_list.append(vy)

    # 2. On isole la phase où la balle est "dans" ou "sous" l'arceau
    # On cherche une décélération (Vy qui diminue) alors que la balle tombe (Vy > 0)
    decelerations = []
    for i in range(1, len(vy_list)):
        accel = vy_list[i] - vy_list[i-1]
        
        # Si la balle est dans la zone du panier
        curr_y = ball_history[i][2]
        if hy1 < curr_y < hy2:
            decelerations.append(accel)

    if not decelerations:
        return 0.0

    # 3. Analyse du profil
    # Un swish = décélération constante et modérée (la balle freine dans les mailles)
    # Un choc (arceau) = décélération énorme et instantanée
    # Un airball = accélération positive (pesanteur)
    
    avg_accel = sum(decelerations) / len(decelerations)
    
    # Score 1.0 si on observe un freinage fluide (avg_accel est négatif)
    # On calibre : une petite décélération constante est idéale.
    if avg_accel < 0:
        # Plus c'est fluide et constant, plus le score est haut
        # On évite les chocs trop violents (rebonds)
        score = min(1.0, abs(avg_accel) / 5.0) 
        return float(score)
    
    return 0.0