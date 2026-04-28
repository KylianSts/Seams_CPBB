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
    Trace une ligne UNIQUEMENT si la balle a formellement traversé l'arceau 
    de haut en bas, en respectant l'ordre chronologique.
    """
    if len(ball_history) < 2 or hoop_bbox is None:
        return 0.0

    hx1, hy1, hx2, hy2 = hoop_bbox
    hoop_width = hx2 - hx1
    hoop_center_x = (hx1 + hx2) / 2.0

    p_above = None
    p_below = None

    # On remonte le temps : du présent (fin de liste) vers le passé
    for _, bx, by in reversed(ball_history):
        
        # 1. On cherche d'abord la position BASSE (au présent ou passé récent)
        if by > hy2 and p_below is None:
            p_below = (bx, by)
        
        # 2. On cherche ensuite la position HAUTE (dans le passé plus lointain)
        elif by < hy1:
            if p_below is not None:
                # OK : On a bien vu la balle en bas, PUIS on la retrouve en haut dans le passé.
                p_above = (bx, by)
                break
            else:
                # VETO TEMPOREL : La balle est en haut, mais on n'a pas encore vu la descente.
                # Le tir est donc encore en l'air. On coupe immédiatement.
                return 0.0

    if not p_above or not p_below:
        return 0.0

    # Interpolation mathématique de l'intersection (inchangée)
    try:
        intersect_x = p_above[0] + (p_below[0] - p_above[0]) * (hy1 - p_above[1]) / (p_below[1] - p_above[1])
    except ZeroDivisionError:
        return 0.0

    dist_to_center = abs(intersect_x - hoop_center_x)
    max_dist = hoop_width / 2.0

    if dist_to_center > max_dist:
        return 0.0 
    
    return float(max(0.0, 1.0 - (dist_to_center / max_dist)))

# ---------------------------------------------------------------------------
# 2. VALIDATION PHYSIQUE (SAM2 - Gonflement du filet)
# ---------------------------------------------------------------------------
def check_net_area_variation(net_area_history: list) -> float:
    """
    Analyse la courbe en cloche de l'aire du filet (Change Point Detection).
    Cherche une expansion soutenue par rapport à une baseline au repos.
    """
    # Il faut au moins ~10 frames d'historique pour voir une bosse se former
    if len(net_area_history) < 10:
        return 0.0

    # La "Baseline" (filet au repos) = le 20ème percentile (ignore les valeurs aberrantes basses)
    baseline_area = np.percentile(net_area_history, 20)
    
    if baseline_area < 100:  # Sécurité si le masque est vide ou buggé
        return 0.0
        
    # L'aire maximale atteinte très récemment (sur les 5 dernières frames)
    recent_max_area = max(net_area_history[-5:])
    
    # Calcul du ratio d'expansion
    expansion_ratio = (recent_max_area - baseline_area) / baseline_area
    
    # Normalisation : un vrai Swish fait gonfler le filet d'environ 40% à 80%
    # On donne le score max (1.0) dès 50% d'expansion
    score = expansion_ratio / 0.50
    
    return float(min(1.0, max(0.0, score)))


# ---------------------------------------------------------------------------
# 3. VALIDATION PAR FLUX OPTIQUE (Mouvement interne masqué)
# ---------------------------------------------------------------------------
def get_hoop_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray, hoop_bbox: tuple, net_mask: np.ndarray) -> float:
    """
    Calcule la magnitude du flux optique UNIQUEMENT sur les pixels du filet (masqué par SAM).
    Renvoie le mouvement moyen pour la frame T.
    """
    if prev_frame is None or curr_frame is None or hoop_bbox is None or net_mask is None:
        return 0.0

    x1, y1, x2, y2 = map(int, hoop_bbox)
    margin = 20
    h_img, w_img = curr_frame.shape[:2]
    
    # Sécurité des bords
    y1_m, y2_m = max(0, y1-margin), min(h_img, y2+margin)
    x1_m, x2_m = max(0, x1-margin), min(w_img, x2+margin)

    roi_prev = prev_frame[y1_m:y2_m, x1_m:x2_m]
    roi_curr = curr_frame[y1_m:y2_m, x1_m:x2_m]
    mask_crop = net_mask[y1_m:y2_m, x1_m:x2_m]
    
    if roi_prev.size == 0 or not np.any(mask_crop): 
        return 0.0

    gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # LA MAGIE : On multiplie le mouvement par le masque (0 ou 1)
    masked_mag = mag * mask_crop.astype(np.float32)
    net_pixels_count = np.sum(mask_crop)
    
    if net_pixels_count == 0:
        return 0.0
        
    avg_net_movement = np.sum(masked_mag) / net_pixels_count
    
    # Normalisation : un mouvement moyen de 5 pixels par frame est très fort
    return float(min(1.0, avg_net_movement / 5.0))


def check_optical_flow_signature(flow_history: list, threshold: float = 0.25, min_duration: int = 3) -> float:
    """
    Change Point Detection sur l'historique du flux optique.
    Renvoie 1.0 si le filet a bougé de manière CONTINUE pendant au moins 'min_duration' frames.
    """
    if len(flow_history) < min_duration:
        return 0.0
        
    consecutive_frames = 0
    max_consecutive = 0
    
    for val in flow_history:
        if val >= threshold:
            consecutive_frames += 1
            if consecutive_frames > max_consecutive:
                max_consecutive = consecutive_frames
        else:
            consecutive_frames = 0
            
    # Si on a détecté une vague de mouvement soutenue, c'est validé
    return 1.0 if max_consecutive >= min_duration else 0.0