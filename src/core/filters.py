"""
filters.py
----------
Module d'heuristiques et de filtrage métier.
Nettoie les détections brutes en appliquant les règles physiques du basketball.
"""

import math
import numpy as np

from typing import List, Tuple, Optional
from core.state import MatchState

def filter_top_10_players(raw_players: List[Tuple]) -> List[Tuple]:
    """
    Règle : Pas plus de 10 joueurs sur le terrain.
    Garde uniquement les 10 détections avec la plus haute probabilité.
    (Les joueurs sont déjà triés par probabilité décroissante dans detect_objects.py)
    """
    return raw_players[:10]


def filter_best_ball(raw_balls: List[Tuple], state: MatchState) -> Optional[Tuple]:
    """
    Règle : Un seul ballon.
    Score combiné = (Confiance RF-DETR) + (Proximité spatiale avec la frame précédente).
    """
    if not raw_balls:
        return None
    
    # Si le détecteur renvoie un tuple simple (x1, y1, x2, y2, conf) au lieu d'une liste
    if isinstance(raw_balls, tuple) or (isinstance(raw_balls, list) and isinstance(raw_balls[0], float)):
        raw_balls = [raw_balls] # On l'encapsule dans une liste pour pouvoir boucler dessus

    # S'il n'y a pas d'historique (première frame), on prend la plus probable
    if not state.ball_history:
        return raw_balls[0]

    # Récupération de la position (x, y) de la balle à la frame précédente
    _, last_x, last_y = state.ball_history[-1]

    best_ball = None
    best_score = -1.0

    for bx1, by1, bx2, by2, conf in raw_balls:
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        
        # Distance Euclidienne
        dist = ((cx - last_x)**2 + (cy - last_y)**2)**0.5
        
        # Conversion de la distance en un "Score de Proximité" (0 à 1)
        # Plus la distance est grande, plus le score tend vers 0.
        # Le '50.0' est un facteur d'atténuation (demi-vie à 50 pixels).
        dist_score = 1.0 / (1.0 + (dist / 50.0))
        
        # Poids égal (50/50) entre la confiance de l'IA et la physique
        combined_score = (conf * 0.5) + (dist_score * 0.5)
        
        if combined_score > best_score:
            best_score = combined_score
            best_ball = (bx1, by1, bx2, by2, conf)

    return best_ball

def apply_ema_2d(current_pos: Tuple[float, float], 
                 previous_pos: Optional[Tuple[float, float]], 
                 alpha: float = 0.2) -> Tuple[float, float]:
    """
    Applique un lissage de Moyenne Mobile Exponentielle (EMA) sur des coordonnées 2D.
    - current_pos : La position brute (x, y) de la frame actuelle.
    - previous_pos : La position lissée (x, y) de la frame précédente.
    - alpha : Réactivité (0.0 à 1.0). Plus c'est proche de 0, plus c'est fluide mais avec de la latence.
    """
    if previous_pos is None:
        return current_pos # Première frame, pas de lissage possible

    smooth_x = (alpha * current_pos[0]) + ((1.0 - alpha) * previous_pos[0])
    smooth_y = (alpha * current_pos[1]) + ((1.0 - alpha) * previous_pos[1])
    
    return (smooth_x, smooth_y)

class OneEuroFilter:
    """Filtre 1 Euro pour un seul point 2D (utilisé pour un joueur)."""
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        # Lissage de la vitesse
        a_d = self.smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Lissage de la position
        speed = np.linalg.norm(dx_hat)
        cutoff = self.mincutoff + self.beta * speed
        a_k = self.smoothing_factor(t_e, cutoff)
        x_hat = a_k * x + (1 - a_k) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
    
def filter_isolated_players(raw_players: list, max_overlap_ratio: float = 0.05) -> list:
    """
    Conserve uniquement les joueurs dont le TORSE n'est pas occlusé par un autre TORSE.
    Utilise les mêmes ratios de sous-boîte que detect_team.py pour valider la donnée.
    """
    n = len(raw_players)
    if n == 0:
        return []

    is_occluded = [False] * n

    # Fonction locale pour extraire les coordonnées de la sous-boîte (Torse)
    def get_torso_box(box):
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        tx1 = x1 + (w * 0.25)
        ty1 = y1 + (h * 0.20)
        tx2 = x2 - (w * 0.25)
        ty2 = y2 - (h * 0.40)
        return tx1, ty1, tx2, ty2

    for i in range(n):
        if is_occluded[i]: 
            continue

        # Torse du joueur A
        t_ix1, t_iy1, t_ix2, t_iy2 = get_torso_box(raw_players[i])
        area_i = max(0, t_ix2 - t_ix1) * max(0, t_iy2 - t_iy1)

        for j in range(i + 1, n):
            # Torse du joueur B
            t_jx1, t_jy1, t_jx2, t_jy2 = get_torso_box(raw_players[j])
            area_j = max(0, t_jx2 - t_jx1) * max(0, t_jy2 - t_jy1)

            # Calcul de l'intersection entre les DEUX TORSES
            ix1 = max(t_ix1, t_jx1)
            iy1 = max(t_iy1, t_jy1)
            ix2 = min(t_ix2, t_jx2)
            iy2 = min(t_iy2, t_jy2)

            # S'il y a une vraie superposition
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                
                # Quel pourcentage du torse est recouvert ?
                ratio_i = inter_area / area_i if area_i > 0 else 0
                ratio_j = inter_area / area_j if area_j > 0 else 0

                # Si l'un des deux torses est recouvert à plus de X% par l'autre
                if ratio_i > max_overlap_ratio or ratio_j > max_overlap_ratio:
                    is_occluded[i] = True
                    is_occluded[j] = True

    # On retourne les joueurs certifiés sains
    isolated_players = [raw_players[i] for i in range(n) if not is_occluded[i]]
    return isolated_players


def bidirectional_smooth(pos_history, target_frame_idx: int, window: int = 7) -> Optional[Tuple[float, float]]:
    """
    Lissage bidirectionnel (Non-Causal) pour la minimap.
    Regarde les positions brutes dans le passé et dans le futur autour de `target_frame_idx`.
    """
    points_x = []
    points_y = []
    weights = []

    for item in pos_history:
        # SÉCURITÉ : On s'assure que l'élément est bien un tuple de 3 (frame, x, y)
        if len(item) != 3:
            continue
            
        f_idx, x, y = item
        dist = abs(f_idx - target_frame_idx)
        
        if dist <= window:
            w = 1.0 - (dist / (window + 1.0))
            points_x.append(x * w)
            points_y.append(y * w)
            weights.append(w)

    if not weights:
        return None

    sum_w = sum(weights)
    return (sum(points_x) / sum_w, sum(points_y) / sum_w)


def calculate_occlusion_ratios(players_dict: dict) -> dict:
    """
    Calcule le ratio d'occlusion (0.0 à 1.0) de la sous-boîte 'Torse' de chaque joueur.
    Retourne un dictionnaire { track_id : ratio }.
    """
    ratios = {}
    player_list = list(players_dict.values())
    n = len(player_list)

    # Fonction locale pour extraire le torse exactement comme dans detect_team.py
    def get_torso(box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        return (x1 + w * 0.25, y1 + h * 0.20, x2 - w * 0.25, y2 - h * 0.40)

    for i in range(n):
        p1 = player_list[i]
        t1 = get_torso(p1.bbox_px)
        area1 = max(0, t1[2] - t1[0]) * max(0, t1[3] - t1[1])

        max_occlusion = 0.0
        
        if area1 > 0:
            for j in range(n):
                if i == j: continue
                p2 = player_list[j]
                t2 = get_torso(p2.bbox_px)

                # Calcul de l'intersection
                ix1 = max(t1[0], t2[0])
                iy1 = max(t1[1], t2[1])
                ix2 = min(t1[2], t2[2])
                iy2 = min(t1[3], t2[3])

                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    ratio = inter_area / area1
                    if ratio > max_occlusion:
                        max_occlusion = ratio

        ratios[p1.track_id] = min(1.0, max_occlusion)

    return ratios