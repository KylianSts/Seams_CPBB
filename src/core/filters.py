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
    
def filter_isolated_players(raw_players: list, max_overlap_ratio: float = 0.10) -> list:
    """
    Conserve uniquement les joueurs "isolés".
    Tolère une légère superposition (ex: 10% de l'aire de la BBox) pour ne pas 
    perdre de données sur des contacts mineurs.
    """
    n = len(raw_players)
    if n == 0:
        return []

    is_occluded = [False] * n

    for i in range(n):
        if is_occluded[i]: 
            continue

        bx1, by1, bx2, by2, conf = raw_players[i]
        # Calcul de l'aire totale du joueur A
        area_i = (bx2 - bx1) * (by2 - by1)

        for j in range(i + 1, n):
            cx1, cy1, cx2, cy2, conf2 = raw_players[j]
            # Calcul de l'aire totale du joueur B
            area_j = (cx2 - cx1) * (cy2 - cy1)

            # 1. Calcul des coordonnées du rectangle d'intersection
            ix1 = max(bx1, cx1)
            iy1 = max(by1, cy1)
            ix2 = min(bx2, cx2)
            iy2 = min(by2, cy2)

            # 2. Vérifier s'il y a une vraie intersection mathématique
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                
                # 3. Quel pourcentage du joueur est recouvert par l'autre ?
                ratio_i = inter_area / area_i
                ratio_j = inter_area / area_j

                # Si l'une des deux BBox est recouverte à plus de 10%, le contact est trop fort
                # On marque les DEUX joueurs comme occlusés pour les rejeter.
                if ratio_i > max_overlap_ratio or ratio_j > max_overlap_ratio:
                    is_occluded[i] = True
                    is_occluded[j] = True

    # On retourne uniquement les joueurs sains
    isolated_players = [raw_players[i] for i in range(n) if not is_occluded[i]]
    return isolated_players