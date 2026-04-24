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

class OneEuroFilterVectorized:
    """
    Filtre 1 Euro optimisé pour Numpy.
    Traite un tableau (N, 2) de coordonnées simultanément.
    Gère les points (0, 0) issus des non-détections de YOLO-Pose.
    """
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff  # Lissage au repos (plus c'est bas, plus ça lisse)
        self.beta = beta            # Réactivité (plus c'est haut, plus ça réduit le lissage en mouvement)
        self.dcutoff = dcutoff      # Lissage de la vitesse
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Applique le filtre.
        t : timestamp courant en secondes.
        x : np.ndarray de shape (N, 2) contenant les points bruts.
        """
        # Masque pour ignorer les points que YOLO n'a pas trouvés (0.0, 0.0)
        valid_mask = (x[:, 0] > 0) & (x[:, 1] > 0)
        
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        # Tableaux de travail
        x_hat = self.x_prev.copy()
        dx_hat = self.dx_prev.copy()
        
        if np.any(valid_mask):
            # Extraction des sous-matrices valides
            x_v = x[valid_mask]
            xp_v = self.x_prev[valid_mask]
            dxp_v = self.dx_prev[valid_mask]
            
            # Lissage de la dérivée (vitesse)
            a_d = self.smoothing_factor(t_e, self.dcutoff)
            dx_v = (x_v - xp_v) / t_e
            dx_hat_v = a_d * dx_v + (1 - a_d) * dxp_v
            
            # Calcul de la fréquence de coupure dynamique (basée sur la vitesse)
            speed = np.linalg.norm(dx_hat_v, axis=1, keepdims=True)
            cutoff = self.mincutoff + self.beta * speed
            
            # Lissage de la position
            r = 2 * math.pi * cutoff * t_e
            a_k = r / (r + 1)
            x_hat_v = a_k * x_v + (1 - a_k) * xp_v
            
            # Réintégration
            x_hat[valid_mask] = x_hat_v
            dx_hat[valid_mask] = dx_hat_v

        # Sécurité : Si un point redevient valide après avoir été à (0,0), on reset son état
        zero_prev_mask = (self.x_prev[:, 0] == 0) & (self.x_prev[:, 1] == 0)
        reset_mask = valid_mask & zero_prev_mask
        
        if np.any(reset_mask):
            x_hat[reset_mask] = x[reset_mask]
            dx_hat[reset_mask] = 0.0

        # Mise à jour de la mémoire interne
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t
        
        # On force les points invalides actuels à (0,0) dans la sortie
        # pour que compute_homography les ignore correctement
        out = x_hat.copy()
        out[~valid_mask] = 0.0
        return out