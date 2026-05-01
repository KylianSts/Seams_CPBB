"""
filters.py
----------
Module d'heuristiques, de lissage et de filtrage métier.
Nettoie les détections brutes en appliquant les règles physiques du basketball.
"""

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.state import MatchState

# --- Alias de types pour la lisibilité ---
BBox = Tuple[float, float, float, float]          # (x1, y1, x2, y2)
Detection = Tuple[float, float, float, float, float] # (x1, y1, x2, y2, conf)


@dataclass
class FiltersConfig:
    """Configuration centralisée des paramètres de filtrage métier."""
    max_players_on_court: int = 10
    
    # --- Ratios d'extraction du torse (Ignorer bras/jambes/tête) ---
    torso_crop_x: float = 0.25      # Enlève 25% de chaque côté
    torso_crop_y_top: float = 0.20  # Enlève 20% en haut (tête)
    torso_crop_y_bot: float = 0.40  # Enlève 40% en bas (jambes)
    
    max_overlap_ratio: float = 0.10 # Tolérance d'occlusion (5%) pour joueur 'isolé'
    
    # --- Paramètres de lissage ---
    ball_prox_half_life: float = 50.0 # Demi-vie spatiale (en pixels) pour la continuité de la balle
    ema_alpha_default: float = 0.2    # Réactivité par défaut du lissage EMA
    bidirectional_window: int = 15     # Fenêtre par défaut pour le lissage Non-Causal


# ===========================================================================
# 1. UTILITAIRES GÉOMÉTRIQUES (DRY)
# ===========================================================================

def get_torso_box(box: Tuple[float, ...], cfg: FiltersConfig) -> BBox:
    """
    Extrait la sous-boîte correspondant au torse d'un joueur, 
    partie la plus stable pour l'extraction de couleur et le calcul d'occlusion.
    """
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    
    tx1 = x1 + (w * cfg.torso_crop_x)
    ty1 = y1 + (h * cfg.torso_crop_y_top)
    tx2 = x2 - (w * cfg.torso_crop_x)
    ty2 = y2 - (h * cfg.torso_crop_y_bot)
    
    return (tx1, ty1, tx2, ty2)


def get_intersection_area(box1: BBox, box2: BBox) -> float:
    """Calcule l'aire d'intersection stricte entre deux Bounding Boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    if ix1 < ix2 and iy1 < iy2:
        return (ix2 - ix1) * (iy2 - iy1)
    return 0.0


# ===========================================================================
# 2. FILTRAGES SPATIAUX ET MÉTIERS
# ===========================================================================

def filter_top_players(raw_players: List[Detection], cfg: FiltersConfig = FiltersConfig()) -> List[Detection]:
    """
    Garde uniquement les N détections avec la plus haute probabilité.
    (Présume que la liste est déjà triée par confiance décroissante).
    """
    return raw_players[:cfg.max_players_on_court]


def filter_best_ball(raw_balls: List[Detection], state: 'MatchState', cfg: FiltersConfig = FiltersConfig()) -> Optional[Detection]:
    """
    Détermine la balle la plus probable en pondérant la confiance de l'IA (RF-DETR)
    par la proximité spatiale avec la position de la balle à la frame T-1.
    """
    if not raw_balls:
        return None
    
    if isinstance(raw_balls, tuple) or (isinstance(raw_balls, list) and isinstance(raw_balls[0], float)):
        raw_balls = [raw_balls]  # type: ignore

    if not state.ball_history:
        return raw_balls[0]

    _, last_x, last_y = state.ball_history[-1]

    best_ball = None
    best_score = -1.0

    for bx1, by1, bx2, by2, conf in raw_balls:
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        
        dist = math.hypot(cx - last_x, cy - last_y)
        dist_score = 1.0 / (1.0 + (dist / cfg.ball_prox_half_life))
        
        combined_score = (conf * 0.5) + (dist_score * 0.5)
        
        if combined_score > best_score:
            best_score = combined_score
            best_ball = (bx1, by1, bx2, by2, conf)

    return best_ball


def filter_isolated_players(raw_players: List[Detection], cfg: FiltersConfig = FiltersConfig()) -> List[Detection]:
    """
    Élimine les joueurs subissant une occlusion (même partielle) de leur torse.
    Essentiel pour fournir des échantillons propres au modèle GMM (Couleurs d'équipe).
    """
    n = len(raw_players)
    if n == 0:
        return []

    is_occluded = [False] * n
    torso_boxes = [get_torso_box(p, cfg) for p in raw_players]
    areas = [max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in torso_boxes]

    for i in range(n):
        if is_occluded[i] or areas[i] == 0: 
            continue

        for j in range(i + 1, n):
            if areas[j] == 0:
                continue

            inter_area = get_intersection_area(torso_boxes[i], torso_boxes[j])
            
            if inter_area > 0:
                ratio_i = inter_area / areas[i]
                ratio_j = inter_area / areas[j]

                if ratio_i > cfg.max_overlap_ratio or ratio_j > cfg.max_overlap_ratio:
                    is_occluded[i] = True
                    is_occluded[j] = True

    return [raw_players[i] for i in range(n) if not is_occluded[i]]


def calculate_occlusion_ratios(players_dict: dict, cfg: FiltersConfig = FiltersConfig()) -> Dict[int, float]:
    """
    Calcule le ratio d'occlusion [0.0 - 1.0] du torse de chaque joueur actif.
    Retourne { track_id: occlusion_ratio }.
    """
    ratios = {}
    player_list = list(players_dict.values())
    n = len(player_list)

    torso_data = []
    for p in player_list:
        t_box = get_torso_box(p.bbox_px, cfg)
        area = max(0, t_box[2] - t_box[0]) * max(0, t_box[3] - t_box[1])
        torso_data.append((p.track_id, t_box, area))

    for i in range(n):
        tid1, t1, area1 = torso_data[i]
        max_occlusion = 0.0
        
        if area1 > 0:
            for j in range(n):
                if i == j: 
                    continue
                    
                _, t2, _ = torso_data[j]
                inter_area = get_intersection_area(t1, t2)

                if inter_area > 0:
                    ratio = inter_area / area1
                    if ratio > max_occlusion:
                        max_occlusion = ratio

        ratios[tid1] = min(1.0, max_occlusion)

    return ratios


# ===========================================================================
# 3. LISSAGES TEMPORELS (1D / 2D)
# ===========================================================================

def apply_ema_2d(current_pos: Tuple[float, float], 
                 previous_pos: Optional[Tuple[float, float]], 
                 cfg: FiltersConfig = FiltersConfig()) -> Tuple[float, float]:
    """Lissage par Moyenne Mobile Exponentielle (EMA) sur des coordonnées 2D."""
    if previous_pos is None:
        return current_pos

    alpha = cfg.ema_alpha_default
    smooth_x = (alpha * current_pos[0]) + ((1.0 - alpha) * previous_pos[0])
    smooth_y = (alpha * current_pos[1]) + ((1.0 - alpha) * previous_pos[1])
    
    return (smooth_x, smooth_y)


def bidirectional_smooth(pos_history: List[Tuple[int, float, float]], 
                         target_frame_idx: int, 
                         cfg: FiltersConfig = FiltersConfig()) -> Optional[Tuple[float, float]]:
    """
    Lissage Non-Causal (Look-Ahead).
    Interpole la position à l'instant T en regardant le passé et le futur,
    pondéré linéairement par la distance temporelle.
    """
    points_x, points_y, weights = [], [], []

    for item in pos_history:
        if len(item) != 3:
            continue
            
        f_idx, x, y = item
        dist = abs(f_idx - target_frame_idx)
        
        if dist <= cfg.bidirectional_window:
            w = 1.0 - (dist / (cfg.bidirectional_window + 1.0))
            points_x.append(x * w)
            points_y.append(y * w)
            weights.append(w)

    if not weights:
        return None

    sum_w = sum(weights)
    return (sum(points_x) / sum_w, sum(points_y) / sum_w)


class OneEuroFilter:
    """
    Filtre 1 Euro (Casiez 2012).
    Lissage adaptatif : forte réduction du jitter à basse vitesse, 
    faible latence à haute vitesse.
    """
    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
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
            return self.x_prev # type: ignore

        # Lissage de la dérivée (vitesse)
        a_d = self.smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Lissage de la position adaptatif
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.mincutoff + self.beta * speed
        a_k = self.smoothing_factor(t_e, cutoff)
        x_hat = a_k * x + (1 - a_k) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def get_geometric_capsule_masks(
    player_boxes: List[BBox], 
    frame_shape: Tuple[int, int],
    width_ratio: float = 0.70,
    top_margin_ratio: float = 0.15,
    blur_size: int = 21
) -> List[np.ndarray]:
    """
    Génère des masques approximatifs (Capsules 2D) basés sur la géométrie des Bounding Boxes.
    Alternative très légère au Deep Learning (ex: SAM) pour gérer l'occlusion.
    
    Args:
        player_boxes: Liste des Bounding Boxes (x1, y1, x2, y2) des joueurs.
        frame_shape: Dimensions de l'image source (Hauteur, Largeur).
        width_ratio: Réduction de la largeur de la BBox pour affiner la capsule.
        top_margin_ratio: Marge supérieure pour ajuster la hauteur des épaules/tête.
        blur_size: Taille du noyau gaussien (doit être un nombre impair) pour le feathering.
        
    Returns:
        Liste de masques 2D en virgule flottante [0.0, 1.0].
    """
    if not player_boxes:
        return []

    h_img, w_img = frame_shape[:2]
    masks = []
    
    for box in player_boxes:
        x1, y1, x2, y2 = box
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        capsule_w = int(box_w * width_ratio) 
        radius = int(capsule_w / 2)
        
        cx = int(x1 + (box_w / 2))
        bottom_cy = int(y2 - radius)
        top_cy = int(y1 + (box_h * top_margin_ratio) + radius) 
        
        if top_cy > bottom_cy:
            top_cy = bottom_cy

        cv2.circle(mask, (cx, top_cy), radius, 255, -1)
        cv2.circle(mask, (cx, bottom_cy), radius, 255, -1)
        cv2.rectangle(mask, (cx - radius, top_cy), (cx + radius, bottom_cy), 255, -1)
        
        mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        mask_float = mask_blurred.astype(np.float32) / 255.0
        
        masks.append(mask_float)
        
    return masks