"""
detect_court.py
---------------
Module d'inférence spatiale (Pose Detection) pour le terrain de basket.
Extrait les points clés FIBA via YOLO-Pose et calcule la matrice d'homographie
(Calibration Caméra -> Monde Réel 2D) avec validation de l'erreur de reprojection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ===========================================================================
# GÉOMÉTRIE FIBA ABSOLUE (Constantes Mathématiques)
# ===========================================================================
COURT_L, COURT_W = 28.0, 15.0
X_BASKET, X_FT, X_3PT_START = 1.575, 5.8, 2.99
Y_CENTER, Y_KEY_HALF, Y_3PT_OFFSET = 7.5, 2.45, 0.9
R_3PT = 6.75

# Mapping des points FIBA (Moitié gauche)
_FIBA_M_COORDS = {
    1:  (0.0,  15.0), 2:  (0.0,  15.0 - Y_3PT_OFFSET), 3:  (0.0,  Y_CENTER + Y_KEY_HALF),
    4:  (0.0,  Y_CENTER - Y_KEY_HALF), 5:  (0.0,  Y_3PT_OFFSET), 6:  (0.0,  0.0),
    7:  (X_BASKET, Y_CENTER + 1.25), 8:  (X_BASKET, Y_CENTER - 1.25),
    9:  (X_3PT_START, 15.0 - Y_3PT_OFFSET), 10: (X_3PT_START, Y_3PT_OFFSET),
    11: (X_FT, Y_CENTER + Y_KEY_HALF), 12: (X_FT, Y_CENTER), 13: (X_FT, Y_CENTER - Y_KEY_HALF),
    14: (X_BASKET + R_3PT, Y_CENTER),
    17: (14.0, 15.0), 18: (14.0, 7.5), 19: (14.0, 0.0),
}

# Symétrie pour la moitié droite
_SYMMETRY_PAIRS_1BASED = [
    (1, 30), (2, 31), (3, 32), (4, 33), (5, 34), (6, 35),
    (7, 28), (8, 29), (9, 26), (10, 27),
    (11, 23), (12, 24), (13, 25), (14, 22)
]

def _build_world_coords() -> Dict[int, Tuple[float, float]]:
    """Construit le dictionnaire de référence {yolo_index: (X_meters, Y_meters)}."""
    custom_id_to_xy = {k: v for k, v in _FIBA_M_COORDS.items()}
    for left_id, right_id in _SYMMETRY_PAIRS_1BASED:
        if left_id in _FIBA_M_COORDS:
            custom_id_to_xy[right_id] = (COURT_L - _FIBA_M_COORDS[left_id][0], _FIBA_M_COORDS[left_id][1])
            
    yolo_to_custom_id = [i for i in range(1, 36) if i not in (15, 16, 17, 18)]
    
    # Correction d'inversion spécifique à l'entraînement
    idx_19, idx_21 = yolo_to_custom_id.index(19), yolo_to_custom_id.index(21)
    yolo_to_custom_id[idx_19], yolo_to_custom_id[idx_21] = 21, 19

    return {
        yolo_idx: custom_id_to_xy[custom_id]
        for yolo_idx, custom_id in enumerate(yolo_to_custom_id) if custom_id in custom_id_to_xy
    }

WORLD_COORDS = _build_world_coords()

# Les 4 coins du terrain en mètres pour le lissage de l'homographie
COURT_CORNERS_M = np.array([
    [0.0, 0.0], [COURT_L, 0.0], [COURT_L, COURT_W], [0.0, COURT_W]
], dtype=np.float32).reshape(-1, 1, 2)


# ===========================================================================
# CONFIGURATION ET STRUCTURES
# ===========================================================================

@dataclass
class CourtPoseConfig:
    """Configuration de l'IA et des règles mathématiques (Homographie)."""
    model_path: Path = Path("models/runs/keypoint_detection/yolo11m-pose_1000ep_v1/weights/best.pt")
    device: int = 0
    
    # --- Tolérances de l'IA ---
    conf_keypoint: float = 0.50
    
    # --- Sécurités Homographiques ---
    min_points_required: int = 5    # 4 est le minimum mathématique, 5 offre une redondance
    ransac_threshold: float = 5.0   # Tolérance d'erreur de reprojection en pixels
    min_inlier_ratio: float = 0.60  # Rejette la matrice si RANSAC sacrifie plus de 40% des points


@dataclass
class CourtResult:
    """Conteneur des prédictions brutes de la frame T."""
    keypoints_px: Optional[np.ndarray] = None
    keypoints_conf: Optional[np.ndarray] = None


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_court_detector(config: CourtPoseConfig) -> Any:
    """Charge dynamiquement le modèle Ultralytics YOLO-Pose."""
    logger.info("Chargement du modèle YOLO-Pose (Terrain)...")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {config.model_path.resolve()}")
    
    from ultralytics import YOLO
    model = YOLO(str(config.model_path))
    logger.info("Modèle YOLO-Pose prêt.")
    return model


# ===========================================================================
# INFÉRENCE & GÉOMÉTRIE
# ===========================================================================

def run_court_detection(model: Any, frame: np.ndarray, config: CourtPoseConfig) -> CourtResult:
    """Extrait les points clés du terrain depuis l'image."""
    result = CourtResult()
    if model is None or frame is None:
        return result

    try:
        preds = model(frame, device=config.device, verbose=False)
        
        if not preds or preds[0].keypoints is None or preds[0].keypoints.xy.shape[0] == 0:
            return result
            
        result.keypoints_px = preds[0].keypoints.xy[0].cpu().numpy()
        result.keypoints_conf = preds[0].keypoints.conf[0].cpu().numpy()
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence YOLO-Pose : {e}")

    return result


def compute_homography(court_result: CourtResult, config: CourtPoseConfig) -> Optional[np.ndarray]:
    """
    Calcule la matrice de transformation 2D entre l'image et le modèle FIBA.
    Intègre une validation stricte du ratio d'Inliers via l'algorithme MAGSAC.
    """
    kp_xy = court_result.keypoints_px
    kp_conf = court_result.keypoints_conf

    if kp_xy is None or kp_conf is None:
        return None

    src_pts_px, dst_pts_m = [], []

    # 1. Filtrage par confiance
    for yolo_idx, world_xy in WORLD_COORDS.items():
        if yolo_idx >= len(kp_conf) or kp_conf[yolo_idx] < config.conf_keypoint:
            continue
            
        px, py = kp_xy[yolo_idx]
        if px == 0.0 and py == 0.0:
            continue
            
        src_pts_px.append([float(px), float(py)])
        dst_pts_m.append(list(world_xy))

    if len(src_pts_px) < config.min_points_required:
        return None

    # 2. Calcul robuste (USAC_MAGSAC)
    H, status = cv2.findHomography(
        np.array(src_pts_px, dtype=np.float32), 
        np.array(dst_pts_m, dtype=np.float32), 
        cv2.USAC_MAGSAC, 
        config.ransac_threshold
    )
    
    if H is None or status is None:
        return None

    # 3. Contrôle Qualité (Rejet si la géométrie est absurde)
    inliers_count = int(np.sum(status))
    inlier_ratio = inliers_count / len(src_pts_px)
    
    if inlier_ratio < config.min_inlier_ratio:
        logger.debug(f"Homographie rejetée (Inliers: {inlier_ratio:.2f} < {config.min_inlier_ratio})")
        return None
    
    # 4. Normalisation mathématique standard
    H = H / H[2, 2]
    return H


def smooth_homography(H_raw: np.ndarray, H_old: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """
    Applique un lissage EMA (Exponential Moving Average) sur l'homographie
    en interpolant la projection des 4 coins du terrain dans l'espace Pixel.
    Cette technique garantit une transition géométrique valide (contrairement 
    à un lissage direct des coefficients de la matrice).
    """
    if H_old is None or H_raw is None:
        return H_raw
        
    try:
        H_raw_inv = np.linalg.inv(H_raw)
        H_old_inv = np.linalg.inv(H_old)
        
        pts_px_new = cv2.perspectiveTransform(COURT_CORNERS_M, H_raw_inv)
        pts_px_old = cv2.perspectiveTransform(COURT_CORNERS_M, H_old_inv)
        
        # EMA sur les coordonnées 2D
        pts_px_smooth = (alpha * pts_px_new) + ((1.0 - alpha) * pts_px_old)
        
        # Recalcul d'une homographie parfaite (Sans RANSAC, 4 points = 1 solution exacte)
        H_smooth, _ = cv2.findHomography(pts_px_smooth, COURT_CORNERS_M, 0)
        
        return H_smooth if H_smooth is not None else H_raw
        
    except np.linalg.LinAlgError:
        return H_raw