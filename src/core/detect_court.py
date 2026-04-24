"""
detect_court.py
---------------
Module d'inférence spatiale (Pose Detection) pour le terrain de basket.
Utilise YOLO-Pose pour extraire les 35 points clés du terrain et calcule 
la matrice d'homographie (calibration caméra vers monde réel 2D).

Ne gère pas l'historique ni le lissage temporel (c'est le rôle du pipeline).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Géométrie FIBA (Constantes)
# ---------------------------------------------------------------------------
COURT_L, COURT_W = 28.0, 15.0
X_BASKET, X_FT, X_3PT_START = 1.575, 5.8, 2.99
Y_CENTER, Y_KEY_HALF, Y_3PT_OFFSET = 7.5, 2.45, 0.9
R_3PT, R_FT, R_DS = 6.75, 1.8, 1.25

# Mapping manuel des points FIBA (Moitié gauche)
_FIBA_M_COORDS = {
    1:  (0.0,  15.0), 2:  (0.0,  15.0 - Y_3PT_OFFSET), 3:  (0.0,  Y_CENTER + Y_KEY_HALF),
    4:  (0.0,  Y_CENTER - Y_KEY_HALF), 5:  (0.0,  Y_3PT_OFFSET), 6:  (0.0,  0.0),
    7:  (X_BASKET, Y_CENTER + 1.25), 8:  (X_BASKET, Y_CENTER - 1.25),
    9:  (X_3PT_START, 15.0 - Y_3PT_OFFSET), 10: (X_3PT_START, Y_3PT_OFFSET),
    11: (X_FT, Y_CENTER + Y_KEY_HALF), 12: (X_FT, Y_CENTER), 13: (X_FT, Y_CENTER - Y_KEY_HALF),
    14: (X_BASKET + R_3PT, Y_CENTER),
    17: (14.0, 15.0), 18: (14.0, 7.5), 19: (14.0, 0.0),
}

# Symétrie pour déduire la moitié droite automatiquement
_SYMMETRY_PAIRS_1BASED = [
    (1, 30), (2, 31), (3, 32), (4, 33), (5, 34), (6, 35),
    (7, 28), (8, 29), (9, 26), (10, 27),
    (11, 23), (12, 24), (13, 25), (14, 22)
]

def _build_world_coords() -> Dict[int, Tuple[float, float]]:
    """Génère le dictionnaire {yolo_index: (X_meters, Y_meters)}."""
    custom_id_to_xy = {k: v for k, v in _FIBA_M_COORDS.items()}
    for left_id, right_id in _SYMMETRY_PAIRS_1BASED:
        if left_id in _FIBA_M_COORDS:
            custom_id_to_xy[right_id] = (COURT_L - _FIBA_M_COORDS[left_id][0], _FIBA_M_COORDS[left_id][1])
            
    yolo_to_custom_id = [i for i in range(1, 36) if i not in (15, 16, 17, 18)]
    # Correction d'inversion spécifique à ton entraînement
    idx_19, idx_21 = yolo_to_custom_id.index(19), yolo_to_custom_id.index(21)
    yolo_to_custom_id[idx_19], yolo_to_custom_id[idx_21] = 21, 19

    return {
        yolo_idx: custom_id_to_xy[custom_id]
        for yolo_idx, custom_id in enumerate(yolo_to_custom_id) if custom_id in custom_id_to_xy
    }

WORLD_COORDS = _build_world_coords()


# ---------------------------------------------------------------------------
# Configuration & Structures de données
# ---------------------------------------------------------------------------
@dataclass
class CourtPoseConfig:
    """Configuration de l'inférence YOLO-Pose pour le terrain."""
    model_path: Path = Path("models/runs/keypoint_detection/yolo11m-pose_1000ep_v1/weights/best.pt")
    conf_keypoint: float = 0.50
    device: int = 0


@dataclass
class CourtResult:
    """Conteneur des résultats spatiaux de la frame T."""
    keypoints_px: Optional[np.ndarray] = None  # Tableau (N, 2) des coordonnées X,Y
    keypoints_conf: Optional[np.ndarray] = None # Tableau (N,) des confiances
    homography_matrix: Optional[np.ndarray] = None # Matrice H 3x3


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_court_detector(config: CourtPoseConfig):
    """Charge le modèle Ultralytics YOLO-Pose en VRAM."""
    logger.info(f"Chargement du modèle YOLO-Pose (Terrain)...")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {config.model_path.resolve()}")
    
    from ultralytics import YOLO
    model = YOLO(str(config.model_path))
    logger.info("Modèle YOLO-Pose prêt.")
    return model


# ===========================================================================
# INFÉRENCE & MATHÉMATIQUES
# ===========================================================================

def run_court_detection(model, frame: np.ndarray, config: CourtPoseConfig) -> CourtResult:
    """
    Extrait les points clés du terrain depuis l'image.
    Note : Le seuil interne (conf=0.01) est bas pour forcer YOLO à sortir tous les points,
    le filtrage se fera via config.conf_keypoint lors du calcul de l'homographie.
    """
    result = CourtResult()
    if model is None or frame is None:
        return result

    try:
        # device=config.device pour garantir l'exécution GPU
        preds = model(frame, device=config.device, verbose=False)
        
        if not preds or preds[0].keypoints is None or preds[0].keypoints.xy.shape[0] == 0:
            return result
            
        result.keypoints_px = preds[0].keypoints.xy[0].cpu().numpy()
        result.keypoints_conf = preds[0].keypoints.conf[0].cpu().numpy()
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence YOLO-Pose : {e}")

    return result


def compute_homography(court_result: CourtResult, config: CourtPoseConfig) -> np.ndarray:
    """
    Calcule la matrice de transformation 2D (Homographie) entre l'image et le modèle FIBA en mètres.
    Ne prend en compte que les points avec une confiance > config.conf_keypoint.
    """
    kp_xy = court_result.keypoints_px
    kp_conf = court_result.keypoints_conf

    if kp_xy is None or kp_conf is None:
        return None

    src_pts_px = []
    dst_pts_m = []

    for yolo_idx, world_xy in WORLD_COORDS.items():
        if yolo_idx >= len(kp_conf) or kp_conf[yolo_idx] < config.conf_keypoint:
            continue
            
        px, py = kp_xy[yolo_idx]
        if px == 0.0 and py == 0.0:
            continue
            
        src_pts_px.append([float(px), float(py)])
        dst_pts_m.append(list(world_xy))

    # Il faut un minimum mathématique de 4 points pour une homographie. 
    # 5 est une bonne sécurité pour RANSAC.
    if len(src_pts_px) < 5:
        return None

    H, status = cv2.findHomography(
        np.array(src_pts_px, dtype=np.float32), 
        np.array(dst_pts_m, dtype=np.float32), 
        cv2.RANSAC, 
        5.0
    )
    
    # Normalisation de la matrice (convention mathématique)
    if H is not None:
        H = H / H[2, 2]
        
    return H