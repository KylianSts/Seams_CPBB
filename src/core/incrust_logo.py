"""
incrust_logo.py
---------------
Module de Réalité Augmentée (AR) sur surface plane.
Gère la projection homographique d'assets 2D sur le parquet FIBA,
l'occlusion dynamique par les joueurs, et l'intégration optique (Lens Match).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.state import FrameSnapshot

logger = logging.getLogger(__name__)


@dataclass
class LogoConfig:
    """
    Configuration spatiale et visuelle d'un élément de Réalité Augmentée.
    L'image doit obligatoirement inclure un canal Alpha (RGBA).
    """
    image_path: Path
    center_x_m: float
    center_y_m: float
    size_m: float
    opacity: float = 0.75
    
    # Caches internes générés au chargement
    _logo_img: Optional[np.ndarray] = field(default=None, init=False)
    _world_corners: Optional[np.ndarray] = field(default=None, init=False)


# ===========================================================================
# 1. PRÉPARATION DES ASSETS
# ===========================================================================

def load_ar_assets(config: LogoConfig) -> bool:
    """
    Charge l'image en RAM et calcule ses coordonnées dans le monde réel (mètres)
    tout en préservant strictement son ratio d'aspect original.
    """
    if not config.image_path.exists():
        logger.error(f"Asset AR introuvable : {config.image_path}")
        return False

    img = cv2.imread(str(config.image_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        logger.error(f"Asset invalide (PNG avec canal Alpha requis) : {config.image_path.name}")
        return False
        
    config._logo_img = img

    # Calcul des dimensions physiques (La largeur est imposée, la hauteur s'adapte)
    h_img, w_img = img.shape[:2]
    aspect_ratio = w_img / float(h_img)

    real_width_m = config.size_m
    real_height_m = config.size_m / aspect_ratio

    half_w = real_width_m / 2.0
    half_h = real_height_m / 2.0
    cx, cy = config.center_x_m, config.center_y_m
    
    corners = [
        [cx - half_w, cy - half_h], # Haut-Gauche
        [cx + half_w, cy - half_h], # Haut-Droit
        [cx + half_w, cy + half_h], # Bas-Droit
        [cx - half_w, cy + half_h]  # Bas-Gauche
    ]
    
    config._world_corners = np.array(corners, dtype=np.float32)
    return True


# ===========================================================================
# 2. MOTEUR DE RENDU AR
# ===========================================================================

def apply_virtual_logo(frame: np.ndarray, state: 'FrameSnapshot', config: LogoConfig) -> np.ndarray:
    """
    Projette et incruste le logo sur le terrain via la matrice de caméra courante.
    Utilise le mode de fusion "Multiply" pour préserver les ombres et reflets du parquet.
    Gère l'occlusion spatiale via les masques des joueurs.
    """
    if config._logo_img is None or config._world_corners is None or state.camera_matrix is None:
        return frame

    # Fast-exit : l'animation de disparition est terminée (ou pas commencée)
    if state.ar_alpha_multiplier <= 0.0:
        return frame

    h_frame, w_frame = frame.shape[:2]
    h_logo, w_logo = config._logo_img.shape[:2]

    # 1. Calcul de l'homographie Logo -> Image vidéo
    try:
        H_world_to_frame = np.linalg.inv(state.camera_matrix)
    except np.linalg.LinAlgError:
        return frame

    logo_pixel_corners = np.array([[0, 0], [w_logo, 0], [w_logo, h_logo], [0, h_logo]], dtype=np.float32)
    H_logo_to_world, _ = cv2.findHomography(logo_pixel_corners, config._world_corners)
    
    H_final = H_world_to_frame @ H_logo_to_world

    # 2. Transformation spatiale
    warped_logo = cv2.warpPerspective(config._logo_img, H_final, (w_frame, h_frame), flags=cv2.INTER_LINEAR)
    
    # Intégration optique (Lens Match) : simulation du flou de caméra
    warped_logo = cv2.GaussianBlur(warped_logo, (3, 3), 0)

    # 3. Traitement des canaux de couleur et d'opacité
    warped_rgb = warped_logo[..., :3].astype(np.float32)
    
    current_opacity = config.opacity * state.ar_alpha_multiplier
    warped_alpha = (warped_logo[..., 3].astype(np.float32) / 255.0) * current_opacity

    # 4. Traitement de l'occlusion (Soustraction des masques joueurs)
    if state.player_masks:
        # Empilement vectorisé pour trouver l'occlusion maximale par pixel instantanément
        masks_stack = np.stack(state.player_masks)
        combined_occlusion = np.max(masks_stack, axis=0).astype(np.float32)
        
        # La transparence du logo devient nulle là où un joueur est présent
        warped_alpha *= (1.0 - combined_occlusion)

    # 5. Fusion Photométrique (Multiply Blend)
    frame_float = frame.astype(np.float32)
    warped_alpha_3d = warped_alpha[..., np.newaxis]
    
    # Le logo agit comme un "filtre teintant" sur la luminosité originelle du sol
    logo_normalized = warped_rgb / 255.0
    multiplied_zone = frame_float * logo_normalized

    # Interpolation linéaire finale basée sur l'Alpha calculé
    out_frame = frame_float * (1.0 - warped_alpha_3d) + multiplied_zone * warped_alpha_3d

    return np.clip(out_frame, 0, 255).astype(np.uint8)