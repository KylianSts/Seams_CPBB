"""
incrust_logo.py
---------------
Module de Réalité Augmentée (AR) pour l'incrustation de graphiques sur le terrain.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.state import MatchState

logger = logging.getLogger(__name__)

@dataclass
class LogoConfig:
    image_path: Path = Path("data/demos/assets/veolia.png")
    center_x_m: float = 2.285
    center_y_m: float = 12.0
    size_m: float = 2.0
    opacity: float = 0.75
    _logo_img: Optional[np.ndarray] = field(default=None, init=False)
    _world_corners: Optional[np.ndarray] = field(default=None, init=False)

def load_ar_assets(config: LogoConfig) -> bool:
    """Charge UN SEUL logo et calcule ses 4 coins en mètres en respectant l'aspect ratio."""
    if not config.image_path.exists():
        logger.error(f"Image logo introuvable : {config.image_path}")
        return False

    img = cv2.imread(str(config.image_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        logger.error("Le logo doit être un PNG valide avec un canal Alpha (Transparence).")
        return False
        
    config._logo_img = img

    # 1. Calcul du ratio de l'image (Largeur / Hauteur)
    h_img, w_img = img.shape[:2]
    aspect_ratio = w_img / float(h_img)

    # 2. Dimensions réelles en mètres sur le terrain
    # config.size_m dicte la largeur totale. La hauteur s'adapte.
    real_width_m = config.size_m
    real_height_m = config.size_m / aspect_ratio

    half_w = real_width_m / 2.0
    half_h = real_height_m / 2.0
    
    cx, cy = config.center_x_m, config.center_y_m
    
    # 3. Définition des 4 coins rectangulaires autour du centre
    corners = [
        [cx - half_w, cy - half_h], # Haut-Gauche
        [cx + half_w, cy - half_h], # Haut-Droit
        [cx + half_w, cy + half_h], # Bas-Droit
        [cx - half_w, cy + half_h]  # Bas-Gauche
    ]
    
    config._world_corners = np.array(corners, dtype=np.float32)
    return True

def apply_virtual_logo(frame: np.ndarray, state: MatchState, config: LogoConfig) -> np.ndarray:
    """
    Incruste un logo de manière réaliste (Mode Produit / Multiply).
    Préserve les textures, reflets et ombres du parquet sous le logo.
    """
    if config._logo_img is None or config._world_corners is None or state.camera_matrix is None:
        return frame

    # Optimisation : On ne calcule rien si l'AR est en mode invisible
    if getattr(state, 'ar_alpha_multiplier', 0.0) <= 0.0:
        return frame

    h_frame, w_frame = frame.shape[:2]
    h_logo, w_logo = config._logo_img.shape[:2]

    try:
        H_world_to_frame = np.linalg.inv(state.camera_matrix)
    except np.linalg.LinAlgError:
        return frame

    logo_pixel_corners = np.array([[0, 0], [w_logo, 0], [w_logo, h_logo], [0, h_logo]], dtype=np.float32)
    H_logo_to_world, _ = cv2.findHomography(logo_pixel_corners, config._world_corners)
    H_final = H_world_to_frame @ H_logo_to_world

    # 1. Warp du logo
    warped_logo = cv2.warpPerspective(config._logo_img, H_final, (w_frame, h_frame), flags=cv2.INTER_LINEAR)
    
    # --- Le Micro-Flou (Lens Match) ---
    # Un logo numérique est "trop parfait". On lui donne le flou optique de la caméra.
    warped_logo = cv2.GaussianBlur(warped_logo, (3, 3), 0)

    warped_rgb = warped_logo[..., :3].astype(np.float32)
    
    # Opacité combinée (Config + Machine à état Fade-in)
    current_opacity = config.opacity * state.ar_alpha_multiplier
    warped_alpha = (warped_logo[..., 3].astype(np.float32) / 255.0) * current_opacity

    # 2. Gestion de l'occlusion par les joueurs
    if state.player_masks:
        combined_occlusion = np.zeros((h_frame, w_frame), dtype=np.float32)
        for mask_float in state.player_masks:
            if mask_float.dtype == bool:
                mask_float = mask_float.astype(np.float32)
            if mask_float.shape == combined_occlusion.shape:
                combined_occlusion = np.maximum(combined_occlusion, mask_float)
        # On soustrait les joueurs de l'alpha du logo
        warped_alpha = warped_alpha * (1.0 - combined_occlusion)

    # 3. Préparation des calques pour la fusion physique
    frame_float = frame.astype(np.float32)
    warped_alpha_3d = warped_alpha[..., np.newaxis]

    # --- La Fusion "Multiply" (Produit) ---
    # Au lieu de : frame * (1-a) + logo * a
    # On fait : frame * (1-a) + (frame * (logo/255)) * a
    
    # On normalise le logo entre 0.0 et 1.0 pour la multiplication
    logo_normalized = warped_rgb / 255.0
    
    # Zone où le logo "teinte" le parquet (Multiply)
    # Les reflets du parquet (zones claires de frame_float) passeront à travers
    multiplied_zone = frame_float * logo_normalized

    # Fusion finale pondérée par l'alpha
    out_frame = frame_float * (1.0 - warped_alpha_3d) + multiplied_zone * warped_alpha_3d

    return np.clip(out_frame, 0, 255).astype(np.uint8)