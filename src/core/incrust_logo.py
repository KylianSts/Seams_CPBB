"""
incrust_logo.py
---------------
Module de Réalité Augmentée (AR) pour l'incrustation de graphiques sur le terrain.

Prend une image (logo PNG), la déforme selon la perspective de la caméra (Homographie),
applique les masques d'occlusion (pour que les joueurs passent "devant" le logo),
et fusionne le tout avec l'image vidéo brute.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.state import MatchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LogoConfig:
    """Configuration du logo virtuel et de sa position sur le terrain FIBA."""
    image_path: Path = Path("data/demos/assets/cenergy.png")
    
    # Position du centre du logo en mètres (par défaut : milieu droite du terrain)
    center_x_m: float = 2.285
    center_y_m: float = 12.0
    
    # Taille du logo en mètres
    size_m: float = 2.0
    
    # Opacité globale (0.0 = invisible, 1.0 = opaque)
    opacity: float = 0.75

    # Variable interne pour éviter de recharger l'image depuis le disque à chaque frame
    _logo_img: Optional[np.ndarray] = field(default=None, init=False)
    _world_corners: Optional[np.ndarray] = field(default=None, init=False)


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_ar_assets(config: LogoConfig) -> bool:
    """
    Charge le logo en mémoire et pré-calcule ses coordonnées physiques
    pour DEUX positions (Symétrie Axiale sur le même bord de touche).
    À appeler UNE SEULE FOIS dans run_pipeline.py.
    """
    logger.info(f"Chargement de l'asset AR : {config.image_path.name}...")
    
    if not config.image_path.exists():
        logger.error(f"Logo introuvable : {config.image_path.resolve()}")
        return False

    # Chargement en gardant le canal Alpha (IMREAD_UNCHANGED)
    img = cv2.imread(str(config.image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        logger.error("Le logo doit être un PNG avec transparence (4 canaux : BGRA).")
        return False
        
    config._logo_img = img

    # --- CALCUL DE LA SYMÉTRIE AXIALE ---
    COURT_L = 28.0
    half = config.size_m / 2.0
    
    # Position 1 (Originale, définie dans config)
    cx1, cy1 = config.center_x_m, config.center_y_m
    
    # Position 2 (Miroir gauche/droite, mais on GARDE la même profondeur Y)
    cx2, cy2 = COURT_L - cx1, cy1 

    # On stocke les deux positions dans une liste
    positions = [(cx1, cy1), (cx2, cy2)]
    config._world_corners_list = []

    for cx, cy in positions:
        corners = [
            [cx - half, cy - half],  # Haut-Gauche
            [cx + half, cy - half],  # Haut-Droit
            [cx + half, cy + half],  # Bas-Droit
            [cx - half, cy + half]   # Bas-Gauche
        ]
        config._world_corners_list.append(np.array(corners, dtype=np.float32))
    
    logger.info("Asset AR prêt (Double Logo Axiale).")
    return True


# ===========================================================================
# RENDU (INCRUSTATION AR)
# ===========================================================================

def apply_virtual_logo(frame: np.ndarray, state: MatchState, config: LogoConfig) -> np.ndarray:
    """
    Déforme et incruste les logos sur la frame vidéo.
    Gère automatiquement l'occlusion si des masques de joueurs sont présents.
    """
    if config._logo_img is None or not hasattr(config, '_world_corners_list'):
        return frame

    # 1. Vérification de la calibration caméra
    H_frame_to_world = state.camera.H_matrix
    if H_frame_to_world is None:
        return frame

    h_frame, w_frame = frame.shape[:2]
    h_logo, w_logo = config._logo_img.shape[:2]

    try:
        # Inversion de la matrice : On veut passer du Monde (Mètres) -> à l'Image (Pixels)
        H_world_to_frame = np.linalg.inv(H_frame_to_world)
    except np.linalg.LinAlgError:
        logger.warning("Matrice d'homographie singulière. Incrustation annulée.")
        return frame

    # 2. Pré-calcul du masque d'occlusion global (Optimisation de performance)
    combined_occlusion = np.zeros((h_frame, w_frame), dtype=np.float32)
    if state.player_masks:
        for mask_float in state.player_masks:
            if mask_float.dtype == bool:
                mask_float = mask_float.astype(np.float32)
            if mask_float.shape == combined_occlusion.shape:
                combined_occlusion = np.maximum(combined_occlusion, mask_float)

    # Coordonnées pixel d'origine du logo
    logo_pixel_corners = np.array([[0, 0], [w_logo, 0], [w_logo, h_logo], [0, h_logo]], dtype=np.float32)

    # On convertit la frame en float32 une seule fois pour éviter les pertes de précision lors du blending multiple
    out_frame = frame.copy().astype(np.float32)

    # 3. Boucle d'incrustation pour chaque logo
    for world_corners in config._world_corners_list:
        
        # Calcul de la matrice finale pour ce logo spécifique
        H_logo_to_world, _ = cv2.findHomography(logo_pixel_corners, world_corners)
        H_final = H_world_to_frame @ H_logo_to_world

        # Déformation de l'image (Warping)
        warped_logo = cv2.warpPerspective(
            config._logo_img, 
            H_final, 
            (w_frame, h_frame), 
            flags=cv2.INTER_LINEAR
        )

        # Séparation RGB et calcul de l'Alpha initial
        warped_rgb = warped_logo[..., :3].astype(np.float32)
        warped_alpha = (warped_logo[..., 3].astype(np.float32) / 255.0) * config.opacity

        # 4. Application de l'occlusion
        if state.player_masks:
            warped_alpha = warped_alpha * (1.0 - combined_occlusion)

        # 5. Fusion finale pour ce logo (Alpha Blending)
        warped_alpha_3d = warped_alpha[..., np.newaxis]
        out_frame = out_frame * (1.0 - warped_alpha_3d) + warped_rgb * warped_alpha_3d

    # Reconversion en format d'image standard
    return out_frame.astype(np.uint8)