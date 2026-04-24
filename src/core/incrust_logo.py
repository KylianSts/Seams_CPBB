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
    Charge le logo en mémoire et pré-calcule ses coordonnées physiques.
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

    # Pré-calcul des 4 coins du logo en MÈTRES sur le terrain FIBA
    half = config.size_m / 2.0
    cx, cy = config.center_x_m, config.center_y_m
    
    corners = [
        [cx - half, cy - half],  # Haut-Gauche
        [cx + half, cy - half],  # Haut-Droit
        [cx + half, cy + half],  # Bas-Droit
        [cx - half, cy + half]   # Bas-Gauche
    ]
    config._world_corners = np.array(corners, dtype=np.float32)
    
    logger.info("Asset AR prêt.")
    return True


# ===========================================================================
# RENDU (INCRUSTATION AR)
# ===========================================================================

def apply_virtual_logo(frame: np.ndarray, state: MatchState, config: LogoConfig) -> np.ndarray:
    """
    Déforme et incruste le logo sur la frame vidéo.
    Gère automatiquement l'occlusion si des masques de joueurs sont présents dans le state.
    """
    if config._logo_img is None or config._world_corners is None:
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

    # 2. Calcul de la déformation du logo
    # On calcule d'abord comment placer les pixels du PNG carré sur les coordonnées en mètres
    logo_pixel_corners = np.array([[0, 0], [w_logo, 0], [w_logo, h_logo], [0, h_logo]], dtype=np.float32)
    H_logo_to_world, _ = cv2.findHomography(logo_pixel_corners, config._world_corners)

    # Matrice finale : Pixels Logo -> Mètres Terrain -> Pixels Frame
    H_final = H_world_to_frame @ H_logo_to_world

    # 3. Déformation de l'image (Warping)
    warped_logo = cv2.warpPerspective(
        config._logo_img, 
        H_final, 
        (w_frame, h_frame), 
        flags=cv2.INTER_LINEAR
    )

    # Séparation des couleurs (RGB) et de la transparence (Alpha)
    warped_rgb = warped_logo[..., :3]
    # Normalisation de l'Alpha entre 0.0 et 1.0, pondérée par l'opacité désirée
    warped_alpha = (warped_logo[..., 3].astype(np.float32) / 255.0) * config.opacity

    # 4. Gestion de l'occlusion (Masques continus / Feathering)
    if state.player_masks:
        # On crée une toile vide de type "float32" au lieu de "bool"
        combined_occlusion = np.zeros((h_frame, w_frame), dtype=np.float32)
        
        for mask_float in state.player_masks:
            # Sécurité pour vérifier que c'est bien un masque float32 (compatible SAM et Capsule)
            if mask_float.dtype == bool:
                mask_float = mask_float.astype(np.float32)
                
            if mask_float.shape == combined_occlusion.shape:
                # On utilise np.maximum pour fusionner les opacités des joueurs 
                # (empêche l'opacité de dépasser 1.0 s'ils se chevauchent)
                combined_occlusion = np.maximum(combined_occlusion, mask_float)
                
        # Modification progressive de la transparence du logo
        # Ex: Si le joueur est à 0.8 d'opacité (bord flou), le logo devient à 0.2 d'opacité
        warped_alpha = warped_alpha * (1.0 - combined_occlusion)

    # 5. Fusion finale (Alpha Blending optimisé via NumPy)
    out_frame = frame.copy()
    
    # Expansion de la dimension alpha pour le broadcast NumPy (H, W) -> (H, W, 1)
    warped_alpha_3d = warped_alpha[..., np.newaxis]
    
    # Formule standard de l'alpha blending : Result = (1 - alpha) * BG + alpha * FG
    out_frame = out_frame * (1.0 - warped_alpha_3d) + warped_rgb * warped_alpha_3d

    return out_frame.astype(np.uint8)