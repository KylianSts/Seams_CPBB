"""
segmentation.py
---------------
Module d'extraction de masques au pixel près (SAM 2).

Workflow optimisé :
  1. Chargement du modèle (1 fois au démarrage).
  2. Encodage de l'image (1 fois par frame) -> Étape la plus lourde.
  3. Requêtes de masques via Bounding Boxes (Joueurs ou Panier) -> Instantané.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SegmentationConfig:
    """Configuration du modèle SAM 2."""
    checkpoint_path: Path = Path("models/weights/sam2.1_hiera_small.pt")
    config_name: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    device: int = 0
    
    # Marge d'agrandissement de la BBox du panier pour être sûr de capturer le filet en dessous
    net_margin_ratio: float = 0.03 

    @property
    def torch_device_str(self) -> str:
        return f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_segmentation_model(config: SegmentationConfig) -> SAM2ImagePredictor:
    """
    Charge le modèle SAM 2 en VRAM.
    À appeler une seule fois dans run_pipeline.py.
    """
    logger.info("Chargement du modèle de segmentation (SAM 2)...")
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Poids SAM introuvables : {config.checkpoint_path}")

    # Optimisation PyTorch pour les GPU modernes (Séries RTX 3000/4000)
    if "cuda" in config.torch_device_str and torch.cuda.get_device_properties(config.device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(config.config_name, str(config.checkpoint_path), device=config.torch_device_str)
    predictor = SAM2ImagePredictor(sam2_model)
    
    logger.info("Modèle SAM 2 prêt.")
    return predictor


# ===========================================================================
# ENCODAGE DE L'IMAGE (Le goulot d'étranglement)
# ===========================================================================

def encode_frame(predictor: SAM2ImagePredictor, frame_bgr: np.ndarray, config: SegmentationConfig) -> None:
    """
    Convertit l'image en RGB et calcule ses 'features'. 
    À appeler strictement UNE SEULE FOIS par frame, avant de demander des masques.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Utilisation d'autocast pour accélérer massivement l'encodage sur GPU
    with torch.autocast(device_type="cuda" if "cuda" in config.torch_device_str else "cpu", dtype=torch.bfloat16):
        predictor.set_image(frame_rgb)


# ===========================================================================
# REQUÊTES DE MASQUES (Rapide)
# ===========================================================================

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
# (Assure-toi d'importer SAM2ImagePredictor si ce n'est pas déjà fait)

def get_players_masks(
    player_boxes: List[Tuple[float, float, float, float]], 
    config: SegmentationConfig,
    predictor: Optional['SAM2ImagePredictor'] = None, 
    method: str = "sam",
    frame_shape: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    Retourne une liste de masques booléens pour chaque joueur.
    
    Args:
        method: "sam" (précis mais lent) ou "ellipse" (approximation ultra-rapide).
        frame_shape: (Hauteur, Largeur) requis uniquement pour la méthode "ellipse".
    """
    if not player_boxes:
        return []

    # ==========================================
    # MÉTHODE 1 : L'APPROXIMATION GÉOMÉTRIQUE
    # ==========================================
    if method == "ellipse":
        if frame_shape is None:
            raise ValueError("L'argument 'frame_shape' (H, W) est requis pour la méthode ellipse.")
            
        h_img, w_img = frame_shape[:2]
        masks = []
        
        for box in player_boxes:
            x1, y1, x2, y2 = box
            
            # 1. Création d'une toile noire (0)
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            
            # 2. Calcul des paramètres de l'ellipse
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # ASTUCE PRO : Un humain est plus fin qu'une BBox globale (qui inclut les bras/jambes écartées).
            # On réduit légèrement la largeur de l'ellipse (ex: 80% de la largeur de la bbox)
            # pour éviter d'effacer trop de logo sur les côtés.
            width_radius = int(((x2 - x1) / 2) * 0.80) 
            height_radius = int((y2 - y1) / 2)
            axes = (width_radius, height_radius)
            
            # 3. Dessin de l'ellipse remplie en blanc (1)
            cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
            
            # 4. Conversion en booléen pour correspondre au format de sortie de SAM
            masks.append(mask.astype(bool))
            
        return masks

    # ==========================================
    # MÉTHODE 2 : LA SEGMENTATION PROFONDE (SAM)
    # ==========================================
    elif method == "sam":
        if predictor is None:
            raise ValueError("L'objet 'predictor' SAM doit être fourni pour la méthode 'sam'.")
            
        boxes_array = np.array(player_boxes, dtype=np.float32)

        with torch.autocast(device_type="cuda" if "cuda" in config.torch_device_str else "cpu", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=None, 
                point_labels=None,
                box=boxes_array,
                multimask_output=False
            )
        
        return [mask.squeeze().astype(bool) for mask in masks]
        
    else:
        raise ValueError(f"Méthode de masque inconnue : {method}")


def get_net_mask(
    predictor: SAM2ImagePredictor, 
    hoop_box: Tuple[float, float, float, float], 
    img_width: int, 
    img_height: int, 
    config: SegmentationConfig
) -> Optional[np.ndarray]:
    """
    Prend la boîte de l'arceau, l'agrandit légèrement vers le bas pour inclure le filet,
    et demande à SAM de le détourer.
    """
    if not hoop_box:
        return None

    x1, y1, x2, y2 = hoop_box
    w, h = x2 - x1, y2 - y1
    
    # On agrandit la boîte (Surtout vers le bas, où se trouve le filet)
    nx1 = max(0, x1 - w * config.net_margin_ratio)
    ny1 = max(0, y1 - h * config.net_margin_ratio)
    nx2 = min(img_width, x2 + w * config.net_margin_ratio)
    
    # On descend beaucoup plus bas pour être sûr de choper tout le filet déformé
    ny2 = min(img_height, y2 + h * (config.net_margin_ratio * 5)) 

    box_array = np.array([[nx1, ny1, nx2, ny2]], dtype=np.float32)

    with torch.autocast(device_type="cuda" if "cuda" in config.torch_device_str else "cpu", dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(
            point_coords=None, 
            point_labels=None,
            box=box_array,
            multimask_output=False
        )
        
    return masks[0].squeeze().astype(bool)