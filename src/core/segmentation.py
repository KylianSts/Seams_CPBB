"""
segmentation.py
---------------
Module d'extraction de masques au pixel près via SAM 2.1.
Gère exclusivement l'inférence par Deep Learning.
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

logger = logging.getLogger(__name__)

# --- Alias de type ---
BBox = Tuple[float, float, float, float]


@dataclass
class SegmentationConfig:
    """Configuration matérielle et métier du modèle SAM 2."""
    checkpoint_path: Path = Path("models/weights/sam2.1_hiera_small.pt")
    config_name: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    device: int = 0
    
    net_margin_ratio: float = 0.03 

    @property
    def torch_device_str(self) -> str:
        return f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# INITIALISATION ET ENCODAGE
# ===========================================================================

def load_segmentation_model(config: SegmentationConfig) -> SAM2ImagePredictor:
    """Charge le modèle SAM 2 en mémoire vidéo (VRAM)."""
    logger.info("Chargement du modèle de segmentation (SAM 2)...")
    
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Poids SAM introuvables : {config.checkpoint_path}")

    if "cuda" in config.torch_device_str and torch.cuda.get_device_properties(config.device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(config.config_name, str(config.checkpoint_path), device=config.torch_device_str)
    predictor = SAM2ImagePredictor(sam2_model)
    
    logger.info("Modèle SAM 2 prêt.")
    return predictor


def encode_frame(predictor: SAM2ImagePredictor, frame_bgr: np.ndarray, config: SegmentationConfig) -> None:
    """
    Calcule les features de l'image.
    Doit être appelé une seule fois par frame avant de soumettre des requêtes de masques.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    device_type = "cuda" if "cuda" in config.torch_device_str else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        predictor.set_image(frame_rgb)


# ===========================================================================
# INFÉRENCE DES MASQUES
# ===========================================================================

def get_players_masks(
    predictor: SAM2ImagePredictor,
    player_boxes: List[BBox], 
    config: SegmentationConfig
) -> List[np.ndarray]:
    """
    Retourne les masques booléens des joueurs basés sur leurs Bounding Boxes.
    Nécessite que `encode_frame` ait été appelé au préalable.
    """
    if not player_boxes:
        return []

    boxes_array = np.array(player_boxes, dtype=np.float32)
    device_type = "cuda" if "cuda" in config.torch_device_str else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(
            point_coords=None, 
            point_labels=None,
            box=boxes_array,
            multimask_output=False
        )
    
    return [mask.squeeze().astype(bool) for mask in masks]


def get_net_mask(
    predictor: SAM2ImagePredictor, 
    hoop_box: BBox, 
    img_width: int, 
    img_height: int, 
    config: SegmentationConfig
) -> Optional[np.ndarray]:
    """Extrait le masque du filet en extrapolant la zone sous l'arceau."""
    if not hoop_box:
        return None

    x1, y1, x2, y2 = hoop_box
    w, h = x2 - x1, y2 - y1
    
    nx1 = max(0, x1 - w * config.net_margin_ratio)
    ny1 = max(0, y1 - h * config.net_margin_ratio)
    nx2 = min(img_width, x2 + w * config.net_margin_ratio)
    ny2 = min(img_height, y2 + h * (config.net_margin_ratio * 5)) 

    box_array = np.array([[nx1, ny1, nx2, ny2]], dtype=np.float32)
    device_type = "cuda" if "cuda" in config.torch_device_str else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(
            point_coords=None, 
            point_labels=None,
            box=box_array,
            multimask_output=False
        )
        
    return masks[0].squeeze().astype(bool)