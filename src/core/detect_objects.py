"""
detect_objects.py
-----------------
Module d'inférence spatiale (Object Detection).
Prend une frame vidéo brute et retourne les Bounding Boxes des entités du match 
(Joueurs, Balle, Arbitres, Panier) via le modèle RF-DETR.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rfdetr import RFDETRMedium, RFDETRBase, RFDETRSmall

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    checkpoint_path: Path = Path("models/runs/object_detection/rfdetr-medium_1280px_100ep_v1/checkpoint_best_ema.pth")
    model_size: str = "medium"
    
    # Résolution forcée à 1280
    resolution: int = 1280 
    
    class_id_player: int = 0
    class_id_ball: int = 1
    class_id_referee: int = 2
    class_id_hoop: int = 3
    
    conf_player: float = 0.40
    conf_ball: float = 0.40
    conf_referee: float = 0.40
    conf_hoop: float = 0.30
    
    @property
    def min_threshold(self) -> float:
        return min(self.conf_player, self.conf_ball, self.conf_referee, self.conf_hoop)

@dataclass
class DetectionResult:
    players: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    referees: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    hoops: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    ball: Optional[Tuple[float, float, float, float, float]] = None


# ===========================================================================
# INTERPOLATION DES POIDS (LA MAGIE MATHÉMATIQUE)
# ===========================================================================

def _resize_position_embeddings(checkpoint_path: Path, target_resolution: int) -> Path:
    """
    Ouvre le checkpoint, étire la grille de position (ex: 576 -> 1280) 
    et sauvegarde un nouveau checkpoint pour tromper la librairie de manière propre.
    """
    # Calcul du nombre de patchs nécessaires (Patch size = 16)
    target_patches = (target_resolution // 16) ** 2 + 1
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    
    pos_key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
    if pos_key not in state_dict:
        return checkpoint_path
        
    pos_emb = state_dict[pos_key]
    
    # Si le modèle est déjà à la bonne taille, on ne touche à rien
    if pos_emb.shape[1] == target_patches:
        return checkpoint_path 
        
    logger.info(f"Adaptation des poids : Interpolation de la grille vers {target_resolution}px...")
    
    cls_token = pos_emb[:, :1, :]
    patch_emb = pos_emb[:, 1:, :]
    
    # Transformation 1D vers 2D pour l'interpolation spatiale
    current_dim = int(np.sqrt(patch_emb.shape[1]))
    patch_emb = patch_emb.reshape(1, current_dim, current_dim, 384).permute(0, 3, 1, 2)
    
    # Interpolation Bicubique vers la nouvelle grille (ex: 80x80)
    new_dim = target_resolution // 16
    patch_emb_resized = F.interpolate(patch_emb, size=(new_dim, new_dim), mode="bicubic", align_corners=False)
    
    # Retour au format plat attendu par le Transformer
    patch_emb_resized = patch_emb_resized.permute(0, 2, 3, 1).reshape(1, new_dim * new_dim, 384)
    
    # Fusion et écrasement dans le dictionnaire
    new_pos_emb = torch.cat([cls_token, patch_emb_resized], dim=1)
    state_dict[pos_key] = new_pos_emb
    
    # Sauvegarde du nouveau fichier
    new_ckpt_path = checkpoint_path.parent / f"{checkpoint_path.stem}_{target_resolution}px_interpolated.pth"
    torch.save(ckpt, new_ckpt_path)
    
    return new_ckpt_path


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_object_detector(config: DetectionConfig):
    logger.info(f"Chargement du modèle RF-DETR ({config.model_size}) à {config.resolution}px...")
    
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {config.checkpoint_path.resolve()}")
    
    architectures = {
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase
    }
    model_class = architectures.get(config.model_size.lower())
    
    if not model_class:
        raise ValueError(f"Taille de modèle '{config.model_size}' non supportée.")
        
    # 1. On retaille le checkpoint automatiquement pour qu'il corresponde
    adapted_ckpt_path = _resize_position_embeddings(config.checkpoint_path, config.resolution)
    
    # 2. On instancie la classe normalement
    model = model_class(pretrain_weights=str(adapted_ckpt_path), resolution=config.resolution)
    
    logger.info("Modèle RF-DETR prêt.")
    return model


# ===========================================================================
# INFÉRENCE (Par Frame)
# ===========================================================================

def run_object_detection(model, frame: np.ndarray, config: DetectionConfig) -> DetectionResult:
    result = DetectionResult()
    if model is None or frame is None:
        return result

    try:
        preds = model.predict(frame, threshold=config.min_threshold)
        dets = preds[0] if isinstance(preds, list) else preds
        
        if not hasattr(dets, 'xyxy') or dets.xyxy is None or len(dets.xyxy) == 0:
            return result

        raw_players, raw_balls, raw_referees, raw_hoops = [], [], [], []

        for i, box in enumerate(dets.xyxy):
            x1, y1, x2, y2 = map(float, box)
            conf = float(dets.confidence[i]) if hasattr(dets, 'confidence') else 1.0
            cid = int(dets.class_id[i]) if hasattr(dets, 'class_id') else 0
            
            if cid == config.class_id_player and conf >= config.conf_player:
                raw_players.append((x1, y1, x2, y2, conf))
            elif cid == config.class_id_ball and conf >= config.conf_ball:
                raw_balls.append((x1, y1, x2, y2, conf))
            elif cid == config.class_id_referee and conf >= config.conf_referee:
                raw_referees.append((x1, y1, x2, y2, conf))
            elif cid == config.class_id_hoop and conf >= config.conf_hoop:
                raw_hoops.append((x1, y1, x2, y2, conf))

        result.players = sorted(raw_players, key=lambda d: d[4], reverse=True)
        result.referees = sorted(raw_referees, key=lambda d: d[4], reverse=True)
        result.hoops = sorted(raw_hoops, key=lambda d: d[4], reverse=True)
        
        if raw_balls:
            result.ball = sorted(raw_balls, key=lambda d: d[4], reverse=True)[0]

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence RF-DETR : {e}")

    return result