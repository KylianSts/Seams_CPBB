"""
detect_objects.py
-----------------
Module d'inférence spatiale (Object Detection).
Gère le chargement hybride : TensorRT 10 (.engine) avec Letterboxing mathématique
ou PyTorch (.pth) en fallback.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# --- Alias de type ---
BBox      = Tuple[float, float, float, float]
Detection = Tuple[float, float, float, float, float]  # (x1, y1, x2, y2, conf)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class DetectionConfig:
    """Configuration centralisée du détecteur d'objets."""
    checkpoint_path: Path  = Path("models/weights/rfdetr_1280_fp32.engine")
    model_size:      str   = "medium"
    resolution:      int   = 1280

    class_id_player:   int = 0
    class_id_ball:     int = 1
    class_id_referee:  int = 2
    class_id_hoop:     int = 3

    conf_player:   float = 0.40
    conf_ball:     float = 0.40
    conf_referee:  float = 0.40
    conf_hoop:     float = 0.30

    @property
    def min_threshold(self) -> float:
        """Seuil plancher pour le filtrage initial (avant dispatch par classe)."""
        return min(self.conf_player, self.conf_ball, self.conf_referee, self.conf_hoop)


@dataclass
class DetectionResult:
    """Conteneur des détections triées par classe et par confiance décroissante."""
    players:  List[Detection] = field(default_factory=list)
    referees: List[Detection] = field(default_factory=list)
    hoops:    List[Detection] = field(default_factory=list)
    ball:     List[Detection] = field(default_factory=list)


# ===========================================================================
# MOTEUR TENSORRT 10 (HAUTE PERFORMANCE)
# ===========================================================================

class TRTDetector:
    """
    Encapsule un moteur TensorRT 10 pour l'inférence RF-DETR.
    Gère le Letterboxing, la normalisation et la synchronisation GPU.
    """

    def __init__(self, engine_path: Path, resolution: int) -> None:
        import tensorrt as trt

        self.resolution = resolution
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        logger.info(f"Initialisation du moteur TensorRT : {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self._trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Récupération dynamique des types et formes de sortie
        def _trt_to_torch(trt_type) -> torch.dtype:
            mapping = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF:  torch.float16,
                trt.DataType.INT32: torch.int32,
            }
            return mapping.get(trt_type, torch.float32)

        self.input_dtype = _trt_to_torch(self.engine.get_tensor_dtype("images"))

        shape_logits = self.engine.get_tensor_shape("logits")
        shape_boxes  = self.engine.get_tensor_shape("boxes")
        dtype_logits = _trt_to_torch(self.engine.get_tensor_dtype("logits"))
        dtype_boxes  = _trt_to_torch(self.engine.get_tensor_dtype("boxes"))

        # Buffers GPU pré-alloués (évite les allocations dans la boucle chaude)
        self._output_logits = torch.empty(tuple(shape_logits), device="cuda", dtype=dtype_logits).contiguous()
        self._output_boxes  = torch.empty(tuple(shape_boxes),  device="cuda", dtype=dtype_boxes).contiguous()

        logger.info("Moteur TensorRT prêt.")

    def predict(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, float, int, int]:
        """
        Exécute l'inférence sur une frame BGR.

        Returns:
            probs  : Tensor (N, C) — probabilités par classe après sigmoid.
            boxes  : Tensor (N, 4) — boîtes normalisées (cx, cy, w, h) dans l'espace paddé.
            scale  : Facteur de redimensionnement appliqué à l'image source.
            dw     : Padding horizontal (pixels ajoutés à gauche).
            dh     : Padding vertical   (pixels ajoutés en haut).
        """
        import tensorrt as trt

        # 1. Letterboxing strict (conservation des proportions)
        h_src, w_src = frame.shape[:2]
        scale = min(self.resolution / h_src, self.resolution / w_src)
        new_w = int(w_src * scale)
        new_h = int(h_src * scale)

        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_padded  = np.full((self.resolution, self.resolution, 3), 114, dtype=np.uint8)

        dw = (self.resolution - new_w) // 2
        dh = (self.resolution - new_h) // 2
        img_padded[dh:dh + new_h, dw:dw + new_w] = img_resized

        # 2. Mise en forme CHW + alignement mémoire contiguë
        img = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # 3. Transfert GPU avec cast vers le type attendu par le moteur
        torch_dtype = torch.float16 if self.engine.get_tensor_dtype("images") == trt.DataType.HALF else torch.float32
        img_t = torch.from_numpy(img).cuda().to(dtype=torch_dtype).div_(255.0).unsqueeze(0).contiguous()

        # 4. Inférence asynchrone
        self.context.set_input_shape("images", tuple(img_t.shape))
        self.context.set_tensor_address("images",  img_t.data_ptr())
        self.context.set_tensor_address("logits",  self._output_logits.data_ptr())
        self.context.set_tensor_address("boxes",   self._output_boxes.data_ptr())

        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        probs = self._output_logits.float().sigmoid().squeeze(0)
        boxes = self._output_boxes.float().squeeze(0)

        return probs, boxes, scale, dw, dh


# ===========================================================================
# INITIALISATION
# ===========================================================================

def load_object_detector(config: DetectionConfig) -> Union[TRTDetector, object]:
    """Charge le détecteur selon l'extension du checkpoint (.engine ou .pth)."""
    if config.checkpoint_path.suffix == ".engine":
        return TRTDetector(config.checkpoint_path, config.resolution)

    logger.info("Chargement du modèle RF-DETR PyTorch...")
    from rfdetr import RFDETRBase, RFDETRMedium, RFDETRSmall
    architectures = {"small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase}
    model_class = architectures.get(config.model_size.lower(), RFDETRMedium)
    return model_class(pretrain_weights=str(config.checkpoint_path), resolution=config.resolution)


# ===========================================================================
# INFÉRENCE GÉNÉRIQUE
# ===========================================================================

def _unletterbox(
    cx_norm: float, cy_norm: float, w_norm: float, h_norm: float,
    scale: float, dw: int, dh: int, resolution: int,
    w_orig: int, h_orig: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convertit une boîte normalisée (cx, cy, w, h) dans l'espace de l'image paddée
    vers des coordonnées pixel absolues dans l'espace de la vidéo source.

    Opérations :
        1. Dénormalisation → espace paddé (résolution × résolution)
        2. Soustraction du padding → espace image redimensionnée
        3. Division par le scale → espace image originale

    Returns:
        (x1, y1, x2, y2) clippées sur les dimensions originales, ou None si dégénérée.
    """
    cx_pad = cx_norm * resolution
    cy_pad = cy_norm * resolution
    w_pad  = w_norm  * resolution
    h_pad  = h_norm  * resolution

    x1 = (cx_pad - w_pad / 2.0 - dw) / scale
    y1 = (cy_pad - h_pad / 2.0 - dh) / scale
    x2 = (cx_pad + w_pad / 2.0 - dw) / scale
    y2 = (cy_pad + h_pad / 2.0 - dh) / scale

    x1 = max(0.0, min(float(x1), float(w_orig)))
    y1 = max(0.0, min(float(y1), float(h_orig)))
    x2 = max(0.0, min(float(x2), float(w_orig)))
    y2 = max(0.0, min(float(y2), float(h_orig)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def run_object_detection(
    model: Union[TRTDetector, object],
    frame: np.ndarray,
    config: DetectionConfig,
    ignore_humans: bool = False
) -> DetectionResult:
    """
    Exécute la détection d'objets et dispatch les résultats par classe.

    Args:
        model:         Instance TRTDetector ou modèle PyTorch RF-DETR.
        frame:         Image BGR (H, W, 3).
        config:        Paramètres de détection.
        ignore_humans: Si True, ignore les joueurs et arbitres (utile pour les passes AR).

    Returns:
        DetectionResult avec les listes triées par confiance décroissante.
    """
    result = DetectionResult()
    if model is None or frame is None:
        return result

    h_orig, w_orig = frame.shape[:2]

    try:
        if isinstance(model, TRTDetector):

            # --- Inférence TensorRT ---
            probs, boxes, scale, dw, dh = model.predict(frame)

            scores, labels = probs.max(dim=-1)
            mask = scores > config.min_threshold

            filt_scores = scores[mask].cpu().numpy()
            filt_labels = labels[mask].cpu().numpy()
            filt_boxes  = boxes[mask].cpu().numpy()

            for i in range(len(filt_scores)):
                label = int(filt_labels[i])
                score = float(filt_scores[i])

                if ignore_humans and label in (config.class_id_player, config.class_id_referee):
                    continue

                cx, cy, w, h = filt_boxes[i]
                coords = _unletterbox(cx, cy, w, h, scale, dw, dh, model.resolution, w_orig, h_orig)
                if coords is None:
                    continue

                x1, y1, x2, y2 = coords
                det: Detection = (x1, y1, x2, y2, score)

                if   label == config.class_id_player  and score >= config.conf_player:
                    result.players.append(det)
                elif label == config.class_id_referee and score >= config.conf_referee:
                    result.referees.append(det)
                elif label == config.class_id_ball    and score >= config.conf_ball:
                    result.ball.append(det)
                elif label == config.class_id_hoop    and score >= config.conf_hoop:
                    result.hoops.append(det)

        else:
            # --- Inférence PyTorch RF-DETR ---
            preds = model.predict(frame, threshold=config.min_threshold)
            dets  = preds[0] if isinstance(preds, list) else preds

            if hasattr(dets, "xyxy") and len(dets.xyxy) > 0:
                for i, box in enumerate(dets.xyxy):
                    cid  = int(dets.class_id[i])
                    conf = float(dets.confidence[i])

                    if ignore_humans and cid in (config.class_id_player, config.class_id_referee):
                        continue

                    x1, y1, x2, y2 = map(float, box)
                    det: Detection = (x1, y1, x2, y2, conf)

                    if   cid == config.class_id_player  and conf >= config.conf_player:
                        result.players.append(det)
                    elif cid == config.class_id_referee and conf >= config.conf_referee:
                        result.referees.append(det)
                    elif cid == config.class_id_ball    and conf >= config.conf_ball:
                        result.ball.append(det)
                    elif cid == config.class_id_hoop    and conf >= config.conf_hoop:
                        result.hoops.append(det)

        # Tri final par confiance décroissante
        result.players.sort( key=lambda d: d[4], reverse=True)
        result.referees.sort(key=lambda d: d[4], reverse=True)
        result.hoops.sort(   key=lambda d: d[4], reverse=True)
        result.ball.sort(    key=lambda d: d[4], reverse=True)

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence : {e}", exc_info=True)

    return result