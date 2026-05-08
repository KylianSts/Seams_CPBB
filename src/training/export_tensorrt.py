"""
export_tensorrt.py
------------------
Module de compilation matérielle (TensorRT).
Permet d'exporter les modèles PyTorch (.pth / .pt) vers des moteurs TensorRT (.engine)
optimisés pour le GPU cible, avec sélection de la précision (FP32 / FP16).
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Ajout du dossier racine au PATH ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.detect_objects import DetectionConfig, load_object_detector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("ExportTRT")


@dataclass
class ExportConfig:
    """Configuration de l'export TensorRT."""
    model_type: str            # 'rfdetr' ou 'yolo'
    precision: str             # 'fp32' ou 'fp16'
    resolution: int            # 1280 (RF-DETR) ou 640 (YOLO)
    weights_path: Path
    output_dir: Path = Path("models/weights")

    @property
    def engine_name(self) -> str:
        return f"{self.model_type}_{self.resolution}_{self.precision}.engine"
        
    @property
    def onnx_name(self) -> str:
        return f"{self.model_type}_{self.resolution}_static.onnx"


# ===========================================================================
# 1. EXPORT YOLO (Natif Ultralytics)
# ===========================================================================

def export_yolo(cfg: ExportConfig):
    """
    Exporte un modèle YOLO via l'API officielle Ultralytics.
    Gère nativement le FP16 et TensorRT sans avoir besoin de sous-processus.
    """
    logger.info(f"Démarrage de l'export YOLO-Pose vers TensorRT ({cfg.precision.upper()})...")
    from ultralytics import YOLO
    
    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {cfg.weights_path}")

    model = YOLO(str(cfg.weights_path))
    
    exported_path = model.export(
        format="engine",
        imgsz=cfg.resolution,
        half=(cfg.precision == "fp16"),
        device=0,
        workspace=2 
    )
    
    logger.info(f"✅ Export YOLO terminé : {exported_path}")


# ===========================================================================
# 2. EXPORT RF-DETR (Interpolation + ONNX + Sous-processus)
# ===========================================================================

def _resize_position_embeddings(checkpoint_path: Path, target_resolution: int) -> Path:
    """
    Ouvre le checkpoint, étire la grille de position mathématiquement
    et sauvegarde un checkpoint temporaire interpolé.
    """
    target_patches = (target_resolution // 16) ** 2 + 1
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    
    pos_key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
    if pos_key not in state_dict:
        return checkpoint_path
        
    pos_emb = state_dict[pos_key]
    
    if pos_emb.shape[1] == target_patches:
        return checkpoint_path 
        
    logger.info(f"Adaptation des poids : Interpolation de la grille vers {target_resolution}px...")
    
    cls_token = pos_emb[:, :1, :]
    patch_emb = pos_emb[:, 1:, :]
    
    current_dim = int(np.sqrt(patch_emb.shape[1]))
    patch_emb = patch_emb.reshape(1, current_dim, current_dim, 384).permute(0, 3, 1, 2)
    
    new_dim = target_resolution // 16
    patch_emb_resized = F.interpolate(patch_emb, size=(new_dim, new_dim), mode="bicubic", align_corners=False)
    
    patch_emb_resized = patch_emb_resized.permute(0, 2, 3, 1).reshape(1, new_dim * new_dim, 384)
    
    new_pos_emb = torch.cat([cls_token, patch_emb_resized], dim=1)
    state_dict[pos_key] = new_pos_emb
    
    new_ckpt_path = checkpoint_path.parent / f"{checkpoint_path.stem}_{target_resolution}px_interpolated.pth"
    torch.save(ckpt, new_ckpt_path)
    
    return new_ckpt_path


def _patch_interpolate():
    original = F.interpolate
    def patched(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        return original(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=False)
    return original, patched


def export_rfdetr_to_onnx(model, onnx_path: Path, resolution: int):
    logger.info(f"[1/2] Exportation ONNX de RF-DETR (Résolution FIXE : {resolution}px)...")

    raw_model = model.model.model if hasattr(model, 'model') and hasattr(model.model, 'model') else (model.model if hasattr(model, 'model') else model)

    class InferenceWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out["pred_logits"], out["pred_boxes"]

    model_to_export = InferenceWrapper(raw_model).cuda().eval()
    dummy_input = torch.randn(1, 3, resolution, resolution).cuda()

    original_interpolate, patched_interpolate = _patch_interpolate()
    torch.nn.functional.interpolate = patched_interpolate

    try:
        with torch.no_grad():
            torch.onnx.export(
                model_to_export, dummy_input, str(onnx_path),
                dynamo=False, export_params=True, opset_version=16, do_constant_folding=True,
                input_names=['images'], output_names=['logits', 'boxes']
            )
    finally:
        torch.nn.functional.interpolate = original_interpolate

    logger.info(f"✅ Fichier ONNX généré : {onnx_path.name}")


def compile_trt_subprocess(onnx_path: Path, engine_path: Path, precision: str):
    logger.info(f"[2/2] Compilation TensorRT ({precision.upper()}) via sous-processus isolé...")

    trt_script = f"""
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024 ** 3))

# --- GESTION DYNAMIQUE DE LA PRÉCISION ---
if "{precision}" == "fp16" and builder.platform_has_fast_fp16:
    print("Activation des optimisations FP16 matérielles...")
    config.set_flag(trt.BuilderFlag.FP16)
else:
    print("Compilation en FP32 (Précision maximale pour Transformer)...")

with open(r"{str(onnx_path)}", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Echec du parsing ONNX")

print("Optimisation des kernels CUDA en cours (cela peut prendre quelques minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("build_serialized_network a retourne None")

with open(r"{str(engine_path)}", "wb") as f:
    f.write(serialized_engine)
"""

    result = subprocess.run([sys.executable, "-c", trt_script], capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Le sous-processus TensorRT a échoué (code {result.returncode})")

    logger.info(f"✅ Moteur TensorRT sauvegardé : {engine_path}")


def export_rfdetr(cfg: ExportConfig):
    # 1. Retaillage des poids (Le fameux correctif !)
    adapted_weights_path = _resize_position_embeddings(cfg.weights_path, cfg.resolution)

    # 2. Chargement PyTorch
    det_cfg = DetectionConfig(resolution=cfg.resolution, checkpoint_path=adapted_weights_path)
    model = load_object_detector(det_cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = cfg.output_dir / cfg.onnx_name
    engine_file = cfg.output_dir / cfg.engine_name

    # 3. Export ONNX
    export_rfdetr_to_onnx(model, onnx_file, cfg.resolution)

    del model
    torch.cuda.empty_cache()

    # 4. Compilation TensorRT
    compile_trt_subprocess(onnx_file, engine_file, cfg.precision)
    
    if onnx_file.exists():
        onnx_file.unlink()


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outil d'exportation TensorRT pour les modèles du pipeline.")
    
    parser.add_argument("--model", type=str, choices=["rfdetr", "yolo"], required=True, 
                        help="Modèle à exporter : 'rfdetr' ou 'yolo'")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Chemin vers le fichier source (.pth ou .pt)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16",
                        help="Précision cible (fp32 pour rfdetr, fp16 pour yolo)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Résolution (Par défaut : 1280 pour rfdetr, 640 pour yolo)")
    
    args = parser.parse_args()

    target_res = args.resolution or (1280 if args.model == "rfdetr" else 640)

    cfg = ExportConfig(
        model_type=args.model,
        precision=args.precision,
        resolution=target_res,
        weights_path=Path(args.weights)
    )

    logger.info(f"=== DÉBUT DE LA COMPILATION MATÉRIELLE ===")
    logger.info(f"Cible    : {cfg.model_type.upper()}")
    logger.info(f"Précision: {cfg.precision.upper()}")
    logger.info(f"Taille   : {cfg.resolution}px")

    if cfg.model_type == "rfdetr":
        if cfg.precision == "fp16":
            logger.warning("ATTENTION : Le FP16 dégrade fortement RF-DETR. FP32 est recommandé.")
        export_rfdetr(cfg)
    elif cfg.model_type == "yolo":
        export_yolo(cfg)

    logger.info(f"=== TERMINÉ ! ===")