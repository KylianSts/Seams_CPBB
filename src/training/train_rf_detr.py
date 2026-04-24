"""
train_rf_detr.py
----------------
Script d'entraînement industriel pour RF-DETR.
Gère la création d'un environnement de run unique, l'enregistrement des métriques 
(TensorBoard), un scheduler LR personnalisé (Warmup + Cosine), et les augmentations 
de données massives pour la détection de petits objets (ballon de basket).

Workflow :
  Étape 1 — Résolution de la configuration et création du dossier de Run (anti-écrasement).
  Étape 2 — Génération des rapports de traçabilité (config.json, README.md).
  Étape 3 — Préparation des Callbacks (TensorBoard, Tqdm, LR Scheduler).
  Étape 4 — Lancement de l'entraînement avec fallback de sécurité.
"""

import json
import logging
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (Dataclasses)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Configuration centrale de l'entraînement."""
    # Modèle et Données
    model_size: str = "medium"  # "nano" | "small" | "medium" | "base" | "large"
    resolution: int = 1280
    dataset_dir: Path = Path("data/datasets/coco")
    runs_base_dir: Path = Path("models/runs/object_detection")

    # Hyperparamètres d'entraînement
    epochs: int = 150
    batch_size: int = 2
    grad_accum: int = 16
    learning_rate: float = 3e-5
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    num_workers: int = 8
    label_smoothing: float = 0.1

    # Early Stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.001

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


# ---------------------------------------------------------------------------
# Registre des modèles & Augmentations
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "nano":   {"class": RFDETRNano,   "label": "RFDETRNano"},
    "small":  {"class": RFDETRSmall,  "label": "RFDETRSmall"},
    "medium": {"class": RFDETRMedium, "label": "RFDETRMedium"},
    "base":   {"class": RFDETRBase,   "label": "RFDETRBase"},
    "large":  {"class": RFDETRLarge,  "label": "RFDETRLarge"},
}

SPORT_AUG_CONFIG_V2 = {
    **AUG_AGGRESSIVE,
    "RandomScale": {"scale_limit": (-0.9, 1.0), "p": 0.8},
    "RandomSizedBBoxSafeCrop": {"height": 1280, "width": 1280, "erosion_rate": 0.0, "p": 0.5},
    "MotionBlur": {"blur_limit": (3, 15), "p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.35, "contrast_limit": 0.35, "p": 0.5},
    "HueSaturationValue": {"hue_shift_limit": 15, "sat_shift_limit": 40, "val_shift_limit": 25, "p": 0.45},
    "GaussNoise": {"var_limit": (10.0, 60.0), "p": 0.3},
    "CoarseDropout": {"num_holes_range": (1, 6), "hole_height_range": (10, 50), "hole_width_range": (10, 50), "p": 0.25},
    "ImageCompression": {"quality_range": (50, 95), "p": 0.25},
    "ElasticTransform": {"alpha": 30, "sigma": 5, "p": 0.15},
    "GaussianBlur": {"blur_limit": (3, 7), "p": 0.2},
    "Perspective": {"scale": (0.02, 0.08), "p": 0.2},
}


# ===========================================================================
# UTILITAIRES : Système et Mathématiques
# ===========================================================================

def get_git_commit() -> str:
    """Récupère le commit Git actuel pour la traçabilité."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "N/A"

def compute_lr_factor(epoch: int, config: TrainConfig) -> float:
    """Calcule le multiplicateur LR (Warmup linéaire puis Cosine Decay)."""
    if epoch < config.warmup_epochs:
        return (config.lr_min + (config.learning_rate - config.lr_min) * epoch / max(config.warmup_epochs, 1)) / config.learning_rate
    
    progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = config.lr_min + (config.learning_rate - config.lr_min) * cos_decay
    return lr / config.learning_rate


# ===========================================================================
# GESTION DES DOSSIERS DE RUN & RAPPORTS
# ===========================================================================

def create_run_directory(config: TrainConfig) -> Path:
    """Génère un dossier unique anti-écrasement pour le run actuel."""
    run_name = f"rfdetr-{config.model_size}_{config.resolution}px_{config.epochs}ep"
    version = 1
    while True:
        candidate = config.runs_base_dir / f"{run_name}_v{version}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        version += 1

def export_run_metadata(run_dir: Path, config: TrainConfig, model_label: str) -> None:
    """Génère le config.json et le README.md descriptif du run."""
    metadata = {
        "run_name": run_dir.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": {"size": config.model_size, "architecture": model_label, "resolution": config.resolution},
        "training": {
            "epochs": config.epochs, "batch_size": config.batch_size, 
            "grad_accum": config.grad_accum, "lr": config.learning_rate, 
            "warmup": config.warmup_epochs
        },
        "environment": {"python": sys.version.split()[0], "git_commit": get_git_commit()}
    }
    
    # JSON
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    # README (Simplifié pour la lisibilité de la fonction)
    readme_content = f"""# {metadata['run_name']}
- **Modèle :** {model_label} ({config.resolution}px)
- **Batch Effectif :** {config.effective_batch_size}
- **LR :** {config.learning_rate} (Warmup: {config.warmup_epochs}ep)
- **Démarré le :** {metadata['started_at']} (Commit: {metadata['environment']['git_commit']})
"""
    with open(run_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)


# ===========================================================================
# MOTEUR D'ENTRAÎNEMENT & CALLBACKS
# ===========================================================================

def setup_callbacks(config: TrainConfig, run_dir: Path) -> Tuple[SummaryWriter, tqdm, Callable, Callable, dict]:
    """Prépare TensorBoard, la barre de progression et les fonctions de suivi (callbacks)."""
    tb_writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    pbar = tqdm(total=config.epochs, desc=f"Training {config.model_size}", unit="ep", dynamic_ncols=True)
    
    state = {"best_map": float("-inf"), "best_epoch": 0, "start_time": 0.0}

    def on_train_start(*args, **kwargs):
        state["start_time"] = time.time()

    def on_fit_epoch_end(*args, **kwargs):
        # Extraction sécurisée des métriques
        metrics = args[0] if args and isinstance(args[0], dict) else {}
        metrics.update(kwargs)
        if isinstance(metrics.get("train_logs"), dict):
            metrics.update(metrics["train_logs"])

        epoch = int(metrics.get("epoch", pbar.n))
        loss = metrics.get("train_loss", metrics.get("loss", float("nan")))
        map_val = metrics.get("map", float("nan"))

        # Résolution du mAP si caché dans le JSON de test
        if math.isnan(map_val):
            test_res = metrics.get("test_results_json", {})
            if isinstance(test_res, str):
                try: test_res = json.loads(test_res)
                except json.JSONDecodeError: test_res = {}
            map_val = test_res.get("map", float("nan"))

        # Enregistrement TensorBoard (LR, Loss, mAP)
        lr_actual = config.learning_rate * compute_lr_factor(epoch, config)
        tb_writer.add_scalar("Hyperparams/lr_actual", lr_actual, epoch)
        tb_writer.add_scalar("Loss/train", loss, epoch)
        if not math.isnan(map_val):
            tb_writer.add_scalar("Metrics/mAP@50", map_val, epoch)

        # Suivi du meilleur modèle
        if not math.isnan(map_val) and map_val > state["best_map"]:
            state["best_map"] = map_val
            state["best_epoch"] = epoch
            logger.info(f" ★ Nouveau meilleur mAP : {map_val:.4f} (epoch {epoch})")

        # Mise à jour Tqdm
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss:.4f}", "mAP": f"{map_val:.4f}", "best": f"{state['best_map']:.4f}"})

    return tb_writer, pbar, on_train_start, on_fit_epoch_end, state


def run_training(config: TrainConfig) -> None:
    """Orchestre l'initialisation du modèle, l'environnement et la boucle de fit."""
    model_info = MODEL_REGISTRY.get(config.model_size.lower())
    if not model_info:
        raise ValueError(f"Taille '{config.model_size}' invalide. Choix: {list(MODEL_REGISTRY.keys())}")

    run_dir = create_run_directory(config)
    export_run_metadata(run_dir, config, model_info["label"])

    logger.info(f"=== Nouveau run : {run_dir.name} ===")
    logger.info(f"Modèle: {model_info['label']} | Batch Eff: {config.effective_batch_size} | LR: {config.learning_rate}")
    logger.info(f"TensorBoard: tensorboard --logdir {run_dir / 'logs'}")

    # Préparation Modèle et Callbacks
    model = model_info["class"](resolution=config.resolution)
    tb_writer, pbar, cb_start, cb_epoch, state = setup_callbacks(config, run_dir)
    model.callbacks["on_train_start"].append(cb_start)
    model.callbacks["on_fit_epoch_end"].append(cb_epoch)

    # Paramètres d'entraînement communs
    train_kwargs = {
        "dataset_dir": str(config.dataset_dir),
        "output_dir": str(run_dir),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum,
        "num_workers": config.num_workers,
        "lr": config.learning_rate,
        "pin_memory": True,
        "amp": True,
        "gradient_checkpointing": True,
        "early_stopping": True,
        "early_stopping_patience": config.early_stop_patience,
        "early_stopping_min_delta": config.early_stop_min_delta,
        "aug_config": SPORT_AUG_CONFIG_V2,
        "tensorboard": True,
        "imgsz": config.resolution,        
        "img_size": config.resolution,    
        "resolution": config.resolution,  
    }

    try:
        # Tentative avec Label Smoothing
        model.train(**train_kwargs, label_smoothing=config.label_smoothing)
    except TypeError as e:
        if "label_smoothing" in str(e):
            logger.warning("Label smoothing non supporté par la lib actuelle. Relance sans.")
            model.train(**train_kwargs)
        else:
            raise
    finally:
        pbar.close()
        tb_writer.close()

    elapsed_h = (time.time() - state["start_time"]) / 3600
    logger.info(f"=== Entraînement terminé en {elapsed_h:.2f}h ===")
    logger.info(f"Meilleur mAP : {state['best_map']:.4f} (Epoch {state['best_epoch']})")


# ===========================================================================
# Point d'entrée
# ===========================================================================
def main():
    cfg = TrainConfig()
    
    if not cfg.dataset_dir.exists():
        logger.error(f"Dataset introuvable : {cfg.dataset_dir.resolve()}")
    else:
        run_training(cfg)

if __name__ == "__main__":
    main()