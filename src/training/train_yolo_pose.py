"""
train_yolo_pose.py
------------------
Script d'entraînement pour YOLO-Pose appliqué à la topologie d'un terrain de basket.

Particularités :
  - Augmentations géométriques restreintes pour ne pas briser la topologie (Mosaic réduit).
  - Gestion du MLflow et lancement automatique de TensorBoard en arrière-plan.
  - Support de la reprise d'entraînement (--resume).
"""

import os
# Configuration MLflow en amont
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

import argparse
import logging
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (Dataclass)
# ---------------------------------------------------------------------------
@dataclass
class PoseTrainConfig:
    """Configuration hyper-spécifique pour l'apprentissage d'une topologie rigide (terrain)."""
    
    # Fichiers et architecture
    yaml_path: Path     = Path("data/datasets/yolo_pose_court/data.yaml")
    runs_base_dir: Path = Path("models/runs/keypoint_detection")
    model_name: str     = "yolo11m-pose"
    epochs: int         = 1000
    imgsz: int          = 800
    device: int         = 0
    workers: int        = 4

    # Hyperparamètres d'entraînement
    batch: int          = 8       # ↑ Gradients stables sur petit dataset
    lr0: float          = 0.001   # ↓ Plus stable avec AdamW
    lrf: float          = 0.01    # Descend très bas pour affiner
    warmup_epochs: int  = 5
    weight_decay: float = 0.001
    patience: int       = 50
    cos_lr: bool        = True

    # Poids des erreurs (Loss)
    pose: float         = 15.0    # Focus absolu sur la précision des points
    kobj: float         = 2.0     # Tolérance pour la détection hors-cadre

    # Augmentations Géométriques (Adaptées au Terrain)
    mosaic: float       = 0.15    # ↓↓ Préserve la topologie globale
    close_mosaic: int   = 30
    perspective: float  = 0.001   # Simule angles de caméras
    degrees: float      = 8.0     # Horizon imparfait
    shear: float        = 2.0     # Distorsion grand-angle
    scale: float        = 0.35    # ↓ Évite le sur-zoom destructif
    translate: float    = 0.15    # Recadrage (points hors-cadre)
    fliplr: float       = 0.5     # Symétrie FIBA
    flipud: float       = 0.0     # Un terrain n'est jamais au plafond

    # Augmentations Photométriques
    hsv_h: float        = 0.02
    hsv_s: float        = 0.3
    hsv_v: float        = 0.4
    erasing: float      = 0.35    # Simule l'occultation par les joueurs
    mixup: float        = 0.05
    copy_paste: float   = 0.0

    # Sorties standards
    save: bool          = True
    plots: bool         = True
    verbose: bool       = True


# ===========================================================================
# UTILITAIRES : TensorBoard & Callbacks
# ===========================================================================

def launch_tensorboard(log_dir: Path, port: int = 6006) -> Optional[subprocess.Popen]:
    """Lance TensorBoard en arrière-plan et l'ouvre dans le navigateur."""
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard", "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        url = f"http://localhost:{port}"
        logger.info(f"TensorBoard disponible → {url}")
        webbrowser.open(url)
        return process
    except Exception as exc:
        logger.warning(f"Échec lancement TensorBoard auto : {exc}")
        return None

def teardown_tensorboard(tb_proc: Optional[subprocess.Popen]) -> None:
    """Ferme proprement le processus TensorBoard."""
    if tb_proc:
        logger.info("Arrêt de TensorBoard…")
        tb_proc.terminate()
        try:
            tb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb_proc.kill()

def build_yolo_callbacks(run_dir: Path) -> dict:
    """Crée un SummaryWriter et attache les événements YOLO à TensorBoard."""
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    def on_train_epoch_end(trainer) -> None:
        if hasattr(trainer, "loss_items"):
            for name, val in zip(["train/box_loss", "train/pose_loss", "train/kobj_loss"], trainer.loss_items):
                writer.add_scalar(name, val.item(), trainer.epoch)
        if hasattr(trainer, "lr"):
            for idx, lr_val in enumerate(trainer.lr.values()):
                writer.add_scalar(f"train/lr_pg{idx}", lr_val, trainer.epoch)

    def on_fit_epoch_end(trainer) -> None:
        if hasattr(trainer, "metrics"):
            for key, value in trainer.metrics.items():
                writer.add_scalar(f"val/{key}", value, trainer.epoch)

    def on_train_end(trainer) -> None:
        writer.flush()
        writer.close()

    return {
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end":   on_fit_epoch_end,
        "on_train_end":       on_train_end,
    }


# ===========================================================================
# MOTEUR D'ENTRAÎNEMENT
# ===========================================================================

def resolve_run_dir(base_dir: Path, model_name: str, epochs: int) -> tuple[str, Path]:
    """Génère un dossier de run incrémental."""
    base = f"{model_name}_{epochs}ep"
    version = 1
    while (base_dir / f"{base}_v{version}").exists():
        version += 1
    run_name = f"{base}_v{version}"
    return run_name, base_dir / run_name


def train_yolo_pose(config: PoseTrainConfig, resume_path: Optional[Path] = None, auto_tb: bool = False) -> None:
    """Orchestre le lancement de l'entraînement YOLO-Pose."""
    from ultralytics import YOLO

    if not config.yaml_path.exists():
        logger.error(f"Dataset introuvable : {config.yaml_path.resolve()}")
        return

    # --- Mode REPRISE ---
    if resume_path:
        if not resume_path.exists():
            logger.error(f"Fichier de reprise introuvable : {resume_path}")
            return
            
        run_dir = resume_path.parent.parent
        logger.info(f"=== Reprise de l'entraînement depuis {run_dir.name} ===")
        
        tb_proc = launch_tensorboard(run_dir) if auto_tb else None
        
        model = YOLO(str(resume_path))
        for name, cb in build_yolo_callbacks(run_dir).items():
            model.add_callback(name, cb)
            
        model.train(resume=True)
        teardown_tensorboard(tb_proc)
        return

    # --- Mode NOUVEL ENTRAÎNEMENT ---
    run_name, run_dir = resolve_run_dir(config.runs_base_dir, config.model_name, config.epochs)
    
    logger.info(f"=== Nouvel entraînement YOLO-Pose : {run_name} ===")
    logger.info(f"Dataset : {config.yaml_path} | ImgSize : {config.imgsz}px | Batch : {config.batch}")
    
    tb_proc = launch_tensorboard(run_dir) if auto_tb else None
    
    model = YOLO(f"{config.model_name}.pt")
    for name, cb in build_yolo_callbacks(run_dir).items():
        model.add_callback(name, cb)

    # Conversion de la dataclass en dictionnaire pour les arguments Ultralytics
    # On retire les clés qui ne sont pas des arguments natifs de model.train()
    train_kwargs = asdict(config)
    train_kwargs.pop("yaml_path")
    train_kwargs.pop("runs_base_dir")
    train_kwargs.pop("model_name")
    
    # Ajout des arguments obligatoires
    train_kwargs.update({
        "data": str(config.yaml_path),
        "project": str(config.runs_base_dir.resolve()),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "amp": True,
    })

    model.train(**train_kwargs)
    
    teardown_tensorboard(tb_proc)
    logger.info(f"=== Entraînement terminé : {run_dir.name} ===")


# ===========================================================================
# Point d'entrée & CLI
# ===========================================================================

def main() -> None:
    # La config par défaut est instanciée ici
    default_cfg = PoseTrainConfig()
    
    parser = argparse.ArgumentParser(description="Entraîne YOLO-Pose sur les keypoints de terrain.")
    parser.add_argument("--yaml", type=Path, default=default_cfg.yaml_path, help="Chemin vers data.yaml")
    parser.add_argument("--model", type=str, default=default_cfg.model_name, help="Modèle YOLO-Pose (ex: yolo11m-pose)")
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs, help="Nombre d'epochs")
    parser.add_argument("--resume", type=Path, default=None, help="Chemin vers last.pt pour reprendre")
    parser.add_argument("--tensorboard", action="store_true", help="Lance TensorBoard dans le navigateur")
    
    args = parser.parse_args()

    # Mise à jour de la config avec les arguments de la ligne de commande
    cfg = PoseTrainConfig(
        yaml_path=args.yaml,
        model_name=args.model,
        epochs=args.epochs
    )

    train_yolo_pose(config=cfg, resume_path=args.resume, auto_tb=args.tensorboard)

if __name__ == "__main__":
    main()