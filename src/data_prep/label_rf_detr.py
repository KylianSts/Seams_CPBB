"""
label_rf_detr.py
----------------
Pré-labellise automatiquement les images d'une task CVAT non encore annotées,
puis fusionne les prédictions avec le fichier export CVAT existant (s'il existe)
pour produire un JSON prêt à réimporter dans CVAT.

Workflow :
  Étape 1 — Chargement ou création du JSON COCO de base.
  Étape 2 — Identification des images déjà annotées (pour les ignorer).
  Étape 3 — Inférence image par image via RF-DETR.
  Étape 4 — Sauvegarde du JSON fusionné.

Note : Aucune optimisation par batch n'est appliquée ici pour privilégier 
la simplicité d'usage sur un script à exécution ponctuelle.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
from tqdm import tqdm
from rfdetr import RFDETRSmall, RFDETRMedium

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Dataclasses
# ---------------------------------------------------------------------------

TASK_NAME  = "task_8"
MODEL_NAME = "rfdetr-medium_15ep_v1"

@dataclass
class LabelingConfig:
    """Configuration centralisée pour la pré-labellisation."""
    checkpoint_path: Path = Path(f"models/runs/object_detection/{MODEL_NAME}/checkpoint_best_ema.pth")
    input_dir: Path       = Path(f"data/raw/images/object_detection/{TASK_NAME}")
    existing_json: Path   = Path(f"data/annotations/object_detection/{TASK_NAME}/instances_default.json")
    output_json: Path     = Path(f"data/annotations/object_detection/{TASK_NAME}/instances_cvat_pre_labeled.json")
    
    class_names: Tuple[str, ...] = ("Player", "Ball", "Referee", "Hoop")
    confidence_threshold: float  = 0.3
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


# ===========================================================================
# ÉTAPE 1 — Gestion du JSON COCO
# ===========================================================================

def _build_empty_coco_json(class_names: Tuple[str, ...]) -> dict:
    """Crée un JSON COCO vierge (catégories en base 1 pour CVAT)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": ""}
            for i, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }


def load_or_create_coco_json(path: Path, class_names: Tuple[str, ...]) -> dict:
    """Charge le JSON existant, ou en initialise un vierge si introuvable."""
    if path.exists():
        logger.info(f"Annotations existantes chargées : {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Aucun export trouvé ({path.name}). Création d'un JSON vierge.")
    return _build_empty_coco_json(class_names)


def save_coco_json(data: dict, path: Path) -> None:
    """Sauvegarde les données COCO sur le disque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON sauvegardé : {path}")


# ===========================================================================
# ÉTAPE 2 — Analyse de l'existant
# ===========================================================================

def get_already_labeled_filenames(coco_data: dict) -> Set[str]:
    """Identifie les images ayant déjà des annotations (pour ne pas les écraser)."""
    annotated_image_ids = {ann["image_id"] for ann in coco_data.get("annotations", [])}
    return {
        Path(img["file_name"]).name
        for img in coco_data.get("images", [])
        if img["id"] in annotated_image_ids
    }


def get_filename_to_image_id(coco_data: dict) -> Dict[str, int]:
    """Indexe les images existantes {nom: id} pour conserver les IDs CVAT."""
    return {
        Path(img["file_name"]).name: img["id"]
        for img in coco_data.get("images", [])
    }


def get_next_ids(coco_data: dict) -> Tuple[int, int]:
    """Détermine les prochains IDs disponibles pour éviter les conflits."""
    max_img_id = max((img["id"] for img in coco_data.get("images", [])), default=0)
    max_ann_id = max((ann["id"] for ann in coco_data.get("annotations", [])), default=0)
    return max_img_id + 1, max_ann_id + 1


# ===========================================================================
# ÉTAPE 3 — Inférence et Conversion géométrique
# ===========================================================================

def build_coco_annotation(
    ann_id: int,
    image_id: int,
    class_id: int,
    box_xyxy: List[float],
    confidence: float,
) -> dict:
    """
    Convertit une prédiction en annotation COCO valide.
    Note : Conversion (x1, y1, x2, y2) → (x, y, w, h) et classe base 0 → base 1.
    """
    xmin, ymin, xmax, ymax = box_xyxy
    w = xmax - xmin
    h = ymax - ymin

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": class_id + 1,  # RF-DETR (0) → CVAT (1)
        "segmentation": [],
        "area": w * h,
        "bbox": [xmin, ymin, w, h],
        "iscrowd": 0,
        "attributes": {
            "occluded": False,
            "rotation": 0.0,
            "score": round(confidence, 3),
        },
    }


def run_inference_pipeline(
    model: RFDETRMedium,
    coco_data: dict,
    config: LabelingConfig,
) -> Tuple[int, int]:
    """Exécute les prédictions sur les images vierges et met à jour le JSON."""
    
    already_labeled = get_already_labeled_filenames(coco_data)
    filename_to_id  = get_filename_to_image_id(coco_data)
    next_img_id, next_ann_id = get_next_ids(coco_data)

    folder_name = config.input_dir.name
    all_images  = [f for f in config.input_dir.iterdir() if f.suffix.lower() in config.valid_extensions]
    images_to_process = [f for f in all_images if f.name not in already_labeled]

    if not images_to_process:
        logger.info("Toutes les images sont déjà annotées. Aucune action requise.")
        return 0, 0

    logger.info(f"Images à traiter : {len(images_to_process)} / {len(all_images)} dans '{folder_name}'")

    boxes_added = 0
    images_processed = 0

    # Boucle d'inférence avec barre de progression
    with tqdm(total=len(images_to_process), desc="Pré-labellisation", unit="img") as pbar:
        for img_path in images_to_process:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Image illisible ignorée : {img_path.name}")
                pbar.update(1)
                continue

            height, width = image.shape[:2]

            # Enregistrement de l'image dans le dictionnaire COCO
            current_image_id = filename_to_id.get(img_path.name, next_img_id)
            if current_image_id == next_img_id:
                coco_data["images"].append({
                    "id": current_image_id,
                    "width": width,
                    "height": height,
                    "file_name": f"{folder_name}/{img_path.name}",
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0,
                })
                next_img_id += 1

            # Inférence
            preds = model.predict(image, threshold=config.confidence_threshold)
            result = preds[0] if isinstance(preds, list) else preds

            # Ajout des boîtes détectées
            if result.xyxy is not None and len(result.xyxy) > 0:
                for i in range(len(result.xyxy)):
                    ann = build_coco_annotation(
                        ann_id=next_ann_id,
                        image_id=current_image_id,
                        class_id=int(result.class_id[i]),
                        box_xyxy=[float(v) for v in result.xyxy[i]],
                        confidence=float(result.confidence[i]),
                    )
                    coco_data["annotations"].append(ann)
                    next_ann_id += 1
                    boxes_added += 1

            images_processed += 1
            pbar.update(1)

    return images_processed, boxes_added


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main() -> None:
    config = LabelingConfig()
    logger.info(f"=== Auto-Annotation CVAT : {TASK_NAME} ===")

    if not config.checkpoint_path.exists():
        logger.error(f"Modèle introuvable : {config.checkpoint_path}")
        return
    if not config.input_dir.exists():
        logger.error(f"Dossier d'images introuvable : {config.input_dir}")
        return

    # Chargement
    logger.info("--- Étape 1 : Chargement du Modèle et des Données ---")
    try:
        model = RFDETRMedium(pretrain_weights=str(config.checkpoint_path), resolution=1280)
        logger.info("Modèle RF-DETR Medium chargé avec succès.")
    except Exception as e:
        logger.exception(f"Erreur fatale lors du chargement du modèle : {e}")
        return

    coco_data = load_or_create_coco_json(config.existing_json, config.class_names)
    print()

    # Inférence
    logger.info("--- Étape 2 : Inférence ---")
    nb_images, nb_boxes = run_inference_pipeline(model, coco_data, config)
    print()

    # Sauvegarde
    if nb_images > 0:
        logger.info("--- Étape 3 : Sauvegarde ---")
        save_coco_json(coco_data, config.output_json)
        logger.info(f"✓ Terminé : {nb_images} images traitées, {nb_boxes} boîtes générées.")
        logger.info(f"Fichier prêt pour import CVAT : {config.output_json.name}")


if __name__ == "__main__":
    main()