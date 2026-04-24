"""
label_yolo_pose.py
------------------
Pré-labellise automatiquement les images avec un modèle YOLO-Pose.
Génère ou met à jour un fichier JSON au format COCO Keypoints 1.0,
prêt à être importé dans CVAT.

Workflow :
  Étape 1 — Chargement ou création du JSON COCO existant.
  Étape 2 — Identification des images sans annotation.
  Étape 3 — Inférence image par image via YOLO-Pose.
  Étape 4 — Mapping des Keypoints selon les seuils de visibilité (0, 1, 2).
  Étape 5 — Sauvegarde du JSON fusionné.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Dataclasses
# ---------------------------------------------------------------------------
TASK_NAME = "task_4"
MODEL_NAME = "yolo11m-pose_200ep_v1"

@dataclass
class PoseConfig:
    """Configuration centralisée pour l'auto-annotation de points clés."""
    model_path: Path    = Path(f"models/runs/keypoint_detection/{MODEL_NAME}/weights/best.pt")
    input_dir: Path     = Path(f"data/raw/images/keypoint_detection/{TASK_NAME}")
    existing_json: Path = Path(f"data/annotations/keypoint_detection/{TASK_NAME}/person_keypoint_default.json")
    output_json: Path   = Path(f"data/annotations/keypoint_detection/{TASK_NAME}/person_keypoint_default_prelabeled.json")
    
    # Seuils de confiance
    bbox_conf_threshold: float = 0.50  # Confiance minimum pour la bounding box mère
    conf_visible: float        = 0.50  # Au-dessus = Visible (v=2)
    conf_min: float            = 0.20  # Entre MIN et VISIBLE = Occlus (v=1). En-dessous = Absent (v=0).
    
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    # Définition COCO
    category_id: int = 72
    category_name: str = "court_2"
    kpt_names: List[str] = field(default_factory=lambda: [
        "1","2","3","4","5","6","7","8","9","10",
        "11","12","13","14","17","18","19","22","23","24",
        "25","26","27","28","29","30","31","32","33","34","35"
    ])
    skeleton: List[List[int]] = field(default_factory=lambda: [
        [32,31], [3,4], [25,32], [31,26], [3,13], [5,10], [9,14], [17,18], 
        [17,30], [27,34], [24,23], [33,32], [18,19], [4,5], [29,28], [5,6], 
        [4,11], [23,33], [34,33], [1,2], [10,14], [1,17], [19,35], [25,24], 
        [33,25], [26,22], [6,19], [12,11], [35,34], [31,30], [22,27], [4,13], 
        [11,3], [23,32], [2,3], [2,9], [13,12], [7,8]
    ])


# ===========================================================================
# ÉTAPE 1 — Gestion du JSON COCO Keypoints
# ===========================================================================

def load_or_create_coco(json_path: Path, config: PoseConfig) -> Tuple[dict, int, int, Dict[str, int]]:
    """Charge le JSON COCO existant ou initialise un dictionnaire vierge format Keypoints."""
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            max_img_id = max([img['id'] for img in coco_data.get('images', [])], default=0)
            max_ann_id = max([ann['id'] for ann in coco_data.get('annotations', [])], default=0)
            existing_images = {Path(img['file_name']).name: img['id'] for img in coco_data.get('images', [])}
            
            logger.info(f"JSON existant chargé : {len(existing_images)} images indexées.")
            return coco_data, max_img_id, max_ann_id, existing_images
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture du JSON ({e}). Création d'un nouveau.")

    logger.info("Création d'un nouveau dictionnaire COCO Keypoints 1.0.")
    coco_data = {
        "info": {
            "description": "Pre-labeled by YOLO-Pose",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": config.category_id,
            "name": config.category_name,
            "supercategory": "",
            "keypoints": config.kpt_names,
            "skeleton": config.skeleton
        }]
    }
    return coco_data, 0, 0, {}


# ===========================================================================
# ÉTAPE 2 — Pipeline de Pré-labellisation
# ===========================================================================

def run_inference_pipeline(
    pose_model, 
    coco_data: dict, 
    max_img_id: int, 
    max_ann_id: int, 
    existing_images: Dict[str, int], 
    config: PoseConfig
) -> Tuple[int, int]:
    
    all_images = sorted([f for f in config.input_dir.iterdir() if f.suffix.lower() in config.valid_extensions])
    to_process = [f for f in all_images if f.name not in existing_images]

    if not to_process:
        logger.info("Toutes les images ont déjà des annotations. Rien à faire.")
        return 0, 0

    logger.info(f"Lancement de l'inférence sur {len(to_process)} images...")

    nb_images_processed = 0
    nb_anns_added = 0

    with tqdm(total=len(to_process), desc="Pré-labellisation Pose", unit="img") as pbar:
        for img_path in to_process:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Impossible de lire l'image : {img_path.name}")
                pbar.update(1)
                continue

            h, w = image.shape[:2]
            
            # 1. Enregistrement de l'image
            max_img_id += 1
            img_id = max_img_id
            coco_data['images'].append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": img_path.name
            })

            # 2. Prédiction YOLO-Pose (conf interne très bas pour capturer tous les points possibles)
            results = pose_model(image, conf=0.01, verbose=False) 
            
            for result in results:
                if not result.keypoints or result.keypoints.data.numel() == 0:
                    continue
                    
                for box, kpts in zip(result.boxes, result.keypoints.data):
                    # Filtre de Bounding Box globale
                    if box.conf[0] < config.bbox_conf_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    
                    coco_kpts = []
                    num_visible = 0
                    
                    # 3. Traitement des points (Logique de Visibilité COCO)
                    for kpt in kpts:
                        kx, ky, conf = kpt.tolist()
                        
                        # Cas A : Point invalide, hors cadre, ou confiance trop faible -> Absent (0)
                        if conf < config.conf_min or kx <= 0 or kx >= w or ky <= 0 or ky >= h:
                            coco_kpts.extend([0.0, 0.0, 0]) 
                        
                        # Cas B : Haute confiance -> Visible (2)
                        elif conf >= config.conf_visible:
                            coco_kpts.extend([kx, ky, 2])
                            num_visible += 1
                            
                        # Cas C : Confiance moyenne -> Occlus (1)
                        else:
                            coco_kpts.extend([kx, ky, 1])
                            num_visible += 1
                    
                    # Si aucun point n'est au moins occlus/visible, on ignore cette détection
                    if num_visible == 0:
                        continue 
                        
                    # 4. Ajout de l'annotation
                    max_ann_id += 1
                    coco_data['annotations'].append({
                        "id": max_ann_id,
                        "image_id": img_id,
                        "category_id": config.category_id,
                        "segmentation": [],
                        "area": bw * bh,
                        "bbox": [x1, y1, bw, bh],
                        "iscrowd": 0,
                        "keypoints": coco_kpts,
                        "num_keypoints": num_visible
                    })
                    nb_anns_added += 1

            nb_images_processed += 1
            pbar.update(1)

    return nb_images_processed, nb_anns_added


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main():
    config = PoseConfig()
    logger.info(f"=== Auto-Annotation YOLO-Pose : {TASK_NAME} ===")

    if not config.model_path.exists():
        logger.error(f"Modèle introuvable : {config.model_path.resolve()}")
        return
    if not config.input_dir.exists():
        logger.error(f"Dossier d'images introuvable : {config.input_dir.resolve()}")
        return

    # Chargement du modèle
    try:
        from ultralytics import YOLO
        pose_model = YOLO(str(config.model_path))
        logger.info("Modèle YOLO-Pose chargé avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement du modèle : {e}")
        return

    # Pipeline
    coco_data, max_img_id, max_ann_id, existing_images = load_or_create_coco(config.existing_json, config)
    print()

    nb_imgs, nb_anns = run_inference_pipeline(
        pose_model, coco_data, max_img_id, max_ann_id, existing_images, config
    )
    print()

    # Sauvegarde
    if nb_imgs > 0 or nb_anns > 0:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=4)
            
        logger.info(f"✓ Terminé : {nb_imgs} images traitées, {nb_anns} annotations (squelettes) ajoutées.")
        logger.info(f"Fichier exporté : {config.output_json.name}")
        logger.info("Dans CVAT : Task -> Upload annotations -> COCO 1.0 (Keypoints)")
    else:
        logger.info("Aucune nouvelle prédiction générée.")


if __name__ == "__main__":
    main()