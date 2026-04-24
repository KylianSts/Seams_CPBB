"""
build_coco_dataset.py
---------------------
Pipeline complète : backups CVAT → dataset COCO (train / valid / test).

  Étape 1 — Découverte des tasks dans ANNOTATIONS_DIR
  Étape 2 — Conversion de chaque backup CVAT → JSON COCO (en mémoire)
  Étape 3 — Fusion de toutes les tasks en un seul JSON
  Étape 4 — Correction des IDs de catégories (base 1 CVAT → base 0 PyTorch)
  Étape 5 — Split train / valid / test
  Étape 6 — Écriture des JSONs et copie des images

Note : les images sans annotation sont conservées dans le dataset.
Elles servent à entraîner le modèle à ne pas prédire en l'absence d'objet.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANNOTATIONS_DIR  = Path("data/annotations/object_detection")
IMAGES_DIR       = Path("data/raw/images/object_detection")
COCO_DATASET_DIR = Path("data/datasets/coco")

VALIDATION_SPLIT = 0.15
DEFAULT_WIDTH    = 1280
DEFAULT_HEIGHT   = 720
CLASSES_TO_KEEP  = {"Player", "Ball", "Referee", "Hoop"}


# ===========================================================================
# ÉTAPE 1 — Découverte des tasks
# ===========================================================================

def discover_tasks(annotations_dir: Path) -> List[Path]:
    """
    Retourne les dossiers de task contenant un backup CVAT valide.
    Un backup valide contient : annotations.json, task.json, Data/manifest.jsonl.
    """
    tasks = []
    for task_dir in sorted(annotations_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        backup_dir = task_dir / "backup"
        required = [
            backup_dir / "annotations.json",
            backup_dir / "task.json",
            backup_dir / "Data" / "manifest.jsonl",
        ]
        if all(p.exists() for p in required):
            tasks.append(task_dir)
        else:
            missing = [p.name for p in required if not p.exists()]
            logger.warning(f"Task '{task_dir.name}' ignorée — manquants dans backup/ : {missing}")
    return tasks


# ===========================================================================
# ÉTAPE 2 — Conversion CVAT → COCO (par task, en mémoire)
# ===========================================================================

def _parse_manifest(path: Path) -> Dict[int, dict]:
    """
    Construit un index {frame_index: {filename, width, height}} depuis le manifest CVAT.
    Les deux premières lignes (version + type) sont ignorées.
    """
    frame_index: Dict[int, dict] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines[2:]):
        entry = json.loads(line)
        if "name" in entry:
            frame_index[i] = {
                "filename": entry["name"] + entry.get("extension", ".png"),
                "width":    entry.get("width",  DEFAULT_WIDTH),
                "height":   entry.get("height", DEFAULT_HEIGHT),
            }
    return frame_index


def _parse_task(path: Path) -> Tuple[Dict[str, int], Set[int]]:
    """
    Extrait depuis task.json :
      - label_to_id     : {nom_label: category_id} base 1, filtré sur CLASSES_TO_KEEP
      - deleted_frames  : indices des frames supprimées dans CVAT
    """
    task = json.loads(path.read_text(encoding="utf-8"))
    label_to_id = {
        label["name"]: i + 1
        for i, label in enumerate(task.get("labels", []))
        if label["name"] in CLASSES_TO_KEEP
    }
    deleted_frames = set(task.get("data", {}).get("deleted_frames", []))
    return label_to_id, deleted_frames


def convert_task_to_coco(task_dir: Path) -> Optional[dict]:
    """
    Convertit un backup CVAT en JSON COCO (en mémoire).

    Toutes les frames non supprimées sont incluses, même sans annotation,
    pour permettre l'entraînement sur images négatives.

    Returns:
        Dict COCO {info, licenses, categories, images, annotations}
        ou None si la task est vide.
    """
    task_name  = task_dir.name
    backup_dir = task_dir / "backup"

    label_to_id, deleted_frames = _parse_task(backup_dir / "task.json")
    frame_index                  = _parse_manifest(backup_dir / "Data" / "manifest.jsonl")
    raw_annotations              = json.loads((backup_dir / "annotations.json").read_text(encoding="utf-8"))

    logger.info(f"  Classes conservées   : {sorted(label_to_id.keys())}")
    logger.info(f"  Frames supprimées    : {len(deleted_frames)}")
    logger.info(f"  Frames dans manifest : {len(frame_index)}")

    # Index des shapes rectangulaires et des bonnes classes, regroupées par frame
    shapes_by_frame: Dict[int, List[dict]] = {}
    for shape in raw_annotations[0].get("shapes", []):
        if shape.get("type") == "rectangle" and shape.get("label") in label_to_id:
            shapes_by_frame.setdefault(shape["frame"], []).append(shape)

    images:      List[dict] = []
    annotations: List[dict] = []
    next_img_id = 1
    next_ann_id = 1

    # On itère sur TOUTES les frames du manifest, même sans annotation
    for frame_idx, meta in sorted(frame_index.items()):
        if frame_idx in deleted_frames:
            continue

        filename = meta["filename"].split("/")[-1]
        images.append({
            "id":            next_img_id,
            "file_name":     f"{task_name}/{filename}",
            "width":         meta["width"],
            "height":        meta["height"],
            "license":       0,
            "flickr_url":    "",
            "coco_url":      "",
            "date_captured": 0,
        })

        for shape in shapes_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = shape["points"]
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            annotations.append({
                "id":           next_ann_id,
                "image_id":     next_img_id,
                "category_id":  label_to_id[shape["label"]] - 1,  # base 1 → base 0
                "segmentation": [],
                "area":         w * h,
                "bbox":         [x, y, w, h],
                "iscrowd":      0,
                "attributes": {
                    "occluded": shape.get("occluded", False),
                    "rotation": shape.get("rotation", 0.0),
                },
            })
            next_ann_id += 1

        next_img_id += 1

    annotated   = len({a["image_id"] for a in annotations})
    unannotated = len(images) - annotated
    logger.info(f"  Images : {len(images)} ({annotated} annotées, {unannotated} négatives)")
    logger.info(f"  Annotations : {len(annotations)}")

    if not images:
        logger.warning(f"  ⚠ Aucune image pour '{task_name}'.")
        return None

    categories = [
        {"id": cid - 1, "name": name, "supercategory": ""}
        for name, cid in sorted(label_to_id.items(), key=lambda x: x[1])
    ]

    return {
        "info":        {},
        "licenses":    [],
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }


# ===========================================================================
# ÉTAPE 3 — Fusion de toutes les tasks
# ===========================================================================

def merge_tasks(coco_per_task: List[dict]) -> dict:
    """
    Fusionne plusieurs JSON COCO en un seul.

    Réindexe image_id et annotation_id pour éviter les collisions
    (chaque task repart de 1, ce qui crée des doublons d'ID à la fusion).
    Les catégories sont reprises depuis la première task :
    toutes les tasks partagent les mêmes classes.
    """
    merged_images:      List[dict] = []
    merged_annotations: List[dict] = []
    merged_categories:  List[dict] = []

    next_img_id = 1
    next_ann_id = 1

    for coco in coco_per_task:
        if not merged_categories:
            merged_categories = coco.get("categories", [])

        old_to_new: Dict[int, int] = {}

        for img in coco.get("images", []):
            old_to_new[img["id"]] = next_img_id
            merged_images.append({**img, "id": next_img_id})
            next_img_id += 1

        for ann in coco.get("annotations", []):
            new_img_id = old_to_new.get(ann["image_id"])
            if new_img_id is None:
                logger.warning(f"Annotation orpheline ignorée (image_id={ann['image_id']})")
                continue
            merged_annotations.append({
                **ann,
                "id":       next_ann_id,
                "image_id": new_img_id,
            })
            next_ann_id += 1

    annotated   = len({a["image_id"] for a in merged_annotations})
    unannotated = len(merged_images) - annotated
    logger.info(
        f"Fusion : {len(merged_images)} images "
        f"({annotated} annotées, {unannotated} négatives), "
        f"{len(merged_annotations)} annotations."
    )
    return {
        "info":        {},
        "licenses":    [],
        "categories":  merged_categories,
        "images":      merged_images,
        "annotations": merged_annotations,
    }


# ===========================================================================
# ÉTAPE 4 — Correction des IDs de catégories (base 1 CVAT → base 0 PyTorch)
# ===========================================================================

def fix_category_ids(coco_data: dict) -> dict:
    """
    Réindexe les catégories pour qu'elles commencent à 0.
    CVAT exporte en base 1 ; PyTorch et RF-DETR attendent la base 0.
    Les annotations sont mises à jour en conséquence.
    """
    sorted_cats  = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
    id_mapping:    Dict[int, int] = {}
    new_categories = []

    for new_id, cat in enumerate(sorted_cats):
        id_mapping[cat["id"]] = new_id
        new_categories.append({**cat, "id": new_id})

    coco_data["categories"] = new_categories

    for ann in coco_data.get("annotations", []):
        if ann["category_id"] in id_mapping:
            ann["category_id"] = id_mapping[ann["category_id"]]

    logger.info(f"Catégories : {[c['name'] for c in new_categories]}")
    return coco_data


# ===========================================================================
# ÉTAPES 5 & 6 — Split et écriture
# ===========================================================================

@dataclass
class SplitPaths:
    root:       Path
    annotation: Path


def _setup_output(base_dir: Path, task_names: List[str]) -> Dict[str, SplitPaths]:
    """Crée la structure train/valid/test avec un sous-dossier par task."""
    if base_dir.exists():
        logger.info(f"Suppression du dataset existant : {base_dir}")
        shutil.rmtree(base_dir)

    splits = {}
    for subset in ("train", "valid", "test"):
        root = base_dir / subset
        for task_name in task_names:
            (root / task_name).mkdir(parents=True, exist_ok=True)
        splits[subset] = SplitPaths(root=root, annotation=root / "_annotations.coco.json")

    logger.info(f"Structure créée dans : {base_dir}")
    return splits


def write_split(
    images:          List[dict],
    all_annotations: List[dict],
    base_json:       dict,
    subset:          str,
    paths:           SplitPaths,
    images_dir:      Path,
) -> None:
    """
    Écrit le JSON d'un split et copie physiquement ses images.
    Les images sans annotation sont incluses avec une liste d'annotations vide.
    """
    image_ids = {img["id"] for img in images}
    split_json = {
        "info":        base_json.get("info", {}),
        "licenses":    base_json.get("licenses", []),
        "categories":  base_json.get("categories", []),
        "images":      images,
        "annotations": [a for a in all_annotations if a["image_id"] in image_ids],
    }

    with open(paths.annotation, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=4, ensure_ascii=False)

    annotated   = len({a["image_id"] for a in split_json["annotations"]})
    unannotated = len(images) - annotated
    logger.info(
        f"[{subset}] {len(images)} images "
        f"({annotated} annotées, {unannotated} négatives), "
        f"{len(split_json['annotations'])} annotations → {paths.annotation}"
    )

    missing = 0
    for img_info in tqdm(images, desc=f"Copie [{subset}]", unit="img"):
        src = images_dir / img_info["file_name"]
        dst = paths.root  / img_info["file_name"]
        if src.exists():
            shutil.copy(src, dst)
        else:
            missing += 1
            logger.warning(f"Image introuvable : {src}")

    if missing:
        logger.warning(f"[{subset}] {missing} image(s) introuvable(s).")


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main() -> None:
    logger.info("=== Build dataset COCO ===\n")

    for path, label in [(ANNOTATIONS_DIR, "annotations"), (IMAGES_DIR, "images")]:
        if not path.exists():
            logger.error(f"Dossier introuvable ({label}) : {path}")
            return

    # Étape 1 — Découverte
    logger.info("--- Étape 1 : Découverte des tasks ---")
    tasks = discover_tasks(ANNOTATIONS_DIR)
    if not tasks:
        logger.error("Aucune task avec backup valide trouvée.")
        return
    logger.info(f"{len(tasks)} task(s) : {[t.name for t in tasks]}\n")

    # Étape 2 — Conversion CVAT → COCO par task
    logger.info("--- Étape 2 : Conversion CVAT → COCO ---")
    coco_per_task = []
    task_names    = []
    for task_dir in tasks:
        logger.info(f"[{task_dir.name}]")
        coco = convert_task_to_coco(task_dir)
        if coco:
            coco_per_task.append(coco)
            task_names.append(task_dir.name)
        print()

    if not coco_per_task:
        logger.error("Aucune task convertie avec succès.")
        return

    # Étape 3 — Fusion
    logger.info("--- Étape 3 : Fusion des tasks ---")
    merged = merge_tasks(coco_per_task)
    print()

    # Étape 4 — Correction des IDs de catégories
    logger.info("--- Étape 4 : Correction des catégories ---")
    merged = fix_category_ids(merged)
    print()

    # Étape 5 — Split train / valid / test
    logger.info("--- Étape 5 : Split train / valid / test ---")
    train_imgs, val_imgs = train_test_split(
        merged["images"], test_size=VALIDATION_SPLIT, random_state=42
    )
    logger.info(f"Train : {len(train_imgs)} | Valid + Test : {len(val_imgs)}\n")

    # Étape 6 — Écriture du dataset
    logger.info("--- Étape 6 : Écriture du dataset ---")
    splits = _setup_output(COCO_DATASET_DIR, task_names)
    write_split(train_imgs, merged["annotations"], merged, "train", splits["train"], IMAGES_DIR)
    write_split(val_imgs,   merged["annotations"], merged, "valid", splits["valid"], IMAGES_DIR)
    write_split(val_imgs,   merged["annotations"], merged, "test",  splits["test"],  IMAGES_DIR)

    logger.info("\n=== Dataset créé avec succès. ===")


if __name__ == "__main__":
    main()