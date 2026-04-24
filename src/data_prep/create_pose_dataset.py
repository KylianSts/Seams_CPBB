"""
build_yolo_pose_dataset.py
--------------------------
Pipeline complète : exports CVAT (COCO Keypoints) → dataset YOLO-Pose.

  Étape 1 — Discover  : trouve les JSON de keypoints dans chaque task
  Étape 2 — Merge     : fusionne en un seul COCO avec réindexation des IDs
  Étape 3 — Clean     : points hors écran → 0, liens parasites supprimés
  Étape 4 — Fix       : corrige les inversions haut/bas dans les annotations
  Étape 5 — Convert   : COCO keypoints → labels YOLO-pose (.txt)
  Étape 6 — Split     : train / val (85/15)
  Étape 7 — YAML      : génère data.yaml pour Ultralytics

Format YOLO-pose (une ligne par annotation) :
  class_id cx cy bw bh  x1 y1 v1  x2 y2 v2 ...  (tout normalisé [0,1])
  v=2 : visible | v=0 : invisible / hors écran

Dépendances : scikit-learn, ultralytics, pyyaml, tqdm
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
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

ANNOTATIONS_DIR      = Path("data/annotations/keypoint_detection")
IMAGES_DIR           = Path("data/raw/images/keypoint_detection")
YOLO_DATASET_DIR     = Path("data/datasets/yolo_pose_court")
GROUNDTRUTH_FILENAME = "person_keypoints_default.json"

VALIDATION_SPLIT = 0.15

# Liens du squelette à supprimer (paires en indexation 1-based)
SKELETON_LINKS_TO_REMOVE = [
    {3, 13}, {4, 11}, {25, 32}, {23, 33},
]

# Paires de symétrie pour flip horizontal (gauche, droite) en 1-based
SYMMETRY_PAIRS = [
    (1, 30), (2, 31), (3, 32), (4, 33), (5, 34), (6, 35),
    (7, 28), (8, 29), (9, 26), (10, 27),
    (11, 23), (12, 24), (13, 25), (14, 22),
]

# Paires (kp_haut, kp_bas) : kp_haut doit avoir un Y plus petit que kp_bas
# (Y augmente vers le bas en coordonnées image)
VERTICAL_ORDER = [
    ("21", "19"),
    ("13", "11"),
    ( "8",  "7"),
    ("25", "24"),
]


# ===========================================================================
# ÉTAPE 1 — Découverte des tasks
# ===========================================================================

def discover_tasks(annotations_dir: Path, filename: str) -> List[Path]:
    """Retourne les fichiers `filename` trouvés dans les sous-dossiers de task."""
    found = []
    for task_dir in sorted(annotations_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        gt = task_dir / filename
        if gt.exists():
            found.append(gt)
        else:
            logger.warning(f"Dossier sans annotations ignoré : {task_dir.name}/")
    return found


# ===========================================================================
# ÉTAPE 2 — Fusion multi-tasks
# ===========================================================================

def merge_tasks(json_paths: List[Path]) -> Tuple[dict, List[str]]:
    """
    Fusionne plusieurs exports COCO Keypoints en un seul JSON.
    Réindexe image_id et annotation_id pour éviter les collisions entre tasks.
    Les catégories (keypoints, skeleton) sont issues de la première task.
    """
    merged_images:      List[dict] = []
    merged_annotations: List[dict] = []
    merged_categories:  List[dict] = []
    task_names:         List[str]  = []
    next_img_id = 1
    next_ann_id = 1

    for json_path in json_paths:
        task_name = json_path.parent.name
        task_names.append(task_name)

        data = json.loads(json_path.read_text(encoding="utf-8"))

        if not merged_categories:
            merged_categories = data.get("categories", [])

        old_to_new: Dict[int, int] = {}
        for img in data.get("images", []):
            old_to_new[img["id"]] = next_img_id
            merged_images.append({
                **img,
                "id":        next_img_id,
                "file_name": f"{task_name}/{Path(img['file_name']).name}",
            })
            next_img_id += 1

        for ann in data.get("annotations", []):
            new_img_id = old_to_new.get(ann["image_id"])
            if new_img_id is None:
                logger.warning(f"Annotation orpheline ignorée (image_id={ann['image_id']})")
                continue
            merged_annotations.append({**ann, "id": next_ann_id, "image_id": new_img_id})
            next_ann_id += 1

        logger.info(f"  [{task_name}] {len(data['images'])} images, {len(data['annotations'])} annotations.")

    logger.info(f"Total : {len(merged_images)} images, {len(merged_annotations)} annotations.")
    return {
        "info": {}, "licenses": [],
        "categories":  merged_categories,
        "images":      merged_images,
        "annotations": merged_annotations,
    }, task_names


# ===========================================================================
# ÉTAPE 3 — Nettoyage des annotations
# ===========================================================================

def clean_annotations(coco_data: dict) -> dict:
    """
    Deux nettoyages en un seul passage :
      - Points hors écran (x/y hors dimensions image) → mis à 0
      - Liens parasites dans le squelette → supprimés
    """
    # Index des dimensions d'image
    img_dims = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}

    zeroed = 0
    for ann in coco_data["annotations"]:
        kpts  = ann.get("keypoints", [])
        img_w, img_h = img_dims.get(ann["image_id"], (0, 0))
        if not kpts or img_w == 0:
            continue
        for i in range(len(kpts) // 3):
            x, y, v = kpts[i*3], kpts[i*3+1], kpts[i*3+2]
            if v > 0 and (x <= 0 or x >= img_w or y <= 0 or y >= img_h):
                kpts[i*3], kpts[i*3+1], kpts[i*3+2] = 0.0, 0.0, 0
                zeroed += 1
    logger.info(f"Nettoyage : {zeroed} points hors écran mis à zéro.")

    # Suppression des liens parasites dans le squelette
    for cat in coco_data.get("categories", []):
        if "skeleton" in cat:
            before = len(cat["skeleton"])
            cat["skeleton"] = [e for e in cat["skeleton"] if set(e) not in SKELETON_LINKS_TO_REMOVE]
            removed = before - len(cat["skeleton"])
            if removed:
                logger.info(f"Squelette : {removed} lien(s) parasites supprimé(s).")

    return coco_data


# ===========================================================================
# ÉTAPE 4 — Correction des inversions verticales
# ===========================================================================

def fix_vertical_inversions(coco_data: dict) -> dict:
    """
    Pour chaque paire (kp_haut, kp_bas) définie dans VERTICAL_ORDER :
    si kp_haut.y > kp_bas.y (inversion détectée), échange leurs coordonnées.
    """
    kpt_names = []
    for cat in coco_data["categories"]:
        if cat.get("keypoints"):
            kpt_names = cat["keypoints"]
            break

    if not kpt_names:
        logger.warning("Correction inversions : noms de keypoints introuvables.")
        return coco_data

    name_to_idx = {name: i for i, name in enumerate(kpt_names)}
    total_fixed = 0

    for ann in coco_data["annotations"]:
        kpts = ann.get("keypoints", [])
        if not kpts:
            continue

        for name_haut, name_bas in VERTICAL_ORDER:
            ih = name_to_idx.get(name_haut)
            ib = name_to_idx.get(name_bas)
            if ih is None or ib is None:
                continue

            xh, yh, vh = kpts[ih*3], kpts[ih*3+1], kpts[ih*3+2]
            xb, yb, vb = kpts[ib*3], kpts[ib*3+1], kpts[ib*3+2]

            if vh > 0 and vb > 0 and yh > yb:
                kpts[ih*3], kpts[ih*3+1], kpts[ih*3+2] = xb, yb, vb
                kpts[ib*3], kpts[ib*3+1], kpts[ib*3+2] = xh, yh, vh
                total_fixed += 1

    if total_fixed:
        logger.info(f"Inversions corrigées : {total_fixed} paire(s) échangée(s).")
    else:
        logger.info("Inversions : aucune détectée.")

    return coco_data


# ===========================================================================
# ÉTAPE 5 — Conversion COCO keypoints → labels YOLO-pose
# ===========================================================================

def convert_to_yolo_pose(coco_data: dict, labels_dir: Path) -> Tuple[List[str], List[str], int]:
    """
    Écrit un fichier .txt par image au format YOLO-pose.

    Format par ligne :
      class_id cx cy bw bh  x1 y1 v1  x2 y2 v2 ...  (normalisé [0,1])

    La bbox est calculée depuis les keypoints visibles + marge 5%.

    Returns:
        (chemins relatifs des images converties, noms des keypoints, nb keypoints)
    """
    # Récupérer le nombre de keypoints depuis la catégorie
    num_kpts  = 0
    kpt_names = []
    for cat in coco_data["categories"]:
        if cat.get("keypoints"):
            kpt_names = cat["keypoints"]
            num_kpts  = len(kpt_names)
            break

    if num_kpts == 0:
        logger.error("Impossible de déterminer le nombre de keypoints.")
        return [], [], 0

    img_by_id    = {img["id"]: img for img in coco_data["images"]}
    anns_by_img: Dict[int, list] = {}
    for ann in coco_data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    valid_files = []
    skipped     = 0

    for img_id, anns in tqdm(anns_by_img.items(), desc="Conversion COCO→YOLO"):
        img  = img_by_id.get(img_id)
        if img is None:
            continue

        img_w, img_h = img["width"], img["height"]
        lines = []

        for ann in anns:
            kpts = ann.get("keypoints", [])
            if not kpts:
                continue

            # Séparer les points visibles et invisibles
            points = []
            visible_x, visible_y = [], []
            for i in range(num_kpts):
                x, y, v = (kpts[i*3], kpts[i*3+1], kpts[i*3+2]) if i*3+2 < len(kpts) else (0.0, 0.0, 0)
                points.append((x, y, v))
                if v > 0:
                    visible_x.append(x)
                    visible_y.append(y)

            if not visible_x:
                skipped += 1
                continue

            # Bbox englobante depuis les keypoints visibles + marge 5%
            min_x, max_x = min(visible_x), max(visible_x)
            min_y, max_y = min(visible_y), max(visible_y)
            mx = (max_x - min_x) * 0.05
            my = (max_y - min_y) * 0.05
            cx = ((min_x + max_x) / 2) / img_w
            cy = ((min_y + max_y) / 2) / img_h
            bw = max((max_x - min_x + 2 * mx) / img_w, 0.01)
            bh = max((max_y - min_y + 2 * my) / img_h, 0.01)

            parts = [f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}"]
            for x, y, v in points:
                if v > 0:
                    parts.append(f"{x/img_w:.5f} {y/img_h:.5f} 2")
                else:
                    parts.append("0.00000 0.00000 0")

            lines.append(" ".join(parts))

        if lines:
            stem = Path(img["file_name"]).stem
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            valid_files.append(img["file_name"])

    logger.info(f"Conversion : {len(valid_files)} images, {skipped} annotations ignorées.")
    return valid_files, kpt_names, num_kpts


# ===========================================================================
# ÉTAPES 6 & 7 — Split, copie et génération du data.yaml
# ===========================================================================

def build_flip_idx(kpt_names: List[str], pairs: List[Tuple[int, int]]) -> List[int]:
    """
    Construit flip_idx pour YOLO : flip_idx[i] = j signifie que le point i
    devient le point j après flip horizontal. Les points sans paire sont leur
    propre symétrique (médiane du terrain).
    """
    flip_idx   = list(range(len(kpt_names)))
    name_to_i  = {name: i for i, name in enumerate(kpt_names)}

    for left, right in pairs:
        il = name_to_i.get(str(left))
        ir = name_to_i.get(str(right))
        if il is not None and ir is not None:
            flip_idx[il] = ir
            flip_idx[ir] = il

    return flip_idx


def build_dataset(
    valid_files: List[str],
    kpt_names:   List[str],
    num_kpts:    int,
    labels_dir:  Path,
) -> None:
    """
    Copie images et labels dans la structure YOLO (images/ + labels/ × train/val)
    et génère data.yaml.
    """
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    for split in ("train", "val"):
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_files, val_files = train_test_split(valid_files, test_size=VALIDATION_SPLIT, random_state=42)
    logger.info(f"Split : {len(train_files)} train / {len(val_files)} val.")

    for split, files in [("train", train_files), ("val", val_files)]:
        for rel_path in tqdm(files, desc=f"Copie [{split}]"):
            img_src = IMAGES_DIR  / rel_path
            lbl_src = labels_dir  / f"{Path(rel_path).stem}.txt"
            fname   = Path(rel_path).name
            if img_src.exists() and lbl_src.exists():
                shutil.copy(img_src, YOLO_DATASET_DIR / "images" / split / fname)
                shutil.copy(lbl_src, YOLO_DATASET_DIR / "labels" / split / lbl_src.name)
            else:
                logger.warning(f"Fichier manquant : {rel_path}")

    yaml_data = {
        "path":      str(YOLO_DATASET_DIR.resolve()),
        "train":     "images/train",
        "val":       "images/val",
        "nc":        1,
        "names":     ["basketball_court"],
        "kpt_shape": [num_kpts, 3],
        "flip_idx":  build_flip_idx(kpt_names, SYMMETRY_PAIRS),
    }
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml.dump(yaml_data, sort_keys=False), encoding="utf-8")
    logger.info(f"data.yaml écrit : {yaml_path}")


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main() -> None:
    logger.info("=== Build dataset YOLO-Pose ===\n")

    for path, label in [(ANNOTATIONS_DIR, "annotations"), (IMAGES_DIR, "images")]:
        if not path.exists():
            logger.error(f"Dossier introuvable ({label}) : {path}")
            return

    # Étape 1 — Découverte
    logger.info("--- Étape 1 : Découverte ---")
    json_paths = discover_tasks(ANNOTATIONS_DIR, GROUNDTRUTH_FILENAME)
    if not json_paths:
        logger.error(f"Aucun fichier '{GROUNDTRUTH_FILENAME}' trouvé dans {ANNOTATIONS_DIR}.")
        return
    logger.info(f"{len(json_paths)} task(s) : {[p.parent.name for p in json_paths]}\n")

    # Étape 2 — Fusion
    logger.info("--- Étape 2 : Fusion ---")
    coco_data, _ = merge_tasks(json_paths)
    print()

    # Étape 3 — Nettoyage
    logger.info("--- Étape 3 : Nettoyage ---")
    coco_data = clean_annotations(coco_data)
    print()

    # Étape 4 — Correction inversions
    logger.info("--- Étape 4 : Corrections verticales ---")
    coco_data = fix_vertical_inversions(coco_data)
    print()

    # Étape 5 — Conversion COCO → YOLO
    logger.info("--- Étape 5 : Conversion COCO → YOLO-Pose ---")
    labels_tmp = YOLO_DATASET_DIR.parent / "_labels_tmp"
    valid_files, kpt_names, num_kpts = convert_to_yolo_pose(coco_data, labels_tmp)
    if not valid_files:
        logger.error("Aucun fichier converti.")
        shutil.rmtree(labels_tmp, ignore_errors=True)
        return
    print()

    # Étapes 6 & 7 — Dataset + YAML
    logger.info("--- Étapes 6 & 7 : Dataset + YAML ---")
    build_dataset(valid_files, kpt_names, num_kpts, labels_tmp)

    shutil.rmtree(labels_tmp)
    logger.info("\n=== Dataset créé avec succès ===")
    logger.info(f"Dossier : {YOLO_DATASET_DIR}")
    logger.info(f"Pour entraîner : yolo pose train data={YOLO_DATASET_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()