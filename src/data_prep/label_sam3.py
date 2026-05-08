"""
label_sam3.py
-------------
Pré-labellise automatiquement les objets d'une task CVAT via SAM 3.
Classes détectées : Player, Ball, Referee, Hoop (2-stage), Shot_clock.
Détection Hoop : passe 1 sur image complète ('hoop'), passe 2 sur crop ('Net').
Supporte la détection multi-classes en un seul script.
Format de sortie : COCO JSON compatible avec instances_default.json existant.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASK_NAME = "task_0"

@dataclass
class DetectionTarget:
    """
    Associe une catégorie COCO à un prompt texte SAM 3.

    Si first_prompt est renseigné, active le mode two-stage :
      - Passe 1 : détection sur l'image complète avec first_prompt
      - Crop   : chaque bbox occupe crop_fill (ex. 75 %) de l'image rognée
      - Passe 2 : détection sur le crop avec text_prompt
      - Remappage des coordonnées vers l'image originale
    """
    class_name:   str            # Correspond au 'name' dans categories du JSON
    text_prompt:  str            # Prompt SAM 3 (passe unique ou passe 2 du two-stage)
    first_prompt: Optional[str] = None  # Prompt passe 1 (active le mode two-stage)
    crop_fill:    float         = 0.75  # La bbox doit occuper cette fraction du crop


@dataclass
class LabelingConfig:
    input_dir:     Path = Path(f"data/raw/images/object_detection/{TASK_NAME}")
    existing_json: Path = Path(f"data/annotations/object_detection/{TASK_NAME}/instances_default_false.json")
    output_json:   Path = Path(f"data/annotations/object_detection/{TASK_NAME}/instances_cvat_sam3_labeled.json")

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    # ------------------------------------------------------------------ #
    # Classes à détecter                                                   #
    # class_name DOIT correspondre au 'name' dans les catégories du JSON. #
    # Si la catégorie n'existe pas encore, elle sera créée automatiquement.#
    #                                                                      #
    # Correspondance class_name <-> prompt(s) SAM 3 :                     #
    #   Player     -> "basketball player"         (1 passe)               #
    #   Ball       -> "basketball"                (1 passe)               #
    #   Referee    -> "referee"                   (1 passe)               #
    #   Hoop       -> "hoop" puis "Net" (2-stage, crop 75 %)             #
    #   Shot_clock -> "basketball shot clock"     (1 passe)               #
    # ------------------------------------------------------------------ #
    detection_targets: List[DetectionTarget] = field(default_factory=lambda: [
        DetectionTarget(class_name="Player",     text_prompt="basketball player"),
        DetectionTarget(class_name="Ball",       text_prompt="basketball"),
        DetectionTarget(class_name="Referee",    text_prompt="referee"),
        DetectionTarget(
            class_name="Hoop",
            text_prompt="Net",     # passe 2 : détection fine sur le crop
            first_prompt="hoop",   # passe 1 : localisation grossière sur image complète
            crop_fill=0.75,        # la bbox passe-1 occupe 75 % du crop passe-2
        ),
        DetectionTarget(class_name="Shot_clock", text_prompt="shot clock"),
    ])

    # --- Paramètres d'inférence ---
    batch_size:        int   = 2     # Images traitées simultanément
    score_threshold:   float = 0.5   # Confiance minimale par instance
    mask_threshold:    float = 0.5   # Seuil de binarisation du masque SAM 3
    min_area_pixels:   int   = 80    # Surface minimale d'une bbox (w*h) en pixels²

    # --- FP16 (CUDA uniquement) ---
    use_fp16: bool = torch.cuda.is_available()


# ===========================================================================
# Gestion du JSON COCO
# ===========================================================================

def load_or_create_coco_json(path: Path, targets: List[DetectionTarget]) -> dict:
    """
    Charge le JSON existant ou crée un squelette vide conforme à instances_default.json.
    S'assure ensuite que chaque classe cible possède une entrée dans 'categories'.
    """
    if path.exists():
        logger.info(f"JSON existant chargé : {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            coco = json.load(f)
    else:
        logger.info("Aucun JSON existant — création d'un fichier vide.")
        coco = {
            "licenses":    [{"name": "", "id": 0, "url": ""}],
            "info":        {"contributor": "", "date_created": "", "description": "",
                            "url": "", "version": "", "year": ""},
            "categories":  [],
            "images":      [],
            "annotations": [],
        }

    # Ajoute les catégories manquantes
    existing_names = {cat["name"] for cat in coco.get("categories", [])}
    max_cat_id     = max((cat["id"] for cat in coco["categories"]), default=0)

    for target in targets:
        if target.class_name not in existing_names:
            max_cat_id += 1
            coco["categories"].append({
                "id":           max_cat_id,
                "name":         target.class_name,
                "supercategory": "",
            })
            logger.info(f"Catégorie ajoutée : '{target.class_name}' (id={max_cat_id})")

    return coco


def save_coco_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON sauvegardé : {path}")


def get_category_id(coco: dict, class_name: str) -> int:
    for cat in coco.get("categories", []):
        if cat["name"] == class_name:
            return cat["id"]
    raise ValueError(f"Catégorie introuvable dans le JSON : '{class_name}'")


def build_lookup_tables(coco: dict) -> Tuple[Dict[str, int], Dict[int, Set[int]]]:
    """
    Retourne :
      filename_to_id  : { basename → image_id }
      labeled_per_cat : { category_id → set(image_ids déjà annotés) }

    Permet de savoir précisément quelles (image, classe) ont déjà été traitées
    et d'éviter de re-générer des doublons.
    """
    filename_to_id: Dict[str, int] = {
        Path(img["file_name"]).name: img["id"]
        for img in coco.get("images", [])
    }
    labeled_per_cat: Dict[int, Set[int]] = {}
    for ann in coco.get("annotations", []):
        labeled_per_cat.setdefault(ann["category_id"], set()).add(ann["image_id"])

    return filename_to_id, labeled_per_cat


def get_next_ids(coco: dict) -> Tuple[int, int]:
    max_img_id = max((img["id"] for img in coco.get("images",      [])), default=0)
    max_ann_id = max((ann["id"] for ann in coco.get("annotations", [])), default=0)
    return max_img_id + 1, max_ann_id + 1


# ===========================================================================
# Construction d'une annotation COCO
# ===========================================================================

def build_annotation(
    ann_id: int, image_id: int, category_id: int,
    box_xyxy: List[float], score: float,
) -> dict:
    """
    Formate une annotation au standard instances_default.json :
      bbox en [x, y, w, h]  |  area = w * h  |  attributes sans segmentation
    Le champ 'score' (SAM 3) est ajouté dans attributes — absent des
    annotations manuelles existantes, mais ignoré par CVAT à l'import.
    """
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    return {
        "id":           ann_id,
        "image_id":     image_id,
        "category_id":  category_id,
        "segmentation": [],
        "area":         round(w * h, 4),
        "bbox":         [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
        "iscrowd":      0,
        "attributes":   {"occluded": False, "rotation": 0.0, "score": round(score, 4)},
    }


# ===========================================================================
# Pipeline d'inférence multi-classes
# ===========================================================================

# ===========================================================================
# Helpers Two-Stage (détection Hoop en deux passes)
# ===========================================================================

def crop_with_margin(
    image: Image.Image,
    box_xyxy: List[float],
    crop_fill: float = 0.75,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Rogne l'image autour de box_xyxy de sorte que la bbox occupe
    crop_fill de la dimension du crop (ex. 75 %).

    Retourne :
      cropped_image : Image PIL rognée
      (off_x, off_y) : décalage du coin supérieur-gauche du crop
                        dans l'image originale (pour le remappage)
    """
    w_orig, h_orig = image.size
    x1, y1, x2, y2 = box_xyxy
    box_w, box_h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # Taille du crop pour que la bbox n'occupe que crop_fill
    crop_w = box_w / crop_fill
    crop_h = box_h / crop_fill

    # Crop centré sur la bbox, clampé aux bords de l'image
    cx1 = max(0,      int(cx - crop_w / 2))
    cy1 = max(0,      int(cy - crop_h / 2))
    cx2 = min(w_orig, int(cx + crop_w / 2))
    cy2 = min(h_orig, int(cy + crop_h / 2))

    cropped = image.crop((cx1, cy1, cx2, cy2))
    return cropped, (cx1, cy1)


def remap_box_to_original(
    box_in_crop: List[float],
    offset: Tuple[int, int],
) -> List[float]:
    """
    Convertit une bbox [x1,y1,x2,y2] dans l'espace du crop
    vers l'espace de l'image originale en ajoutant l'offset.
    """
    off_x, off_y = offset
    x1, y1, x2, y2 = box_in_crop
    return [x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y]


def run_two_stage_for_image(
    processor:   "Sam3Processor",
    model:       "Sam3Model",
    image:       Image.Image,
    target:      DetectionTarget,
    config:      "LabelingConfig",
    dtype:       torch.dtype,
) -> List[Tuple[List[float], float]]:
    """
    Détection two-stage pour une seule image :
      1. Passe 1 sur l'image complète avec target.first_prompt
         → N bboxes grossières (une par panier détecté)
      2. Pour chaque bbox grossière :
           a. Crop centré avec marge (bbox = crop_fill du crop)
           b. Passe 2 sur le crop avec target.text_prompt
           c. Remappage des coordonnées vers l'image originale
      3. Retourne [(box_xyxy_originale, score), ...]

    Si aucun panier n'est trouvé en passe 1, retourne [].
    """
    device = config.device
    results_out: List[Tuple[List[float], float]] = []

    # ── Passe 1 : localisation grossière sur image complète ─────────────── #
    inputs_p1 = processor(
        images=[image],
        text=[target.first_prompt],
        return_tensors="pt",
    ).to(device)
    if config.use_fp16:
        inputs_p1 = {k: (v.to(dtype) if torch.is_floating_point(v) else v)
                     for k, v in inputs_p1.items()}

    outputs_p1 = model(**inputs_p1)

    target_sizes_p1 = (
        inputs_p1.get("original_sizes").tolist()
        if inputs_p1.get("original_sizes") is not None
        else [[image.height, image.width]]
    )
    results_p1 = processor.post_process_instance_segmentation(
        outputs_p1,
        threshold=config.score_threshold,
        mask_threshold=config.mask_threshold,
        target_sizes=target_sizes_p1,
    )

    rough_boxes  = results_p1[0]["boxes"]   # Tensor [N, 4]
    rough_scores = results_p1[0]["scores"]  # Tensor [N]

    if len(rough_boxes) == 0:
        return []

    # ── Passe 2 : détection fine sur chaque crop ─────────────────────────── #
    for j in range(len(rough_boxes)):
        rough_box = rough_boxes[j].tolist()

        # Filtre surface minimale dès la passe 1
        rx1, ry1, rx2, ry2 = rough_box
        if (rx2 - rx1) * (ry2 - ry1) < config.min_area_pixels:
            continue

        # Crop centré autour de la bbox passe-1
        crop_img, (off_x, off_y) = crop_with_margin(image, rough_box, target.crop_fill)

        inputs_p2 = processor(
            images=[crop_img],
            text=[target.text_prompt],
            return_tensors="pt",
        ).to(device)
        if config.use_fp16:
            inputs_p2 = {k: (v.to(dtype) if torch.is_floating_point(v) else v)
                         for k, v in inputs_p2.items()}

        outputs_p2 = model(**inputs_p2)

        target_sizes_p2 = (
            inputs_p2.get("original_sizes").tolist()
            if inputs_p2.get("original_sizes") is not None
            else [[crop_img.height, crop_img.width]]
        )
        results_p2 = processor.post_process_instance_segmentation(
            outputs_p2,
            threshold=config.score_threshold,
            mask_threshold=config.mask_threshold,
            target_sizes=target_sizes_p2,
        )

        fine_boxes  = results_p2[0]["boxes"]
        fine_scores = results_p2[0]["scores"]

        if len(fine_boxes) == 0:
            # Passe 2 n'a rien trouvé : on conserve la bbox grossière de passe 1
            # remapée (meilleure que rien) avec le score de passe 1
            logger.debug(
                f"Two-stage : passe 2 vide sur le crop, "
                f"fallback bbox passe 1 (score={float(rough_scores[j]):.3f})"
            )
            results_out.append((rough_box, float(rough_scores[j])))
            continue

        # On ne retient que la bbox avec le meilleur score du crop
        best_idx  = int(torch.argmax(fine_scores))
        best_box  = fine_boxes[best_idx].tolist()
        best_score = float(fine_scores[best_idx])

        # Filtre surface minimale sur le résultat passe-2
        fx1, fy1, fx2, fy2 = best_box
        if (fx2 - fx1) * (fy2 - fy1) < config.min_area_pixels:
            continue

        # Remappage vers l'espace de l'image originale
        original_box = remap_box_to_original(best_box, (off_x, off_y))
        results_out.append((original_box, best_score))

    return results_out


def run_inference_pipeline(
    processor: Sam3Processor,
    model:     Sam3Model,
    coco:      dict,
    config:    LabelingConfig,
) -> Dict[str, Tuple[int, int]]:
    """
    Pour chaque DetectionTarget :
      1. Identifie les images non encore annotées pour cette classe.
      2. Les traite en batches via SAM 3.
      3. Ajoute les annotations dans `coco` en place.

    Retourne { class_name: (nb_images_traitées, nb_boxes_ajoutées) }.
    """
    all_paths = sorted(
        p for p in config.input_dir.iterdir()
        if p.suffix.lower() in config.valid_extensions
    )
    if not all_paths:
        logger.warning(f"Aucune image trouvée dans {config.input_dir}")
        return {}

    filename_to_id, labeled_per_cat = build_lookup_tables(coco)
    next_img_id, next_ann_id        = get_next_ids(coco)
    dtype      = torch.float16 if config.use_fp16 else torch.float32
    # Préfixe file_name conforme au JSON de référence : "object_detection/{TASK_NAME}/"
    file_prefix = f"object_detection/{TASK_NAME}"

    stats: Dict[str, Tuple[int, int]] = {}

    # ======================================================================
    # Boucle 1 : une passe complète par classe cible
    # ======================================================================
    for target in config.detection_targets:
        category_id  = get_category_id(coco, target.class_name)
        done_img_ids = labeled_per_cat.get(category_id, set())

        # Images sans annotation pour CETTE classe (qu'elles soient ou non
        # déjà dans le JSON — une image peut être annotée Ball mais pas Player)
        to_process = [
            p for p in all_paths
            if not (p.name in filename_to_id and filename_to_id[p.name] in done_img_ids)
        ]

        if not to_process:
            logger.info(f"[{target.class_name}] Toutes les images déjà annotées — skip.")
            stats[target.class_name] = (0, 0)
            continue

        logger.info(
            f"[{target.class_name}] {len(to_process)} image(s) à traiter "
            f"(prompt : \"{target.text_prompt}\")"
        )

        boxes_added      = 0
        images_processed = 0

        with torch.inference_mode(), tqdm(
            total=len(to_process),
            desc=f"SAM 3 — {target.class_name:<8}",
            unit="img",
        ) as pbar:

            # ==============================================================
            # Boucle 2 : batches d'images
            # ==============================================================
            for i in range(0, len(to_process), config.batch_size):
                batch_paths  = to_process[i : i + config.batch_size]
                batch_images: List[Image.Image] = []
                batch_ids:    List[int]         = []

                # ── Chargement des images ──────────────────────────────── #
                for img_path in batch_paths:
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                    except Exception as exc:
                        logger.warning(f"Lecture impossible ({img_path.name}) : {exc}")
                        continue

                    # Enregistre l'image dans le JSON si elle n'y est pas encore
                    if img_path.name not in filename_to_id:
                        w_img, h_img = pil_img.size
                        coco["images"].append({
                            "id":            next_img_id,
                            "width":         w_img,
                            "height":        h_img,
                            "file_name":     f"{file_prefix}/{img_path.name}",
                            "license":       0,
                            "flickr_url":    "",
                            "coco_url":      "",
                            "date_captured": 0,
                        })
                        filename_to_id[img_path.name] = next_img_id
                        next_img_id += 1

                    batch_images.append(pil_img)
                    batch_ids.append(filename_to_id[img_path.name])

                if not batch_images:
                    continue

                # ── Préparation des inputs ─────────────────────────────── #
                # Même prompt pour toutes les images du batch
                batch_texts = [target.text_prompt] * len(batch_images)

                inputs = processor(
                    images=batch_images,
                    text=batch_texts,
                    return_tensors="pt",
                ).to(config.device)

                # Cast FP16 uniquement sur les tenseurs flottants
                if config.use_fp16:
                    inputs = {
                        k: (v.to(dtype) if torch.is_floating_point(v) else v)
                        for k, v in inputs.items()
                    }

                # ── Inférence + annotations ──────────────────────────── #
                # Mode two-stage (ex: Hoop) : une image à la fois
                # Mode standard            : traitement du batch entier
                if target.first_prompt is not None:
                    # Two-stage : on itère image par image dans le batch
                    for b_idx, (pil_img, img_id) in enumerate(zip(batch_images, batch_ids)):
                        detections = run_two_stage_for_image(
                            processor, model, pil_img, target, config, dtype
                        )
                        for box_xyxy, score in detections:
                            ann = build_annotation(next_ann_id, img_id, category_id, box_xyxy, score)
                            coco["annotations"].append(ann)
                            next_ann_id += 1
                            boxes_added += 1

                        labeled_per_cat.setdefault(category_id, set()).add(img_id)
                        images_processed += 1
                        pbar.update(1)

                else:
                    # Mode standard : inférence sur le batch complet en une passe
                    outputs = model(**inputs)

                    original_sizes = inputs.get("original_sizes")
                    target_sizes   = (
                        original_sizes.tolist() if original_sizes is not None
                        else [[img.height, img.width] for img in batch_images]
                    )
                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=config.score_threshold,
                        mask_threshold=config.mask_threshold,
                        target_sizes=target_sizes,
                    )

                    for b_idx, result in enumerate(results):
                        img_id         = batch_ids[b_idx]
                        detected_boxes = result["boxes"]    # Tensor [N, 4] xyxy
                        detected_scores = result["scores"]  # Tensor [N]

                        for j in range(len(detected_boxes)):
                            box_xyxy = detected_boxes[j].tolist()
                            score    = float(detected_scores[j])

                            x1, y1, x2, y2 = box_xyxy
                            if (x2 - x1) * (y2 - y1) < config.min_area_pixels:
                                continue

                            ann = build_annotation(next_ann_id, img_id, category_id, box_xyxy, score)
                            coco["annotations"].append(ann)
                            next_ann_id  += 1
                            boxes_added  += 1

                        labeled_per_cat.setdefault(category_id, set()).add(img_id)
                        images_processed += 1
                        pbar.update(1)

        stats[target.class_name] = (images_processed, boxes_added)
        logger.info(
            f"[{target.class_name}] ✓ {images_processed} images | "
            f"{boxes_added} instances détectées."
        )

    return stats


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main() -> None:
    config = LabelingConfig()

    logger.info(f"=== Auto-Annotation SAM 3 Multi-Classes : {TASK_NAME} ===")
    logger.info(f"Device : {config.device} | FP16 : {config.use_fp16}")
    logger.info(f"Classes : {[t.class_name for t in config.detection_targets]} ({len(config.detection_targets)} cibles)")

    if "cuda" in config.device:
        torch.backends.cudnn.benchmark = True

    # ── Chargement du modèle ──────────────────────────────────────────── #
    try:
        processor  = Sam3Processor.from_pretrained("facebook/sam3")
        model_dtype = torch.float16 if config.use_fp16 else torch.float32
        model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=model_dtype,
        ).to(config.device).eval()
        logger.info("Modèle SAM 3 chargé.")
    except Exception:
        logger.exception("Erreur lors du chargement de SAM 3.")
        return

    # ── Chargement / création du JSON ────────────────────────────────── #
    coco = load_or_create_coco_json(config.existing_json, config.detection_targets)

    # ── Inférence ────────────────────────────────────────────────────── #
    stats = run_inference_pipeline(processor, model, coco, config)

    # ── Sauvegarde & résumé ──────────────────────────────────────────── #
    total_images = sum(imgs  for imgs,  _     in stats.values())
    total_boxes  = sum(boxes for _,     boxes in stats.values())

    if total_images > 0:
        save_coco_json(coco, config.output_json)
        logger.info("=" * 55)
        logger.info(f"RÉSUMÉ  —  {total_images} images traitées  |  {total_boxes} instances ajoutées")
        for class_name, (imgs, boxes) in stats.items():
            logger.info(f"  {class_name:<10} : {imgs:>4} images  |  {boxes:>5} instances")
        logger.info("=" * 55)
    else:
        logger.info("Aucune nouvelle image traitée — JSON inchangé.")


if __name__ == "__main__":
    main()