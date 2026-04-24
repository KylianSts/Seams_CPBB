"""
neon_bullet_3pt.py – v9.0 (Couleur Libre + Direction + Mode Instantané + Maintien)
─────────────────────────────────────────────────────────────────────────────
Paramètres clés (section ★ PARAMÈTRES VISUELS ★) :

  LINE_COLOR_BGR  – couleur principale de la ligne (format BGR)
  DIRECTION       – "forward" | "backward" | None
                    • "forward"  : trace de gauche → droite (sens normal)
                    • "backward" : trace de droite → gauche (sens inverse)
                    • None       : tout le tracé apparaît instantanément
  HOLD_FRAMES     – nb de frames pendant lesquelles la ligne reste visible
                    à pleine intensité avant de s'estomper progressivement
  SPEED           – points de chemin avancés par frame (ignoré si DIRECTION is None)
  TAIL_LEN        – longueur de la queue en points (contrôle la durée du fondu)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from boxmot import BotSort
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PLAYER = Path("models/runs/object_detection/rfdetr-medium_1280px_100ep_v1/checkpoint_best_ema.pth")
CHECKPOINT_COURT  = Path("models/runs/keypoint_detection/yolo11m-pose_1000ep_v1/weights/best.pt")
CHECKPOINT_SAM    = Path("models/weights/sam2.1_hiera_small.pt")
CONFIG_SAM        = "configs/sam2.1/sam2.1_hiera_s.yaml"
SOURCE_VIDEO_PATH = Path("data/demos/video_raw/video_cergy_3pts.mp4")
OUTPUT_PATH       = Path("data/demos/videos_annotated/demo_test.mp4")
REID_WEIGHTS      = Path("models/weights/osnet_x0_25_msmt17.pt")

INFERENCE_RESOLUTION = 800
CONF_PLAYER          = 0.40
CONF_KP              = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# ★ PARAMÈTRES VISUELS — ajustez ici ★
# ─────────────────────────────────────────────────────────────────────────────
START_TIME_SEC = 2.65     # Timestamp de démarrage de l'effet (secondes)
M              = 1        # Multiplicateur d'épaisseur global

# ── Couleur ──────────────────────────────────────────────────────────────────
# Format BGR (Blue, Green, Red).
LINE_COLOR_BGR = (0, 0, 255)   

# ── Direction ────────────────────────────────────────────────────────────────
# "forward"  → le tracé part du point 0 et avance vers la fin
# "backward" → le tracé part de la fin et recule vers le début
# None       → toute la ligne apparaît instantanément, puis disparaît
DIRECTION = "forward"

# ── Durée de maintien ────────────────────────────────────────────────────────
# Nombre de frames pendant lesquelles la ligne reste à pleine intensité
# une fois le tracé terminé, avant de commencer à s'estomper.
HOLD_FRAMES = 45

# ── Vitesse & longueur de queue ──────────────────────────────────────────────
SPEED    = 200    # Points avancés par frame (ignoré si DIRECTION is None)
TAIL_LEN = 7000   # Longueur de la queue en points (plus = fondu plus lent)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DE LA PALETTE DE COULEUR (noir → couleur choisie)
# ─────────────────────────────────────────────────────────────────────────────
def make_color_palette(color_bgr: tuple, num_colors: int) -> np.ndarray:
    """Génère un dégradé linéaire de (0,0,0) vers color_bgr en num_colors étapes."""
    palette = np.zeros((num_colors, 3), dtype=np.float32)
    for i in range(num_colors):
        t = i / max(num_colors - 1, 1)
        palette[i] = [color_bgr[c] * t for c in range(3)]
    return palette

LINE_PALETTE = make_color_palette(LINE_COLOR_BGR, TAIL_LEN)

# ─────────────────────────────────────────────────────────────────────────────
# GÉOMÉTRIE TERRAIN FIBA
# ─────────────────────────────────────────────────────────────────────────────
COURT_L, COURT_W = 28.0, 15.0
X_BASKET         = 1.575
X_FT             = 5.8
X_3PT_START      = 2.99
Y_CENTER         = 7.5
Y_KEY_HALF       = 2.45
Y_3PT_OFFSET     = 0.9
Y_NC_RADIUS      = 1.25
R_3PT            = 6.75
R_FT             = 1.8

APEX_X = X_BASKET + R_3PT
APEX_Y = Y_CENTER

ANGLE_JUNCTION_DEG = np.degrees(np.arccos((X_3PT_START - X_BASKET) / R_3PT))
Y_JUNCTION_TOP     = Y_CENTER + R_3PT * np.sin(np.radians(ANGLE_JUNCTION_DEG))
Y_JUNCTION_BOT     = Y_CENTER - R_3PT * np.sin(np.radians(ANGLE_JUNCTION_DEG))

FIBA_M_COORDS = {
    1:  (0.0,  15.0), 2:  (0.0,  COURT_W - Y_3PT_OFFSET), 3:  (0.0,  Y_CENTER + Y_KEY_HALF),
    4:  (0.0,  Y_CENTER - Y_KEY_HALF), 5:  (0.0,  Y_3PT_OFFSET), 6:  (0.0,  0.0),
    7:  (X_BASKET, Y_CENTER + Y_NC_RADIUS), 8:  (X_BASKET, Y_CENTER - Y_NC_RADIUS),
    9:  (X_3PT_START, COURT_W - Y_3PT_OFFSET), 10: (X_3PT_START, Y_3PT_OFFSET),
    11: (X_FT, Y_CENTER + Y_KEY_HALF), 12: (X_FT, Y_CENTER), 13: (X_FT, Y_CENTER - Y_KEY_HALF),
    14: (APEX_X, APEX_Y), 17: (14.0, 15.0), 18: (14.0, 7.5), 19: (14.0, 0.0),
}

SYMMETRY_PAIRS_1BASED = [
    (1, 30), (2, 31), (3, 32), (4, 33), (5, 34), (6, 35),
    (7, 28), (8, 29), (9, 26), (10, 27),
    (11, 23), (12, 24), (13, 25), (14, 22),
]

YOLO_TO_CUSTOM_ID = [i for i in range(1, 36) if i not in (15, 16, 17, 18)]
_i19, _i21 = YOLO_TO_CUSTOM_ID.index(19), YOLO_TO_CUSTOM_ID.index(21)
YOLO_TO_CUSTOM_ID[_i19], YOLO_TO_CUSTOM_ID[_i21] = 21, 19

def _build_world_coords() -> dict:
    m = {k: v for k, v in FIBA_M_COORDS.items()}
    for l_id, r_id in SYMMETRY_PAIRS_1BASED:
        if l_id in FIBA_M_COORDS:
            m[r_id] = (COURT_L - FIBA_M_COORDS[l_id][0], FIBA_M_COORDS[l_id][1])
    return {yi: m[ci] for yi, ci in enumerate(YOLO_TO_CUSTOM_ID) if ci in m}

WORLD_COORDS = _build_world_coords()

def _arc_pts(cx, cy, r, a_start_deg, a_end_deg, step=0.015):
    arc_len = abs(np.radians(a_end_deg - a_start_deg)) * r
    n = max(4, int(arc_len / step))
    return [(cx + r * np.cos(np.radians(a_start_deg + (a_end_deg - a_start_deg) * i / n)),
             cy + r * np.sin(np.radians(a_start_deg + (a_end_deg - a_start_deg) * i / n)))
            for i in range(n + 1)]

def _line_pts(x1, y1, x2, y2, step=0.015):
    d = np.hypot(x2 - x1, y2 - y1)
    if d < 1e-6: return [(x1, y1)]
    n = max(2, int(d / step))
    return [(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n) for i in range(n + 1)]

def generate_full_arc_path():
    path = (
        _line_pts(0.0, Y_JUNCTION_TOP, X_3PT_START, Y_JUNCTION_TOP)
        + _arc_pts(X_BASKET, Y_CENTER, R_3PT, +ANGLE_JUNCTION_DEG, -ANGLE_JUNCTION_DEG)
        + _line_pts(X_3PT_START, Y_JUNCTION_BOT, 0.0, Y_JUNCTION_BOT)
    )
    return np.array(path, dtype=np.float32).reshape(-1, 1, 2)

_PATH_FORWARD = generate_full_arc_path()
PATH_LEN      = len(_PATH_FORWARD)

# Applique la direction choisie au chemin de travail
if DIRECTION == "backward":
    PATH_WORLD = _PATH_FORWARD[::-1].copy()
else:
    # "forward" et None utilisent tous les deux le chemin normal
    PATH_WORLD = _PATH_FORWARD

# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DE L'INDEX VIRTUEL DE TÊTE selon la phase courante
# ─────────────────────────────────────────────────────────────────────────────
# Trois phases communes à tous les modes :
#   1. TRACÉ   : la ligne se dessine progressivement (sauf si DIRECTION is None)
#   2. MAINTIEN: la ligne reste visible à pleine intensité (HOLD_FRAMES)
#   3. FONDU   : la tête virtuelle continue après la fin, tirant la queue hors écran
#
# Valeur retournée : virtual_head_idx (entier)
#   • [0 .. PATH_LEN[    → on dessine jusqu'à cet index (tête sur le chemin)
#   • [PATH_LEN .. PATH_LEN+TAIL_LEN]  → fondu de sortie (queue qui disparaît)
#   • > PATH_LEN+TAIL_LEN → rien à dessiner, effet terminé

def get_virtual_head_idx(effect_frame: int) -> int:
    if DIRECTION is None:
        # ── Mode instantané ──────────────────────────────────────────────────
        # Phase 1 (maintien) : tout le chemin est affiché pendant HOLD_FRAMES
        if effect_frame <= HOLD_FRAMES:
            return PATH_LEN
        # Phase 2 (fondu) : on avance la tête virtuelle pour tirer la queue
        fade_frame = effect_frame - HOLD_FRAMES
        return PATH_LEN + fade_frame * SPEED

    else:
        # ── Mode directionnel (forward / backward) ───────────────────────────
        # Phase 1 (tracé) : la tête avance de SPEED points par frame
        draw_frames = -(-PATH_LEN // SPEED)   # ceil division
        if effect_frame < draw_frames:
            return effect_frame * SPEED

        # Phase 2 (maintien) : tête figée à la fin du chemin
        hold_elapsed = effect_frame - draw_frames
        if hold_elapsed <= HOLD_FRAMES:
            return PATH_LEN

        # Phase 3 (fondu) : tête virtuelle dépasse la fin
        fade_frame = hold_elapsed - HOLD_FRAMES
        return PATH_LEN + fade_frame * SPEED


def effect_is_finished(virtual_head_idx: int) -> bool:
    """Renvoie True quand la queue a entièrement disparu de l'écran."""
    return virtual_head_idx - TAIL_LEN >= PATH_LEN


# ─────────────────────────────────────────────────────────────────────────────
# VFX — DESSIN DE LA LIGNE NÉON
# ─────────────────────────────────────────────────────────────────────────────
def _draw_neon_line(path_world: np.ndarray,
                    virtual_head_idx: int,
                    H_w2c: np.ndarray,
                    canvas_glow: np.ndarray,
                    canvas_core: np.ndarray) -> None:

    num_pts    = len(path_world)
    tail_start = virtual_head_idx - TAIL_LEN

    start_idx = max(0, tail_start)
    end_idx   = min(num_pts, virtual_head_idx)

    if start_idx >= end_idx:
        return

    indices     = list(range(start_idx, end_idx))
    if len(indices) < 2:
        return

    segment_pts = path_world[indices]
    proj        = cv2.perspectiveTransform(segment_pts, H_w2c).astype(np.int32)
    n_seg       = len(proj)

    for i in range(1, n_seg):
        p1 = tuple(proj[i - 1][0])
        p2 = tuple(proj[i][0])

        k              = indices[i]
        dist_to_head   = virtual_head_idx - k
        # t=0 → bout de queue (transparent), t=1 → tête (pleine intensité)
        t              = max(0.0, 1.0 - (dist_to_head / TAIL_LEN))

        # Échantillonnage dans la palette couleur
        palette_idx    = max(0, min(int(t * (TAIL_LEN - 1)), TAIL_LEN - 1))
        current_color  = LINE_PALETTE[palette_idx]   # BGR float32

        alpha_glow = t ** 2.2
        alpha_core = t ** 1.6

        thick_glow = 3 * M if t > 0.85 else 2 * M
        thick_core = 2 * M if t > 0.90 else 1 * M

        color_glow = tuple(float(current_color[c]) * alpha_glow * 0.75 for c in range(3))
        color_core = (255.0 * alpha_core, 255.0 * alpha_core, 255.0 * alpha_core)

        cv2.line(canvas_glow, p1, p2, color_glow, thick_glow, cv2.LINE_AA)
        cv2.line(canvas_core, p1, p2, color_core, thick_core, cv2.LINE_AA)

    # ── Tête lumineuse (uniquement si elle est encore sur le chemin réel)
    if virtual_head_idx < num_pts:
        head_2d    = tuple(proj[-1][0])
        head_color = LINE_PALETTE[-1]
        cv2.circle(canvas_glow, head_2d, 5, tuple(float(c) for c in head_color), -1, cv2.LINE_AA)
        cv2.circle(canvas_core, head_2d, 3, (255.0, 255.0, 255.0), -1, cv2.LINE_AA)


def apply_neon_line_effect(frame: np.ndarray,
                           H_cam_to_world: np.ndarray,
                           sam_masks: list,
                           effect_frame: int) -> np.ndarray:

    if H_cam_to_world is None:
        return frame

    try:
        H_w2c = np.linalg.inv(H_cam_to_world)
    except np.linalg.LinAlgError:
        return frame

    virtual_head_idx = get_virtual_head_idx(effect_frame)

    if effect_is_finished(virtual_head_idx):
        return frame

    h, w = frame.shape[:2]
    canvas_glow = np.zeros((h, w, 3), dtype=np.float32)
    canvas_core = np.zeros((h, w, 3), dtype=np.float32)

    _draw_neon_line(PATH_WORLD, virtual_head_idx, H_w2c, canvas_glow, canvas_core)

    blur_wide = cv2.GaussianBlur(canvas_glow, (31, 31), 10)
    blur_thin = cv2.GaussianBlur(canvas_core, ( 5,  5),  1.5)

    vfx = (
          blur_wide  * 1.2
        + canvas_glow * 0.6
        + blur_thin  * 1.0
        + canvas_core * 1.8
    )

    peak = 255.0
    vfx  = peak * vfx / (vfx + 120.0) * (1.0 + vfx / (peak * 3.0))
    vfx_u8 = np.clip(vfx, 0, 255).astype(np.uint8)

    if sam_masks:
        master = np.logical_or.reduce(sam_masks)
        vfx_u8[master] = 0

    return cv2.add(frame, vfx_u8)


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL UNIQUE DE L'HOMOGRAPHIE
# ─────────────────────────────────────────────────────────────────────────────
def compute_homography_once(video_path: Path, device: int = 0) -> np.ndarray:
    """
    Scanne la vidéo frame par frame jusqu'à obtenir une homographie valide,
    puis libère immédiatement le modèle de la mémoire GPU.

    Stratégie : on accumule les détections sur MAX_FRAMES_SCAN frames pour
    choisir celle qui fournit le plus grand nombre de keypoints fiables,
    ce qui maximise la précision de l'homographie finale.
    """
    MAX_FRAMES_SCAN = 60   # Nombre max de frames analysées avant de s'arrêter

    print("[INFO] Chargement YOLO Court (calcul homographie unique)...")
    from ultralytics import YOLO
    court_model = YOLO(str(CHECKPOINT_COURT))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {video_path}")

    best_H      = None
    best_n_pts  = 0
    scanned     = 0

    print(f"[INFO] Scan des {MAX_FRAMES_SCAN} premières frames pour l'homographie...")
    while scanned < MAX_FRAMES_SCAN:
        ret, frame = cap.read()
        if not ret:
            break

        res = court_model(frame, device=device, verbose=False)
        if res and res[0].keypoints is not None and len(res[0].keypoints.xy[0]) > 0:
            kxy   = res[0].keypoints.xy[0].cpu().numpy()
            kconf = res[0].keypoints.conf[0].cpu().numpy()
            src, dst = [], []
            for idx, wxy in WORLD_COORDS.items():
                if idx < len(kconf) and kconf[idx] >= CONF_KP and kxy[idx][0] > 0:
                    src.append([float(kxy[idx][0]), float(kxy[idx][1])])
                    dst.append(list(wxy))
            if len(src) >= 5 and len(src) > best_n_pts:
                H_candidate, mask = cv2.findHomography(
                    np.array(src, np.float32), np.array(dst, np.float32),
                    cv2.RANSAC, 5.0)
                if H_candidate is not None:
                    best_H     = H_candidate / H_candidate[2, 2]
                    best_n_pts = len(src)
                    print(f"[INFO]   frame {scanned:>3} — {best_n_pts} keypoints retenus ✓")

        scanned += 1

    cap.release()

    # Libération explicite du modèle (GPU + RAM)
    del court_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if best_H is None:
        sys.exit("[ERREUR] Aucune homographie valide trouvée dans les premières frames.")

    print(f"[INFO] Homographie finale calculée ({best_n_pts} points, frame la plus riche).")
    return best_H


# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(video_path: Path, output_path: Path, device: int = 0):
    torch_device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    # ── Homographie calculée une seule fois, modèle libéré ensuite
    H_cam_to_world = compute_homography_once(video_path, device)

    print("[INFO] Chargement RF-DETR...")
    from rfdetr import RFDETRMedium
    det_model = RFDETRMedium(pretrain_weights=str(CHECKPOINT_PLAYER),
                             resolution=INFERENCE_RESOLUTION)

    print("[INFO] Chargement BoT-SORT...")
    tracker = BotSort(reid_weights=REID_WEIGHTS, device=device, half=False)

    print("[INFO] Chargement SAM 2.1...")
    sam = SAM2ImagePredictor(build_sam2(CONFIG_SAM, str(CHECKPOINT_SAM),
                                        device=torch_device_str))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx    = 0
    effect_frame = 0

    START_FRAME = int(START_TIME_SEC * fps)

    # ── Résumé des paramètres actifs
    dir_label = "instantané" if DIRECTION is None else DIRECTION
    print(f"[INFO] Couleur BGR      : {LINE_COLOR_BGR}")
    print(f"[INFO] Direction        : {dir_label}")
    print(f"[INFO] Maintien         : {HOLD_FRAMES} frames ({HOLD_FRAMES / fps:.2f}s)")
    print(f"[INFO] Vitesse          : {SPEED} pts/frame  |  Queue : {TAIL_LEN} pts")
    print(f"[INFO] Effet démarre à  : t={START_TIME_SEC}s (frame {START_FRAME})")
    total_frames = (PATH_LEN + TAIL_LEN) / SPEED + HOLD_FRAMES
    print(f"[INFO] Durée estimée    : {total_frames / fps:.1f}s")

    with tqdm(total=total, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Détection joueurs + tracking
            preds = det_model.predict(frame, threshold=CONF_PLAYER)
            dets  = preds[0] if isinstance(preds, list) else preds
            boxes = []
            if dets is not None and hasattr(dets, "xyxy") and dets.xyxy is not None:
                for i, box in enumerate(dets.xyxy):
                    cid  = int(dets.class_id[i])    if hasattr(dets, "class_id")   else 0
                    conf = float(dets.confidence[i]) if hasattr(dets, "confidence") else 1.0
                    if cid == 0:
                        boxes.append([*map(int, box), conf, 0])
            det_arr = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 6))
            tracks  = tracker.update(det_arr, frame)

            # ── Masques SAM (joueurs devant la ligne)
            sam_masks = []
            if tracks is not None and len(tracks) > 0:
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16
                ):
                    sam.set_image(frame_rgb)
                    masks, _, _ = sam.predict(
                        point_coords=None, point_labels=None,
                        box=np.array([t[:4] for t in tracks]),
                        multimask_output=False)
                    sam_masks = [m.squeeze().astype(bool) for m in masks]

            # ── Rendu final
            out = frame.copy()
            if frame_idx >= START_FRAME:
                out = apply_neon_line_effect(out, H_cam_to_world, sam_masks, effect_frame)
                effect_frame += 1

            writer.write(out)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"\n[OK] Vidéo sauvegardée : {output_path}")


if __name__ == "__main__":
    process_video(SOURCE_VIDEO_PATH, OUTPUT_PATH, device=0)