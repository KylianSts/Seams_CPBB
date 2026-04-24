"""
basketball_demo.py  –  v7.1 (Ultra-Optimisé + Lissage de l'UI)
─────────────────────────────────────────────────────────────────────────────
Pipeline :
  · RF-DETR      → Détection robuste des BBoxes (invisible à l'écran)
  · BoT-SORT     → Tracking des joueurs pour garder l'ID stable (invisible)
  · SAM 2.1      → Segmentation pixel-perfect (lancée UNIQUEMENT sur la cibe)
  · Hero Focus   → Focus dynamique : Fond N&B, Joueur ciblé en couleur + Stats
  · EMA Filter   → Lissage des coordonnées de l'interface pour éviter les tremblements
"""

import argparse
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
# CHEMINS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PLAYER     = Path("models/runs/object_detection/rfdetr-medium_30ep_v1/checkpoint_best_ema.pth")
CHECKPOINT_SAM        = Path("models/weights/sam2.1_hiera_small.pt")
CONFIG_SAM            = "configs/sam2.1/sam2.1_hiera_s.yaml"
SOURCE_VIDEO_PATH     = Path("data/demos/videos_raw/video_cergy_layup.mp4")
OUTPUT_PATH           = Path("data/demos/videos_annotated/demo_cergy_stats.mp4")
REID_WEIGHTS          = Path("models/weights/osnet_x0_25_msmt17.pt")

# Paramètres du Highlight
ID_PLAYER      = 7
TIME_START     = 3.50
STATS_DURATION = 4.50

# Seuils & Réglages
CONF_PLAYER          = 0.40
CONF_BALL            = 0.30
INFERENCE_RESOLUTION = 1280
SMOOTHING_ALPHA      = 0.15  # Plus c'est proche de 0, plus c'est fluide (mais avec un léger retard)

# Couleurs
C_PLAYER = (255, 190, 0)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def load_detection_model(ckpt: Path):
    from rfdetr import RFDETRMedium
    return RFDETRMedium(pretrain_weights=str(ckpt), resolution=INFERENCE_RESOLUTION)

def load_tracker(device: int = 0) -> BotSort:
    return BotSort(
        reid_weights=REID_WEIGHTS, device=device, half=False, 
        track_high_thresh=0.45, track_low_thresh=0.15, 
        new_track_thresh=0.55, track_buffer=40, match_thresh=0.80
    )

def load_sam_model(ckpt: Path, config: str, device: str):
    sam2_model = build_sam2(config, str(ckpt), device=device)
    return SAM2ImagePredictor(sam2_model)

# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE & TRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_rfdetr_detection(model, frame: np.ndarray, conf_player: float, conf_ball: float) -> tuple:
    if model is None: return [], None
    try:
        preds = model.predict(frame, threshold=min(conf_player, conf_ball))
        dets  = preds[0] if isinstance(preds, list) else preds
        players, balls = [], []
        if hasattr(dets, 'xyxy') and dets.xyxy is not None:
            for i, box in enumerate(dets.xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf = float(dets.confidence[i]) if hasattr(dets, "confidence") else 1.0
                cid  = int(dets.class_id[i])     if hasattr(dets, "class_id")   else 0
                
                if cid == 0 and conf >= conf_player: players.append((x1, y1, x2, y2, conf, cid))
                elif cid == 1 and conf >= conf_ball: balls.append((x1, y1, x2, y2, conf, cid))

        players = sorted(players, key=lambda d: d[4], reverse=True)[:10]
        balls   = sorted(balls,   key=lambda d: d[4], reverse=True)[:1]
        return players, (balls[0] if balls else None)
    except Exception as e:
        print(f"[ERREUR RF-DETR] {e}")
        return [], None

def run_tracking(tracker: BotSort, frame: np.ndarray, player_dets: list) -> np.ndarray:
    det_arr = np.array(player_dets, dtype=np.float32)[:, :6] if player_dets else np.empty((0, 6), dtype=np.float32)
    tracks = tracker.update(det_arr, frame)
    return tracks if tracks is not None and len(tracks) > 0 else np.empty((0, 8))

def run_sam2_segmentation(sam_predictor: SAM2ImagePredictor, frame_rgb: np.ndarray, tracks: np.ndarray) -> list:
    if len(tracks) == 0: return []
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        sam_predictor.set_image(frame_rgb)
        boxes_np = np.array([t[:4] for t in tracks])
        masks, _, _ = sam_predictor.predict(point_coords=None, point_labels=None, box=boxes_np, multimask_output=False)
    return [mask.squeeze().astype(bool) for mask in masks]

# ─────────────────────────────────────────────────────────────────────────────
# HERO FOCUS DYNAMIQUE AVEC LISSAGE EMA
# ─────────────────────────────────────────────────────────────────────────────

def apply_dynamic_hero_effect(original_frame: np.ndarray, player_mask: np.ndarray, bbox: list, stats: dict, state: dict) -> np.ndarray:
    """Applique l'effet de focus avec lissage des mouvements de l'interface."""
    # 1. Fond en N&B assombri
    bg = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    bg = (bg * 0.35).astype(np.uint8) 
    
    # 2. Calcul brut des positions
    x1, y1, x2, y2 = map(int, bbox)
    h_frame, w_frame = original_frame.shape[:2]
    
    panel_w = 180
    panel_h = 30 + len(stats) * 25

    # Choix du côté et du point d'ancrage brut
    if x2 + panel_w + 30 < w_frame:
        raw_panel_x = x2 + 15
        raw_anchor_x = x2
    else:
        raw_panel_x = max(0, x1 - panel_w - 15)
        raw_anchor_x = x1
        
    raw_panel_y = max(50, y1 - 20)
    raw_anchor_y = y1 + 20

    # 3. Lissage exponentiel (EMA)
    if not state:
        # Initialisation propre lors de la première frame
        state['panel_x'] = raw_panel_x
        state['panel_y'] = raw_panel_y
        state['anchor_x'] = raw_anchor_x
        state['anchor_y'] = raw_anchor_y
    else:
        # Lissage actif : on mixe la position précédente et la nouvelle cible
        alpha = SMOOTHING_ALPHA
        state['panel_x'] = alpha * raw_panel_x + (1 - alpha) * state['panel_x']
        state['panel_y'] = alpha * raw_panel_y + (1 - alpha) * state['panel_y']
        state['anchor_x'] = alpha * raw_anchor_x + (1 - alpha) * state['anchor_x']
        state['anchor_y'] = alpha * raw_anchor_y + (1 - alpha) * state['anchor_y']

    # Récupération des entiers lissés pour l'affichage
    p_x = int(state['panel_x'])
    p_y = int(state['panel_y'])
    a_x = int(state['anchor_x'])
    a_y = int(state['anchor_y'])

    # Ligne de connexion entre le joueur et le panneau
    line_end_x = p_x + panel_w if p_x < a_x else p_x
    line_end_y = p_y + 30
    cv2.line(bg, (a_x, a_y), (line_end_x, line_end_y), (255, 255, 255), 1, cv2.LINE_AA)
    
    # Panneau de stats
    cv2.rectangle(bg, (p_x, p_y), (p_x + panel_w, p_y + panel_h), (20, 20, 25), -1)
    cv2.rectangle(bg, (p_x, p_y), (p_x + panel_w, p_y + panel_h), C_PLAYER, 2)

    dy = 25
    for key, value in stats.items():
        cv2.putText(bg, f"{key}: {value}", (p_x + 10, p_y + dy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        dy += 25

    # 4. Ré-incrustation du joueur en couleur par-dessus tout
    bg[player_mask] = original_frame[player_mask]
    return bg

# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_path: Path, conf_player: float, conf_ball: float, device: int, 
                  hl_id: int, hl_start: float, hl_duration: float, stats: dict) -> None:
    
    torch_device_str = "cpu"
    if torch.cuda.is_available():
        torch_device_str = f"cuda:{device}"
        if torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("[INFO] Chargement des modèles...")
    det_model = load_detection_model(CHECKPOINT_PLAYER)
    player_tracker = load_tracker(device)
    
    print("[INFO] Chargement de SAM 2.1...")
    try:
        sam_predictor = load_sam_model(CHECKPOINT_SAM, CONFIG_SAM, torch_device_str)
    except Exception as e:
        sys.exit(f"\n[ERREUR CRITIQUE] SAM 2: {e}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): sys.exit(f"[ERREUR] Impossible d'ouvrir : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calcul des frames de début et de fin pour le Hero Focus
    start_frame = int(hl_start * fps)
    end_frame = int((hl_start + hl_duration) * fps)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))
    
    frame_idx = 0
    hero_state = {}  # Stockage des coordonnées pour le lissage
    
    print(f"[INFO] Traitement de {total} frames → {output_path}")

    with tqdm(total=total, unit="frame", desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            # 1. Détection & Tracking (tourne en permanence mais ne s'affiche pas)
            player_dets, _ = run_rfdetr_detection(det_model, frame, conf_player, conf_ball)
            tracks = run_tracking(player_tracker, frame, player_dets)
            
            final_frame = frame.copy()
            
            # 2. Application du Hero Focus dynamique lissé (si on est dans le créneau)
            if start_frame <= frame_idx <= end_frame and hl_id != -1:
                # On cherche le joueur ciblé parmi les tracks
                target_track = None
                for t in tracks:
                    track_id = int(t[4]) if len(t) > 4 else -1
                    if track_id == hl_id:
                        target_track = t
                        break
                
                # OPTIMISATION : On lance SAM uniquement sur le joueur ciblé !
                if target_track is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sam_masks = run_sam2_segmentation(sam_predictor, frame_rgb, np.array([target_track]))
                    
                    if len(sam_masks) > 0:
                        final_frame = apply_dynamic_hero_effect(frame, sam_masks[0], target_track[:4], stats, hero_state)

            # Rendu d'une image "propre" (soit la frame d'origine intacte, soit avec le Hero Focus)
            writer.write(final_frame)
            pbar.update(1)

    cap.release(); writer.release()
    print(f"\n[OK] Vidéo sauvegardée : {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=Path,  default=SOURCE_VIDEO_PATH)
    parser.add_argument("--output",      type=Path,  default=OUTPUT_PATH)
    parser.add_argument("--conf-player", type=float, default=CONF_PLAYER)
    parser.add_argument("--conf-ball",   type=float, default=CONF_BALL)
    parser.add_argument("--device",      type=int,   default=0)
    
    # Paramètres pour le Hero Focus Dynamique
    parser.add_argument("--hl-id",       type=int,   default=ID_PLAYER, help="ID BoT-SORT du joueur à cibler.")
    parser.add_argument("--hl-start",    type=float, default=TIME_START)
    parser.add_argument("--hl-duration", type=float, default=STATS_DURATION)
    
    args = parser.parse_args()
    
    # Statistiques personnalisées
    player_stats = {
        "PTS": 24,
        "REB": 8,
        "AST": 5,
        "STL": 3
    }

    out_path = args.output if args.output else args.video.with_name(args.video.stem + "_demo_v7.1_Smooth.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    process_video(args.video, out_path, args.conf_player, args.conf_ball, args.device, 
                  args.hl_id, args.hl_start, args.hl_duration, player_stats)

if __name__ == "__main__": main()