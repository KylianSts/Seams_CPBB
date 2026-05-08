"""
pipeline_detect_shots.py
------------------------
Pipeline optimisée pour l'évaluation des tirs.
Inclut un système de débuggage : sauvegarde des clips de tirs (réussis/ratés) 
avec horodatage (MM:SS) pour une analyse facile.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- Ajout du dossier racine au PATH ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Imports Core ---
from core.state import MatchState
from core.detect_objects import DetectionConfig, load_object_detector, run_object_detection
from core.segmentation import SegmentationConfig, load_segmentation_model, encode_frame, get_net_mask
from core.filters import FiltersConfig, filter_best_ball
from core.spatial_triggers import TriggersConfig, is_ball_near_hoop, is_ball_falling, has_ball_passed_above_hoop
from core.detect_shots import (
    ShotConfig, ShotManager, ShotState,
    check_geometric_crossing, check_net_area_variation, 
    get_hoop_optical_flow, check_optical_flow_signature
)


# ===========================================================================
# UTILITAIRES DE DÉBUGGAGE
# ===========================================================================

def format_video_time(raw_frame_count: int, source_fps: float) -> str:
    """Convertit un numéro de frame en temps lisible (MM:SS.ms)."""
    total_seconds = raw_frame_count / source_fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def save_debug_clip(frames: List[np.ndarray], out_path: Path, fps: float):
    """Sauvegarde une liste d'images en fichier MP4."""
    if not frames:
        return
    
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# ===========================================================================
# PIPELINE PRINCIPALE
# ===========================================================================

def run_shot_detection_pipeline(video_path: Path, device: int = 0, target_fps: Optional[int] = None):
    print("\n" + "="*50)
    print("🚀 INITIALISATION DE LA PIPELINE DE DÉBUGGAGE")
    print("="*50)

    # --- 1. Configurations ---
    det_cfg = DetectionConfig(resolution=1280)
    seg_cfg = SegmentationConfig(device=device)
    f_cfg = FiltersConfig()
    t_cfg = TriggersConfig()
    s_cfg = ShotConfig()

    # --- 2. Chargement des Modèles ---
    print("[1/3] Chargement du détecteur d'objets (TensorRT)...")
    det_model = load_object_detector(det_cfg)
    
    print("[2/3] Chargement de SAM 2...")
    sam_predictor = load_segmentation_model(seg_cfg)

    print("[3/3] Initialisation du ShotManager...")
    shot_manager = ShotManager(s_cfg)

    # --- 3. Flux Vidéo & Dossiers ---
    if not video_path.exists():
        print(f"❌ ERREUR : Vidéo introuvable -> {video_path}")
        return

    # Préparation des dossiers de sortie pour le debug
    debug_dir = video_path.parent / "debug_shots"
    scored_dir = debug_dir / "scored"
    missed_dir = debug_dir / "missed"
    scored_dir.mkdir(parents=True, exist_ok=True)
    missed_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calcul du sous-échantillonnage
    if target_fps is not None and target_fps < source_fps:
        stride = max(1, round(source_fps / target_fps))
        effective_fps = source_fps / stride
    else:
        stride = 1
        effective_fps = source_fps

    # --- 4. Buffer Vidéo (La Machine à remonter le temps) ---
    clip_duration_sec = 2.5
    buffer_capacity = max(1, int(effective_fps * clip_duration_sec))
    frame_buffer = deque(maxlen=buffer_capacity)

    # --- 5. État Applicatif ---
    state = MatchState()
    raw_frame_count = 0
    processed_frames = total_frames // stride

    print(f"\n🔥 ANALYSE EN COURS (Clips sauvegardés dans : {debug_dir})")

    with tqdm(total=processed_frames, desc="Basket Shot Analysis", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            raw_frame_count += 1

            if raw_frame_count % stride != 0:
                continue

            state.frame_idx += 1
            timestamp_str = format_video_time(raw_frame_count, source_fps)

            # --- A. DÉTECTION ET FILTRAGE ---
            det_res = run_object_detection(det_model, frame, det_cfg)
            best_ball = filter_best_ball(det_res.ball, state, f_cfg)
            
            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_res.hoops[0][:4] if det_res.hoops else None

            if state.ball_bbox_px:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                state.ball_history.append((state.frame_idx, (bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

            # --- DESSIN DU DÉBUG (Création de l'image annotée pour le clip) ---
            annotated_frame = frame.copy()
            if state.hoop_bbox_px:
                hx1, hy1, hx2, hy2 = map(int, state.hoop_bbox_px)
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (130, 60, 110), 2)
            if state.ball_bbox_px:
                bx1, by1, bx2, by2 = map(int, state.ball_bbox_px)
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (30, 200, 255), 2)
                
            frame_buffer.append(annotated_frame)

            # --- B. TRIGGERS SPATIAUX ---
            is_near = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px, t_cfg)
            is_falling = is_ball_falling(list(state.ball_history), t_cfg)
            came_from_above = has_ball_passed_above_hoop(list(state.ball_history), state.hoop_bbox_px, t_cfg)
            ball_detected = state.ball_bbox_px is not None

            # --- C. MISE À JOUR DE LA MACHINE À ÉTATS ---
            prev_manager_state = shot_manager.state
            current_manager_state = shot_manager.update(ball_detected, is_near, is_falling, came_from_above)

            # --- LOGIQUE DE DÉBUGGAGE (Sauvegarde des clips) ---
            
            # 1. Détection d'un TIR RATÉ (Le manager passe en MISSED via Timeout)
            if current_manager_state == ShotState.MISSED and prev_manager_state != ShotState.MISSED:
                tqdm.write(f"[{timestamp_str}] ❌ TIR RATÉ (Timeout) : Sauvegarde du clip...")
                # Formatage du nom de fichier sans les ":" qui posent problème à Windows
                safe_time = timestamp_str.replace(':', 'm').replace('.', 's')
                save_debug_clip(list(frame_buffer), missed_dir / f"miss_{safe_time}.mp4", effective_fps)

            # --- D. ANALYSE PHYSIQUE LOURDE (Seulement si nécessaire) ---
            if shot_manager.is_tracking:
                encode_frame(sam_predictor, frame, seg_cfg)
                state.net_mask = get_net_mask(sam_predictor, state.hoop_bbox_px, frame.shape[1], frame.shape[0], seg_cfg)
                
                if state.net_mask is not None:
                    state.net_area_history.append(float(np.sum(state.net_mask)))
                
                geom_score = check_geometric_crossing(list(state.ball_history), state.hoop_bbox_px)
                net_score = check_net_area_variation(list(state.net_area_history), s_cfg)
                
                curr_flow = get_hoop_optical_flow(state.prev_frame_bgr, frame, state.hoop_bbox_px, state.net_mask, s_cfg)
                state.optical_flow_history.append(curr_flow)
                
                # 2. Détection d'un TIR MARQUÉ
                if geom_score > 0.5:
                    tqdm.write(f"[{timestamp_str}] 🏀 PANIER MARQUÉ ! (Geom: {geom_score:.2f} | Net: {net_score:.2f})")
                    shot_manager.register_success()
                    
                    safe_time = timestamp_str.replace(':', 'm').replace('.', 's')
                    save_debug_clip(list(frame_buffer), scored_dir / f"score_{safe_time}.mp4", effective_fps)

            state.prev_frame_bgr = frame.copy()
            pbar.update(1)

    cap.release()
    print(f"\n✅ Analyse terminée. Les vidéos de débug sont dans : {debug_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default="data/demos/videos_raw/cergy_1.mp4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None, help="FPS cible pour accélérer le traitement (ex: 15)")
    args = parser.parse_args()
    
    run_shot_detection_pipeline(args.video, args.device, target_fps=args.fps)