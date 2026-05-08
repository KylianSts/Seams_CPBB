"""
pipeline_detect_shots.py
------------------------
Pipeline minimaliste dédiée à l'évaluation pure des tirs.
Optimisée pour TensorRT (.engine) avec lecture vidéo robuste.
"""

import argparse
import sys
from pathlib import Path

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
from core.spatial_triggers import TriggersConfig, is_ball_near_hoop
from core.detect_shots import ShotConfig, check_geometric_crossing, check_net_area_variation, get_hoop_optical_flow, check_optical_flow_signature


def run_shot_detection_pipeline(video_path: Path, device: int = 0):
    print("\n" + "="*50)
    print("🚀 INITIALISATION DE LA PIPELINE TENSORRT")
    print("="*50)

    # --- 1. Configurations ---
    det_cfg = DetectionConfig(resolution=1280)
    seg_cfg = SegmentationConfig(device=device)
    f_cfg = FiltersConfig()
    t_cfg = TriggersConfig()
    s_cfg = ShotConfig()

    # --- 2. Chargement des Modèles ---
    print("[1/2] Chargement du détecteur d'objets (TensorRT)...")
    det_model = load_object_detector(det_cfg)
    
    print("[2/2] Chargement de SAM 2...")
    sam_predictor = load_segmentation_model(seg_cfg)

    # --- 3. Flux Vidéo ---
    if not video_path.exists():
        print(f"❌ ERREUR : Impossible de trouver la vidéo -> {video_path}")
        return

    print(f"[-] Ouverture de la vidéo : {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ ERREUR : OpenCV n'arrive pas à lire la vidéo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[-] Vidéo chargée : {total_frames} frames à {fps} FPS")

    # --- 4. État & Machine à État ---
    state = MatchState()
    shot_in_progress = False
    shot_cooldown = 0
    consecutive_near_frames = 0
    frames_since_last_seen_near = 0

    print("\n🔥 DÉBUT DE L'ANALYSE (Le compteur de FPS va s'afficher)")

    # Utilisation de tqdm pour la barre de progression et les FPS
    with tqdm(total=total_frames, desc="Analyse des tirs", unit="frame", dynamic_ncols=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Fin de la vidéo
                
            state.frame_idx += 1
            if shot_cooldown > 0:
                shot_cooldown -= 1

            # --- A. DÉTECTION SPATIALE (TensorRT) ---
            det_res = run_object_detection(det_model, frame, det_cfg)
            best_ball = filter_best_ball(det_res.ball, state, f_cfg)
            
            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_res.hoops[0][:4] if det_res.hoops else None

            # Mise à jour de l'historique de la balle
            if state.ball_bbox_px:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                state.ball_history.append((state.frame_idx, (bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

            # --- B. TRIGGERS & SMART STICKY ---
            is_near = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px, t_cfg)

            # 1. Mise à jour des compteurs temporels
            if is_near:
                consecutive_near_frames += 1
                frames_since_last_seen_near = 0
            else:
                consecutive_near_frames = 0
                if shot_in_progress:
                    frames_since_last_seen_near += 1

            # 2. Condition d'Entrée
            if consecutive_near_frames >= 3 and not shot_in_progress and shot_cooldown <= 0:
                shot_in_progress = True
                frames_since_last_seen_near = 0
                tqdm.write(f"\n[Frame {state.frame_idx}] ⏳ TIR EN COURS : La balle approche du panier...")

            # 3. Traitement et Conditions de Sortie
            if shot_in_progress:
                encode_frame(sam_predictor, frame, seg_cfg)
                
                state.net_mask = get_net_mask(sam_predictor, state.hoop_bbox_px, frame.shape[1], frame.shape[0], seg_cfg)
                if state.net_mask is not None:
                    state.net_area_history.append(float(np.sum(state.net_mask)))
                
                geom_score = check_geometric_crossing(list(state.ball_history), state.hoop_bbox_px)
                net_score = check_net_area_variation(list(state.net_area_history), s_cfg)
                
                curr_flow = get_hoop_optical_flow(state.prev_frame_bgr, frame, state.hoop_bbox_px, state.net_mask, s_cfg)
                state.optical_flow_history.append(curr_flow)
                flow_score = check_optical_flow_signature(list(state.optical_flow_history), s_cfg)

                # Cas 1 : Tir réussi
                if geom_score > 0.5:
                    tqdm.write(f"[Frame {state.frame_idx}] 🏀 TIR RÉUSSI ! (Geom: {geom_score:.2f} | Net: {net_score:.2f} | Flow: {flow_score:.2f})")
                    shot_in_progress = False
                    shot_cooldown = 30 
                    consecutive_near_frames = 0 
                    frames_since_last_seen_near = 0

                # Cas 2 : Tir manqué (Sticky Timeout)
                elif frames_since_last_seen_near > 15:
                    tqdm.write(f"[Frame {state.frame_idx}] ❌ TIR MANQUÉ (ou passe). La balle a quitté la zone critique.")
                    shot_in_progress = False
                    shot_cooldown = 15
                    consecutive_near_frames = 0
                    frames_since_last_seen_near = 0

            state.prev_frame_bgr = frame.copy()
            pbar.update(1)

    cap.release()
    print("\n✅ Analyse terminée avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pense à bien vérifier que ce chemin vidéo est le bon chez toi !
    parser.add_argument("--video", type=Path, help="Chemin de la vidéo", default="data/demos/videos_raw/video_cergy_long.mp4")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    
    run_shot_detection_pipeline(args.video, args.device)