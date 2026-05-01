"""
run_pipeline.py
---------------
Pipeline de démonstration V2 — Analyse Broadcast de Basketball.
Orchestre l'ensemble des modules d'Intelligence Artificielle, d'heuristiques,
et de rendu pour générer une vidéo augmentée à partir d'un flux brut.
"""

import argparse
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- Ajout du dossier racine au PATH ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Imports Core ---
from core.state import MatchState
from core.video_io import add_audio_from_source

# Configurations
from core.detect_objects import DetectionConfig, load_object_detector, run_object_detection
from core.detect_court import CourtPoseConfig, load_court_detector, run_court_detection, compute_homography, smooth_homography
from core.tracking import load_tracker, update_players_tracking
from core.detect_team import TeamConfig, TeamDetector
from core.segmentation import SegmentationConfig, load_segmentation_model, encode_frame, get_players_masks, get_net_mask
from core.metrics import MetricsConfig, compute_kinematics
from core.filters import FiltersConfig, filter_top_players, filter_best_ball, filter_isolated_players, calculate_occlusion_ratios, bidirectional_smooth, get_geometric_capsule_masks
from core.spatial_triggers import TriggersConfig, is_ball_near_hoop, get_players_in_ar_zone, is_camera_stable, is_ball_falling
from core.detect_shots import ShotConfig, check_geometric_crossing, check_net_area_variation, get_hoop_optical_flow, check_optical_flow_signature
from core.incrust_logo import LogoConfig, load_ar_assets, apply_virtual_logo
from core.render import RenderConfig, MatchRenderer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURATION DU CHEF D'ORCHESTRE
# ===========================================================================

@dataclass
class PipelineConfig:
    """Paramètres temporels et comportementaux du pipeline."""
    look_ahead_frames: int = 15       # Retard du rendu pour anticiper l'avenir (0.5s à 30fps)
    
    # --- Réalité Augmentée ---
    ar_cooldown_frames: int = 15      # Frames de stabilité requises avant d'afficher l'AR
    ar_fadein_frames: int = 4         # Durée du fondu d'apparition
    ar_fadeout_frames: int =4        # Durée du fondu de disparition (anticipée)
    
    # --- Comportement Balle & Tir ---
    ball_lost_timeout: int = 30       # Frames avant d'oublier une balle perdue près du panier
    shot_cooldown_frames: int = 60    # Délai minimum entre deux tirs validés
    shot_confidence_thresh: float = 0.40 # Score composite minimum pour valider un tir
    shot_display_frames: int = 15     # Durée du feedback visuel (Panier Vert)
    
    mask_method: str = "capsule"      # "sam" ou "capsule"


# ===========================================================================
# 1. PRE-FLIGHT (Calibration GMM)
# ===========================================================================

def run_preflight_calibration(cap: cv2.VideoCapture, total_frames: int, det_model, det_cfg: DetectionConfig, filter_cfg: FiltersConfig, team_cfg: TeamConfig) -> TeamDetector:
    """Survole la vidéo pour extraire les couleurs dominantes des deux équipes."""
    detector = TeamDetector(team_cfg)
    samples_needed = 200
    stride = max(1, total_frames // samples_needed)
    
    logger.info(f"Pre-flight : Échantillonnage GMM (1 frame sur {stride})...")
    
    for i in tqdm(range(0, total_frames, stride), desc="Calibration GMM"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        det_res = run_object_detection(det_model, frame, det_cfg)
        isolated_players = filter_isolated_players(det_res.players, filter_cfg)
        detector.collect_from_raw_boxes(frame, isolated_players, filter_cfg)

    detector.calibrate()
    
    # Rembobinage crucial
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return detector


# ===========================================================================
# 2. BOUCLE PRINCIPALE
# ===========================================================================

def process_video(
    video_path: Path, output_path: Path, device: int = 0,
    enable_audio: bool = False, enable_ar: bool = True, enable_sam: bool = True
) -> None:
    
    # --- Initialisation des Configurations Métier ---
    p_cfg, f_cfg, m_cfg = PipelineConfig(), FiltersConfig(), MetricsConfig()
    t_cfg, s_cfg, r_cfg = TriggersConfig(), ShotConfig(), RenderConfig()
    
    # --- Chargement des Modèles ---
    det_cfg = DetectionConfig(resolution=1280)
    det_model = load_object_detector(det_cfg)
    
    court_cfg = CourtPoseConfig(device=device)
    court_model = load_court_detector(court_cfg)
    
    tracker = load_tracker(device=device)
    renderer = MatchRenderer(r_cfg)
    
    seg_cfg, sam_predictor = None, None
    if enable_sam:
        seg_cfg = SegmentationConfig(device=device)
        sam_predictor = load_segmentation_model(seg_cfg)

    # --- Préparation AR ---
    logo_left, logo_right = None, None
    if enable_ar:
        logo_left = LogoConfig(Path("data/demos/assets/veolia.png"), center_x_m=2.285, center_y_m=12.0, size_m=3.0)
        logo_right = LogoConfig(Path("data/demos/assets/veolia.png"), center_x_m=28.0 - 2.285, center_y_m=12.0, size_m=3.0)
        if not load_ar_assets(logo_left) or not load_ar_assets(logo_right):
            enable_ar = False

    # --- Flux Vidéo ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Calibration des équipes ---
    team_detector = run_preflight_calibration(cap, total_frames, det_model, det_cfg, f_cfg, TeamConfig())

    # --- Géométrie de Sortie (Responsive) ---
    hud_h = min(80, max(45, int(vid_h * 0.06)))
    sidebar_h = vid_h + hud_h
    sidebar_w = max(400, int((sidebar_h / 2.0 * 0.85) * (28.0 / 15.0) / 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w + sidebar_w, sidebar_h))

    # --- État Central ---
    state = MatchState()
    look_ahead_buffer = deque()

    # Registres inter-frames
    prev_hoop_bbox, prev_court_kp = None, None
    last_shot_frame = -9999
    shot_display_left = 0

    logger.info(f"Lancement du pipeline principal : {total_frames} frames...")

    with tqdm(total=total_frames, unit="frame", desc="Pipeline") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- Reset des variables volatiles ---
            state.frame_idx += 1
            state.events.clear()
            state.player_masks.clear()
            state.net_mask = None
            state.shot_scores.clear()
            state.active_triggers = {"ball_near_hoop": False, "sam_net_active": False, "sam_players_active": False, "ar_active": False}

            # ---------------------------------------------------------------
            # 1. INFÉRENCE IA
            # ---------------------------------------------------------------
            det_res = run_object_detection(det_model, frame, det_cfg)
            filtered_players = filter_top_players(det_res.players, f_cfg)
            best_ball = filter_best_ball(det_res.ball, state, f_cfg)
            
            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_res.hoops[0][:4] if det_res.hoops else None

            # ---------------------------------------------------------------
            # 2. CAMÉRA ET HOMOGRAPHIE
            # ---------------------------------------------------------------
            is_stable = is_camera_stable(state.hoop_bbox_px, prev_hoop_bbox, state.court_keypoints_px, prev_court_kp, vid_w, t_cfg)
            state.camera.is_stable = is_stable

            if is_stable:
                state.camera.stable_frames_count += 1
            else:
                state.camera.stable_frames_count = 0
                court_res = run_court_detection(court_model, frame, court_cfg)
                if court_res.keypoints_px is not None:
                    state.court_keypoints_px = court_res.keypoints_px
                    state.court_keypoints_conf = court_res.keypoints_conf
                    
                    H_new = compute_homography(court_res, court_cfg)
                    if H_new is not None:
                        state.camera.H_matrix = smooth_homography(H_new, state.camera.H_matrix) if state.camera.H_matrix is not None else H_new

            prev_hoop_bbox = state.hoop_bbox_px
            if state.court_keypoints_px is not None: prev_court_kp = state.court_keypoints_px.copy()

            # ---------------------------------------------------------------
            # 3. TRACKING & ÉQUIPES
            # ---------------------------------------------------------------
            state = update_players_tracking(tracker, filtered_players, frame, state, state.frame_idx / fps)
            
            # Extraction des preuves GMM (SANS muter l'état directement depuis l'IA)
            occlusion_ratios = calculate_occlusion_ratios(state.players, f_cfg)
            evidence = team_detector.extract_evidence(frame, state.players, f_cfg)
            
            for tid, (pA, pB) in evidence.items():
                occ = occlusion_ratios.get(tid, 0.0)
                state.players[tid].gmm_history.append((state.frame_idx, pA, pB, occ))

            # Assignation temporaire des équipes au présent ---
            current_teams = team_detector.resolve_teams(state.players, state.frame_idx)
            for tid, player in state.players.items():
                player.team_id = current_teams.get(tid, player.team_id)

            compute_kinematics(state, fps, m_cfg)

            # ---------------------------------------------------------------
            # 4. TRIGGERS SPATIAUX & MACHINE À ÉTAT
            # ---------------------------------------------------------------
            current_ball_near = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px, t_cfg)
            ball_detected = state.ball_bbox_px is not None
            
            if current_ball_near:
                ball_falling = is_ball_falling(list(state.ball_history), t_cfg)
                if not state.is_ball_near_hoop_sticky:
                    if ball_falling: # (On pourrait ajouter ici le test AABB du layup)
                        state.is_ball_near_hoop_sticky = True
                        state.last_ball_near_hoop_frame = state.frame_idx
                else:
                    state.last_ball_near_hoop_frame = state.frame_idx
            elif state.is_ball_near_hoop_sticky:
                frames_lost = state.frame_idx - state.last_ball_near_hoop_frame
                if ball_detected and frames_lost > 15:
                    state.is_ball_near_hoop_sticky = False
                elif frames_lost > p_cfg.ball_lost_timeout:
                    state.is_ball_near_hoop_sticky = False

            state.active_triggers["ball_near_hoop"] = state.is_ball_near_hoop_sticky

            # AR Machine à états
            if enable_ar and is_stable and state.camera.H_matrix is not None:
                state.ar_stable_frames += 1
                if state.ar_stable_frames >= p_cfg.ar_cooldown_frames:
                    frames_vis = state.ar_stable_frames - p_cfg.ar_cooldown_frames
                    state.ar_alpha_multiplier = min(1.0, frames_vis / p_cfg.ar_fadein_frames)
            else:
                state.ar_stable_frames = 0
                state.ar_alpha_multiplier = 0.0 

            state.active_triggers["ar_active"] = state.ar_alpha_multiplier > 0.0
            
            # Intersection AR
            players_in_ar = []
            if state.active_triggers["ar_active"]:
                for logo in [logo_left, logo_right]:
                    if logo is not None and logo._world_corners is not None and state.camera.H_matrix is not None:
                        H_inv = np.linalg.inv(state.camera.H_matrix)
                        corners_px = cv2.perspectiveTransform(logo._world_corners.reshape(1, -1, 2), H_inv).reshape(-1, 2)
                        ar_bbox = (float(corners_px[:, 0].min()), float(corners_px[:, 1].min()), float(corners_px[:, 0].max()), float(corners_px[:, 1].max()))
                        players_in_ar.extend(get_players_in_ar_zone(state.players, ar_bbox))
            
            mask_players_needed = len(set(players_in_ar)) > 0
            sam_net_needed = enable_sam and state.active_triggers["ball_near_hoop"] and state.hoop_bbox_px is not None
            
            state.active_triggers["sam_net_active"] = sam_net_needed
            state.active_triggers["sam_players_active"] = mask_players_needed

            # ---------------------------------------------------------------
            # 5. SEGMENTATION
            # ---------------------------------------------------------------
            # On encode l'image si on a besoin du filet (toujours SAM) 
            # OU si on a besoin des joueurs avec la méthode SAM.
            need_sam_encoding = sam_net_needed or (mask_players_needed and p_cfg.mask_method == "sam")
            
            if need_sam_encoding and sam_predictor is not None:
                encode_frame(sam_predictor, frame, seg_cfg)

            if mask_players_needed:
                boxes_ar = [state.players[tid].bbox_px for tid in set(players_in_ar) if tid in state.players]
                if p_cfg.mask_method == "capsule":
                    state.player_masks = get_geometric_capsule_masks(boxes_ar, frame.shape)
                elif p_cfg.mask_method == "sam" and sam_predictor:
                    state.player_masks = get_players_masks(sam_predictor, boxes_ar, seg_cfg)

            if sam_net_needed and sam_predictor:
                state.net_mask = get_net_mask(sam_predictor, state.hoop_bbox_px, vid_w, vid_h, seg_cfg)
                if state.net_mask is not None:
                    state.net_area_history.append(float(np.sum(state.net_mask)))

            # ---------------------------------------------------------------
            # 6. DÉTECTION DE TIR
            # ---------------------------------------------------------------
            if state.active_triggers["ball_near_hoop"] and state.hoop_bbox_px is not None:
                sc = {}
                sc["geometry"] = check_geometric_crossing(list(state.ball_history), state.hoop_bbox_px)
                sc["net_area"] = check_net_area_variation(list(state.net_area_history), s_cfg)
                
                curr_flow = get_hoop_optical_flow(state.prev_frame_bgr, frame, state.hoop_bbox_px, state.net_mask, s_cfg)
                state.optical_flow_history.append(curr_flow)
                sc["optical"] = check_optical_flow_signature(list(state.optical_flow_history), s_cfg)
                
                state.shot_scores = sc

                if sc["geometry"] > 0.1:
                    physical_score = (sc["net_area"] * 0.70) + (sc["optical"] * 0.30)
                    if physical_score >= p_cfg.shot_confidence_thresh:
                        if (state.frame_idx - last_shot_frame) >= p_cfg.shot_cooldown_frames:
                            state.events.append("SHOT_DETECTED")
                            last_shot_frame = state.frame_idx
                            logger.info(f"🏀 TIR DÉTECTÉ | Geom: {sc['geometry']:.2f} | Phys: {physical_score:.2f}")

            if state.ball_bbox_px:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                state.ball_history.append((state.frame_idx, (bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

            state.prev_frame_bgr = frame.copy()

            # ---------------------------------------------------------------
            # 7. LOOK-AHEAD BUFFER & RENDU (La Machine Temporelle)
            # ---------------------------------------------------------------
            look_ahead_buffer.append(state.take_snapshot(frame))

            if len(look_ahead_buffer) >= p_cfg.look_ahead_frames:
                past_snap = look_ahead_buffer.popleft()

                # A. Anticipation AR (Fade-out si la caméra VA bouger)
                frames_before_crash = p_cfg.look_ahead_frames
                for i, future_snap in enumerate(look_ahead_buffer):
                    if not future_snap.camera_stable:
                        frames_before_crash = i
                        break
                
                if frames_before_crash < p_cfg.ar_fadeout_frames:
                    ratio = frames_before_crash / float(p_cfg.ar_fadeout_frames)
                    past_snap.ar_alpha_multiplier = min(past_snap.ar_alpha_multiplier, ratio * ratio)

                # B. Feedback Visuel Tir
                if "SHOT_DETECTED" in state.events:
                    shot_display_left = p_cfg.shot_display_frames
                if shot_display_left > 0:
                    past_snap.is_perfect_shot = True
                    shot_display_left -= 1

                # C. Résolution des Équipes & Lissage Spatial (Avec le savoir du futur)
                resolved_teams = team_detector.resolve_teams(state.players, past_snap.frame_idx)
                
                for tid, past_player in past_snap.players.items():
                    past_player.team_id = resolved_teams.get(tid, past_player.team_id)
                    
                    if tid in state.players:
                        live_hist = state.players[tid].gmm_history
                        window = [h for h in live_hist if abs(h[0] - past_snap.frame_idx) <= 15]
                        if window:
                            avg_A = sum(h[1] for h in window) / len(window)
                            avg_B = sum(h[2] for h in window) / len(window)
                            past_player._debug_bidi_avg = (avg_A, avg_B, len(window))      

                    # Lissage
                    raw_hist = state.players[tid].raw_history if tid in state.players else past_player.raw_history
                    smoothed_pos = bidirectional_smooth(list(raw_hist), past_snap.frame_idx, f_cfg)
                    if smoothed_pos: past_player.court_pos_m = smoothed_pos

                # D. Rendu Final
                draw_frame = past_snap.frame_bgr.copy()
                if past_snap.active_triggers.get("ar_active") and past_snap.ar_alpha_multiplier > 0.0:
                    for logo in [logo_left, logo_right]:
                        if logo: draw_frame = apply_virtual_logo(draw_frame, past_snap, logo)

                out_frame = renderer.render_frame(draw_frame, past_snap, sidebar_w, hud_h)
                writer.write(out_frame)
            
            pbar.update(1)

    # --- Vidage du Buffer ---
    while look_ahead_buffer:
        past_snap = look_ahead_buffer.popleft()
        resolved_teams = team_detector.resolve_teams(state.players, past_snap.frame_idx)
        for tid, past_player in past_snap.players.items():
            past_player.team_id = resolved_teams.get(tid, past_player.team_id)
            
            if tid in state.players:
                live_hist = state.players[tid].gmm_history
                window = [h for h in live_hist if abs(h[0] - past_snap.frame_idx) <= 15]
                if window:
                    avg_A = sum(h[1] for h in window) / len(window)
                    avg_B = sum(h[2] for h in window) / len(window)
                    past_player._debug_bidi_avg = (avg_A, avg_B, len(window))
            
            raw_hist = state.players[tid].raw_history if tid in state.players else past_player.raw_history
            smoothed_pos = bidirectional_smooth(list(raw_hist), past_snap.frame_idx, f_cfg)
            if smoothed_pos: past_player.court_pos_m = smoothed_pos

        draw_frame = past_snap.frame_bgr.copy()
        if past_snap.active_triggers.get("ar_active") and past_snap.ar_alpha_multiplier > 0.0:
            for logo in [logo_left, logo_right]:
                if logo: draw_frame = apply_virtual_logo(draw_frame, past_snap, logo)
        writer.write(renderer.render_frame(draw_frame, past_snap, sidebar_w, hud_h))

    # --- Nettoyage ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # --- Post-Processing Audio ---
    if enable_audio:
        add_audio_from_source(video_path, output_path)

    logger.info(f"✅ Traitement terminé : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Basket")
    parser.add_argument("--video", type=Path, default=Path("data/demos/videos_raw/video_cergy_3pts.mp4"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-ar", action="store_true")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--add-audio", action="store_true", help="Associe l'audio original au rendu final")

    args = parser.parse_args()
    output_target = args.output or args.video.parent.parent / "videos_annotated" / (args.video.stem + "_test.mp4")

    process_video(
        video_path=args.video,
        output_path=output_target,
        device=args.device,
        enable_ar=not args.no_ar,
        enable_sam=not args.no_sam,
        enable_audio=args.add_audio
    )