"""
pipeline_detect_shots.py
------------------------
Pipeline optimisée pour l'évaluation et le débugage des tirs.
Sauvegarde des clips de 2 secondes (1s avant + 1s après) en arrière-plan
via un système de capture post-événement non-bloquant.
"""

import argparse
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.detect_objects import DetectionConfig, load_object_detector, run_object_detection
from core.detect_shots import (
    ShotConfig, ShotManager, ShotState,
    check_geometric_crossing, check_net_area_variation,
    get_hoop_optical_flow, check_optical_flow_signature,
)
from core.filters import FiltersConfig, filter_best_ball
from core.segmentation import SegmentationConfig, encode_frame, get_net_mask, load_segmentation_model
from core.spatial_triggers import TriggersConfig, has_ball_passed_above_hoop, is_ball_falling, is_ball_near_hoop
from core.state import MatchState


# ===========================================================================
# STRUCTURES DE DONNÉES
# ===========================================================================

@dataclass
class PendingClip:
    """
    Représente un clip en cours de construction après la détection d'un événement.
    Conserve les frames pré-événement et accumule les frames post-événement.
    """
    pre_frames:       List[np.ndarray]       # Frames capturées AVANT l'événement
    post_frames:      List[np.ndarray] = field(default_factory=list)
    post_frames_needed: int = 0              # Nombre de frames à capturer APRÈS
    label:            str  = "unknown"       # "score" ou "miss"
    timestamp_str:    str  = ""
    geom_score:       float = 0.0
    net_score:        float = 0.0
    out_path:         Optional[Path] = None


# ===========================================================================
# UTILITAIRES
# ===========================================================================

def format_video_time(raw_frame_count: int, source_fps: float) -> str:
    """Convertit un numéro de frame en timestamp lisible (MM:SS.ms)."""
    total_seconds = raw_frame_count / source_fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}m{seconds:05.2f}s"


def _encode_clip(frames: List[np.ndarray], out_path: Path, fps: float) -> None:
    """Encode et sauvegarde un clip vidéo dans un thread séparé."""
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_clip_async(frames: List[np.ndarray], out_path: Path, fps: float) -> None:
    """Lance l'encodage d'un clip dans un thread non-bloquant."""
    thread = threading.Thread(target=_encode_clip, args=(frames, out_path, fps), daemon=True)
    thread.start()


def annotate_frame(
    frame: np.ndarray,
    hoop_bbox: Optional[tuple],
    ball_bbox: Optional[tuple],
    geom_score: float = 0.0,
    net_score:  float = 0.0,
    label:      str   = "",
) -> np.ndarray:
    """Applique les annotations de débugage sur une copie de la frame."""
    out = frame.copy()

    if hoop_bbox:
        hx1, hy1, hx2, hy2 = map(int, hoop_bbox)
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (130, 60, 110), 2, cv2.LINE_AA)

    if ball_bbox:
        bx1, by1, bx2, by2 = map(int, ball_bbox)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        cv2.circle(out, (cx, cy), 11, (30, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 7,  (30, 200, 255), -1, cv2.LINE_AA)

    if label:
        color  = (100, 255, 100) if "score" in label else (50, 100, 255)
        banner = f"{label.upper()}  Geom:{geom_score:.2f}  Net:{net_score:.2f}"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        h_img, w_img = out.shape[:2]
        bx1_b = (w_img - tw) // 2 - 10
        by1_b = 12
        cv2.rectangle(out, (bx1_b, by1_b), (bx1_b + tw + 20, by1_b + th + 12), (15, 18, 22), -1)
        cv2.rectangle(out, (bx1_b, by1_b), (bx1_b + tw + 20, by1_b + th + 12), color, 1)
        cv2.putText(out, banner, (bx1_b + 10, by1_b + th + 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1, cv2.LINE_AA)

    return out


# ===========================================================================
# PIPELINE PRINCIPALE
# ===========================================================================

def run_shot_detection_pipeline(
    video_path: Path,
    device:     int = 0,
    target_fps: Optional[int] = 20,
    pre_sec:    float = 1.0,
    post_sec:   float = 1.0,
) -> None:
    """
    Analyse une vidéo de basket et sauvegarde les clips autour de chaque événement de tir.

    Args:
        video_path: Chemin de la vidéo source.
        device:     Index GPU.
        target_fps: FPS de traitement cible (stride automatique). None = fps natif.
        pre_sec:    Secondes à conserver avant la détection.
        post_sec:   Secondes à capturer après la détection.
    """
    print("\n" + "=" * 60)
    print("  SHOT DETECTION PIPELINE")
    print("=" * 60)

    # --- Configurations ---
    det_cfg = DetectionConfig(resolution=1280)
    seg_cfg = SegmentationConfig(device=device)
    f_cfg   = FiltersConfig()
    t_cfg   = TriggersConfig()
    s_cfg   = ShotConfig()

    # --- Chargement des modèles ---
    print("[1/3] Chargement du détecteur (TensorRT)...")
    det_model = load_object_detector(det_cfg)

    print("[2/3] Chargement de SAM 2...")
    sam_predictor = load_segmentation_model(seg_cfg)

    print("[3/3] Initialisation du ShotManager...")
    shot_manager = ShotManager(s_cfg)

    if not video_path.exists():
        print(f"ERREUR : Vidéo introuvable -> {video_path}")
        return

    # --- Dossiers de sortie ---
    debug_dir  = video_path.parent / "debug_shots"
    scored_dir = debug_dir / "scored"
    missed_dir = debug_dir / "missed"

    # --- Ouverture de la vidéo ---
    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if target_fps is not None and target_fps < source_fps:
        stride        = max(1, round(source_fps / target_fps))
        effective_fps = source_fps / stride
    else:
        stride        = 1
        effective_fps = source_fps

    # --- Buffers ---
    pre_capacity    = max(1, int(effective_fps * pre_sec))
    post_capacity   = max(1, int(effective_fps * post_sec))
    pre_buffer: Deque[np.ndarray] = deque(maxlen=pre_capacity)  # Fenêtre glissante pré-événement

    # Liste des clips en attente de leurs frames post-événement
    pending_clips: List[PendingClip] = []

    state           = MatchState()
    raw_frame_count = 0
    processed_count = total_frames // stride

    # Scores courants (mis à jour dans la section analyse physique)
    current_geom_score = 0.0
    current_net_score  = 0.0

    print(f"\nSortie : {debug_dir}")
    print(f"Fenêtre clip : {pre_sec}s avant + {post_sec}s après = {pre_sec + post_sec}s total")
    print(f"FPS effectif : {effective_fps:.1f} ({pre_capacity} frames pré / {post_capacity} frames post)\n")

    with tqdm(total=processed_count, desc="Analyse", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame_count += 1
            if raw_frame_count % stride != 0:
                continue

            state.frame_idx += 1
            timestamp_str = format_video_time(raw_frame_count, source_fps)

            # ==============================================================
            # A. DÉTECTION & FILTRAGE
            # ==============================================================
            det_res   = run_object_detection(det_model, frame, det_cfg)
            best_ball = filter_best_ball(det_res.ball, state, f_cfg)

            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_res.hoops[0][:4] if det_res.hoops else None

            if state.ball_bbox_px:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                state.ball_history.append((state.frame_idx, (bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

            # ==============================================================
            # B. ANALYSE PHYSIQUE (SAM + Scores) — seulement si tracking actif
            # ==============================================================
            if shot_manager.is_tracking and state.hoop_bbox_px is not None:
                encode_frame(sam_predictor, frame, seg_cfg)
                state.net_mask = get_net_mask(
                    sam_predictor, state.hoop_bbox_px,
                    frame.shape[1], frame.shape[0], seg_cfg
                )
                if state.net_mask is not None:
                    state.net_area_history.append(float(np.sum(state.net_mask)))

                current_geom_score = check_geometric_crossing(list(state.ball_history), state.hoop_bbox_px)
                current_net_score  = check_net_area_variation(list(state.net_area_history), s_cfg)

                curr_flow = get_hoop_optical_flow(
                    state.prev_frame_bgr, frame, state.hoop_bbox_px, state.net_mask, s_cfg
                )
                state.optical_flow_history.append(curr_flow)

            # ==============================================================
            # C. ANNOTATION & PRÉ-BUFFER
            # ==============================================================
            annotated = annotate_frame(frame, state.hoop_bbox_px, state.ball_bbox_px)
            pre_buffer.append(annotated)

            # ==============================================================
            # D. ALIMENTATION DES CLIPS EN ATTENTE (post-événement)
            # ==============================================================
            completed: List[PendingClip] = []

            for clip in pending_clips:
                # On ré-annote les frames post avec le label de l'événement
                post_annotated = annotate_frame(
                    frame, state.hoop_bbox_px, state.ball_bbox_px,
                    label=clip.label,
                    geom_score=clip.geom_score,
                    net_score=clip.net_score,
                )
                clip.post_frames.append(post_annotated)

                if len(clip.post_frames) >= clip.post_frames_needed:
                    completed.append(clip)

            # Sauvegarde asynchrone des clips complets
            for clip in completed:
                pending_clips.remove(clip)
                all_frames = clip.pre_frames + clip.post_frames
                save_clip_async(all_frames, clip.out_path, effective_fps)
                tqdm.write(f"  → Clip sauvegardé : {clip.out_path.name}")

            # ==============================================================
            # E. MACHINE À ÉTATS — DÉTECTION DES ÉVÉNEMENTS
            # ==============================================================
            is_near        = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px, t_cfg)
            is_falling     = is_ball_falling(list(state.ball_history), t_cfg)
            came_from_above = has_ball_passed_above_hoop(list(state.ball_history), state.hoop_bbox_px, t_cfg)
            ball_detected  = state.ball_bbox_px is not None

            prev_state    = shot_manager.state
            current_state = shot_manager.update(ball_detected, is_near, is_falling, came_from_above)

            # --- Tir marqué ---
            if current_geom_score > 0.5 and shot_manager.is_tracking:
                shot_manager.register_success()

                safe_time = timestamp_str
                filename  = f"score_{safe_time}_geom{current_geom_score:.2f}_net{current_net_score:.2f}.mp4"
                out_path  = scored_dir / filename

                tqdm.write(f"[{timestamp_str}] PANIER !  Geom:{current_geom_score:.2f}  Net:{current_net_score:.2f}")

                pending_clips.append(PendingClip(
                    pre_frames        = list(pre_buffer),
                    post_frames_needed = post_capacity,
                    label             = "score",
                    timestamp_str     = timestamp_str,
                    geom_score        = current_geom_score,
                    net_score         = current_net_score,
                    out_path          = out_path,
                ))
                # Reset des scores pour éviter un double-déclenchement
                current_geom_score = 0.0
                current_net_score  = 0.0

            # --- Tir raté ---
            elif current_state == ShotState.MISSED and prev_state != ShotState.MISSED:
                safe_time = timestamp_str
                filename  = f"miss_{safe_time}_geom{current_geom_score:.2f}_net{current_net_score:.2f}.mp4"
                out_path  = missed_dir / filename

                tqdm.write(f"[{timestamp_str}] RATÉ    Geom:{current_geom_score:.2f}  Net:{current_net_score:.2f}")

                pending_clips.append(PendingClip(
                    pre_frames         = list(pre_buffer),
                    post_frames_needed = post_capacity,
                    label              = "miss",
                    timestamp_str      = timestamp_str,
                    geom_score         = current_geom_score,
                    net_score          = current_net_score,
                    out_path           = out_path,
                ))

            state.prev_frame_bgr = frame.copy()
            pbar.update(1)

    cap.release()

    # --- Vidage final : clips dont la vidéo s'est terminée avant la fin du post-buffer ---
    if pending_clips:
        print(f"\nFinalisation de {len(pending_clips)} clip(s) incomplet(s)...")
        for clip in pending_clips:
            all_frames = clip.pre_frames + clip.post_frames
            save_clip_async(all_frames, clip.out_path, effective_fps)
            tqdm.write(f"  → Clip partiel sauvegardé : {clip.out_path.name}")

    print(f"\nTerminé. Clips dans : {debug_dir}")
    print(f"  scored/ -> {scored_dir}")
    print(f"  missed/ -> {missed_dir}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de détection et débugage des tirs")
    parser.add_argument("--video",    type=Path, default=Path("data/demos/videos_raw/cergy_1.mp4"))
    parser.add_argument("--device",   type=int,  default=0)
    parser.add_argument("--fps",      type=int,  default=20,  help="FPS de traitement cible")
    parser.add_argument("--pre-sec",  type=float, default=1.0, help="Secondes avant l'événement")
    parser.add_argument("--post-sec", type=float, default=1.0, help="Secondes après l'événement")
    args = parser.parse_args()

    run_shot_detection_pipeline(
        video_path = args.video,
        device     = args.device,
        target_fps = args.fps,
        pre_sec    = args.pre_sec,
        post_sec   = args.post_sec,
    )