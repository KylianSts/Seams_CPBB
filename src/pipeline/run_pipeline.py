"""
run_pipeline.py
---------------
Pipeline de démonstration V0 — Analyse d'un match de basket en broadcast.

LOGIQUE FRAME PAR FRAME :
  1.  Détection objets     → RF-DETR   (joueurs, balle, arbitres, panier)
  2.  Tracking joueurs     → BotSort + Gap Filling
  3.  Stabilité caméra     → is_camera_stable (basé sur position du panier)
  4.  YOLO-Pose + Homo     → si caméra a bougé  (sinon : recycle H + KP gris)
  5.  Calibration panier   → accumule les dims quand cam stable + balle loin
  6.  Triggers spatiaux    → ball_near_hoop / players_in_ar_zone / ar_possible
  7.  SAM (encodage unique)→ filet si ball_near_hoop | joueurs si in_ar_zone
  8.  Logo AR              → incrustation homographie + occlusion par SAM
  9.  Détection de tir     → 6 métriques calculées si ball_near_hoop
  10. Historique balle     → alimenté à chaque frame
  11. Audio (sifflets)     → pré-calculé, comparaison par timestamp
  12. Rendu                → HUD + vidéo annotée + minimap → mp4

ENTRÉE  : vidéo mp4 broadcast
SORTIE  : vidéo mp4 annotée 
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
import subprocess

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Modules core
# ---------------------------------------------------------------------------
from core.state import MatchState, CameraState
from core.detect_objects import DetectionConfig, load_object_detector, run_object_detection
from core.detect_court import CourtPoseConfig, load_court_detector, run_court_detection, compute_homography
from core.tracking import load_tracker, update_players_tracking
from core.segmentation import SegmentationConfig, load_segmentation_model, encode_frame, get_players_masks, get_net_mask
from core.detect_shots import (
    check_geometric_crossing,
    check_net_area_variation,
    check_hoop_deformation,
    get_hoop_optical_flow,
    check_ball_occlusion,
    check_ball_velocity_profile,
)
from core.detect_audio import AudioConfig, get_match_audio_events
from core.incrust_logo import LogoConfig, load_ar_assets, apply_virtual_logo
from core.spatial_triggers import is_ball_near_hoop, get_players_in_ar_zone, is_camera_stable
from core.render import render_debug_frame
from core.metrics import compute_kinematics
from core.detect_team import TeamDetector
from core.filters import filter_top_10_players, filter_best_ball, OneEuroFilterVectorized
from core.track_supervisor import TrackSupervisor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTES
# ===========================================================================

MASK_METHOD = "capsule"
COURT_L, COURT_W = 28.0, 15.0

# --- Position des deux logos sur le terrain (symétrie) ---
_CX, _CY, _SIZE = 2.285, 12.0, 2.50
_HALF = _SIZE / 2.0

LOGO_CORNERS_LEFT = np.array([
    [_CX - _HALF, _CY - _HALF],
    [_CX + _HALF, _CY - _HALF],
    [_CX + _HALF, _CY + _HALF],
    [_CX - _HALF, _CY + _HALF],
], dtype=np.float32)

LOGO_CORNERS_RIGHT = np.array([
    [COURT_L - _CX - _HALF, COURT_W - _CY - _HALF],
    [COURT_L - _CX + _HALF, COURT_W - _CY - _HALF],
    [COURT_L - _CX + _HALF, COURT_W - _CY + _HALF],
    [COURT_L - _CX - _HALF, COURT_W - _CY + _HALF],
], dtype=np.float32)

# --- Calibration du panier ---
# Nombre de frames stables (balle loin + cam stable) avant de figer la référence
HOOP_CALIB_MIN_SAMPLES = 30

# --- Cooldown entre deux tirs validés ---
SHOT_COOLDOWN_FRAMES = 45       # ~1.5 sec à 30fps
BALL_LOST_TIMEOUT_FRAMES = 90   # ~3.0 sec à 30fps

# --- Seuil de confiance pour valider un tir (moyenne des scores) ---
SHOT_CONFIDENCE_THRESHOLD = 0.4

# --- Durée d'affichage du sifflet à l'écran ---
WHISTLE_DISPLAY_FRAMES = 15     # ~0.5 sec à 30fps

# --- Taille des panneaux de rendu ---
SIDEBAR_W = 600                 # Largeur de la minimap


# ===========================================================================
# FONCTIONS UTILITAIRES INTERNES
# ===========================================================================

def _project_point(foot_pos_px: tuple, H_matrix: np.ndarray):
    """
    Transforme une position pixel (x, y) en mètres sur le terrain FIBA.
    Retourne (X_m, Y_m) ou None si H est invalide.
    """
    if H_matrix is None or foot_pos_px is None:
        return None
    pt = np.array([[[float(foot_pos_px[0]), float(foot_pos_px[1])]]], dtype=np.float32)
    try:
        res = cv2.perspectiveTransform(pt, H_matrix)
        return (float(res[0, 0, 0]), float(res[0, 0, 1]))
    except Exception:
        return None


def _get_logo_bbox_px(logo_corners_world: np.ndarray, H_matrix: np.ndarray,
                      frame_w: int, frame_h: int):
    """
    Projette les 4 coins monde d'un logo en pixels et retourne sa bbox.
    Utilisé pour alimenter get_players_in_ar_zone.
    Retourne (x1, y1, x2, y2) en pixels, ou None si H est invalide.
    """
    if H_matrix is None:
        return None
    try:
        H_inv = np.linalg.inv(H_matrix)
        corners_px = cv2.perspectiveTransform(
            logo_corners_world.reshape(1, -1, 2), H_inv
        ).reshape(-1, 2)
        x1 = float(np.clip(corners_px[:, 0].min(), 0, frame_w))
        y1 = float(np.clip(corners_px[:, 1].min(), 0, frame_h))
        x2 = float(np.clip(corners_px[:, 0].max(), 0, frame_w))
        y2 = float(np.clip(corners_px[:, 1].max(), 0, frame_h))
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _reprojet_all_players(state: MatchState) -> None:
    """
    Met à jour court_pos_m de tous les joueurs après un changement de H.
    Appelé uniquement quand H vient d'être recalculé.
    """
    for player in state.players.values():
        player.court_pos_m = _project_point(
            player.foot_pos_px, state.camera.H_matrix
        )


# ===========================================================================
# PIPELINE PRINCIPALE
# ===========================================================================

def process_video(
    video_path: Path,
    output_path: Path,
    device: int = 0,
    enable_audio: bool = True,
    enable_ar: bool = True,
    enable_sam: bool = True,
    enable_supervisor: bool = False,
) -> None:

    # =======================================================================
    # PRÉ-TRAITEMENT — Chargement des modèles et ressources
    # =======================================================================

    torch_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {torch_device}")

    # --- Modèle de détection d'objets (RF-DETR) ---
    det_config = DetectionConfig()
    det_model  = load_object_detector(det_config)

    # --- Modèle YOLO-Pose (terrain) ---
    court_config = CourtPoseConfig()
    court_model  = load_court_detector(court_config)

    # --- Tracker joueurs (BotSort) ---
    tracker = load_tracker(device=device)

    # --- Superviseur de Tracking (Anti ID-Switch) ---
    supervisor = TrackSupervisor(gmm_veto_threshold=0.75) if enable_supervisor else None

    # --- Modèle de segmentation (SAM 2.1) ---
    seg_config    = None
    sam_predictor = None
    if enable_sam:
        seg_config    = SegmentationConfig()
        sam_predictor = load_segmentation_model(seg_config)

    # --- Logos AR (gauche + droite) ---
    logo_left  = None
    logo_right = None
    if enable_ar:
        # Logo gauche : utilise les paramètres par défaut de LogoConfig
        logo_left = LogoConfig(
            center_x_m=_CX,
            center_y_m=_CY,
            size_m=_SIZE,
        )
        # Logo droite : position miroir sur le terrain
        logo_right = LogoConfig(
            center_x_m=COURT_L - _CX,
            center_y_m=COURT_W - _CY,
            size_m=_SIZE,
        )
        ok_left  = load_ar_assets(logo_left)
        ok_right = load_ar_assets(logo_right)

        if not ok_left or not ok_right:
            logger.warning("Chargement du logo AR échoué → AR désactivé.")
            enable_ar = False
            logo_left = logo_right = None

    # --- Analyse audio (pre-processing complet avant la boucle vidéo) ---
    audio_events = []
    if enable_audio:
        logger.info("Analyse audio (pre-processing)...")
        audio_events = get_match_audio_events(video_path, AudioConfig())
        logger.info(f"  → {len(audio_events)} sifflet(s) détecté(s).")

    # --- Ouverture de la vidéo source ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Impossible d'ouvrir : {video_path}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Dimensions de la frame de sortie (HUD en haut, minimap à droite)
    HUD_H = 55
    out_w = vid_w + SIDEBAR_W
    out_h = vid_h + HUD_H

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h)
    )

    # --- État central du pipeline ---
    state = MatchState()

    # Initialisation du filtre 1 Euro pour la caméra
    # mincutoff=0.01 (lissage fort au repos) | beta=0.50 (réactivité rapide en mouvement)
    state.camera.kp_filter = OneEuroFilterVectorized(mincutoff=0.001, beta=0.10)

    # --- Variables de contrôle inter-frames ---
    prev_hoop_bbox      = None    # Dernière bbox panier connue (pour is_camera_stable)
    last_shot_frame     = -9999   # Frame du dernier tir validé (cooldown)
    whistle_frames_left = 0       # Compteur d'affichage du sifflet à l'écran

    logger.info(f"Lancement du traitement : {total} frames → {output_path}")

    # =======================================================================
    # BOUCLE PRINCIPALE
    # =======================================================================

    # =======================================================================
    # PRE-FLIGHT : Calibration des équipes
    # =======================================================================
    team_detector = TeamDetector(calibration_frames=90, history_size=60, calibration_stride=5)
    logger.info("Pre-flight : Apprentissage des couleurs d'équipes...")
    
    for _ in range(team_detector.calibration_frames):
        ret, pf_frame = cap.read()
        if not ret: break
        
        # On fait juste tourner le détecteur d'objets (ultra rapide)
        det_res = run_object_detection(det_model, pf_frame, det_config)
        team_detector.collect_from_raw_boxes(pf_frame, det_res.players)

    # On force la calibration
    team_detector._run_calibration()
    
    # ON REMBOBINE LA VIDÉO À LA FRAME 0 !
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm(total=total, unit="frame", desc="Pipeline V0") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Réinitialisation des champs volatils du state ---
            state.frame_idx    += 1
            state.events        = []
            state.player_masks  = []
            state.net_mask      = None
            state.shot_scores   = {}
            state.active_triggers = {
                "ball_near_hoop":     False,
                "sam_net_active":     False,
                "sam_players_active": False,
                "ar_active":          False,
            }

            # ---------------------------------------------------------------
            # ÉTAPE 1 — Détection d'objets (RF-DETR)
            # ---------------------------------------------------------------
            det_result = run_object_detection(det_model, frame, det_config)

            # Max 10 joueurs
            filtered_players = filter_top_10_players(det_result.players)
            
            # Balle la plus logique
            best_ball = filter_best_ball(det_result.ball, state) # Attention: detect_objects.py renvoyait det_result.ball comme un Tuple ou une liste selon ta V0, passe bien la liste brute ici ! S'il ne renvoie qu'une seule balle, modifie detect_objects.py pour renvoyer 'raw_balls' au lieu du [0].
            
            # On met à jour le state (pour la balle) et on utilisera filtered_players pour le tracking
            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_result.hoops[0][:4] if det_result.hoops else None

            # ---------------------------------------------------------------
            # ÉTAPE 2 — Tracking des joueurs (BotSort + Gap Filling)
            # ---------------------------------------------------------------
            # update_players_tracking utilise state.camera.H_matrix pour projeter
            # les positions en mètres. Si H est None, court_pos_m = None.
            state = update_players_tracking(
                tracker,
                filtered_players,
                frame,
                state,
                team_detector=team_detector,
                supervisor=supervisor
            )

            # ---------------------------------------------------------------
            # ÉTAPE 2.5 — Détection des équipes
            # ---------------------------------------------------------------
            team_detector.update(state, frame)

            # ---------------------------------------------------------------
            # ÉTAPE 3 — Stabilité caméra + YOLO-Pose + Homographie
            # ---------------------------------------------------------------

            # Calcul de la stabilité (basé sur le panier)
            cam_stable_strict = is_camera_stable(state.hoop_bbox_px, prev_hoop_bbox, threshold_px=5.0)
            cam_stable_ar = is_camera_stable(state.hoop_bbox_px, prev_hoop_bbox, threshold_px=50.0)
            
            state.camera.is_stable = cam_stable_strict

            current_t = state.frame_idx / fps

            if cam_stable_strict:
                state.camera.stable_frames_count += 1
                # On nourrit le filtre avec les points FIXES pour garder l'état "chaud"
                if state.court_keypoints_px is not None:
                    state.court_keypoints_px = state.camera.kp_filter(current_t, state.court_keypoints_px)
                # H reste le même (recyclé)
            else:
                state.camera.stable_frames_count = 0
                court_result = run_court_detection(court_model, frame, court_config)
                
                if court_result.keypoints_px is not None:
                    # La transition sera fluide car x_prev était calé sur la position stable
                    state.court_keypoints_px = state.camera.kp_filter(current_t, court_result.keypoints_px)
                    state.court_keypoints_conf = court_result.keypoints_conf
                    
                    # On recalcule H seulement avec les points lissés
                    court_result.keypoints_px = state.court_keypoints_px
                    H_new = compute_homography(court_result, court_config)
                    if H_new is not None:
                        state.camera.H_matrix = H_new
                        _reprojet_all_players(state)

            # Mémorisation pour la frame suivante
            if state.hoop_bbox_px is not None:
                prev_hoop_bbox = state.hoop_bbox_px
            # ---------------------------------------------------------------
            # ÉTAPE 3.5 — Calcul de la cinématique (Vitesse des joueurs)
            # ---------------------------------------------------------------
            compute_kinematics(state, fps)

            # ---------------------------------------------------------------
            # ÉTAPE 4 — Calibration de référence du panier
            # Conditions : caméra stable ET balle loin du panier
            # But : mesurer les dimensions "au repos" du panier pour
            #       alimenter check_hoop_deformation.
            # ---------------------------------------------------------------
            if (
                state.hoop_bbox_px is not None
                and state.camera.is_stable
                and not is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px)
                and not state.is_hoop_calibrated
            ):
                hx1, hy1, hx2, hy2 = state.hoop_bbox_px
                state.hoop_dims_history.append((hx2 - hx1, hy2 - hy1))

                if len(state.hoop_dims_history) >= HOOP_CALIB_MIN_SAMPLES:
                    state.hoop_ref_w = float(np.mean([d[0] for d in state.hoop_dims_history]))
                    state.hoop_ref_h = float(np.mean([d[1] for d in state.hoop_dims_history]))
                    state.is_hoop_calibrated = True
                    logger.info(
                        f"Panier calibré : ref = "
                        f"{state.hoop_ref_w:.1f}px × {state.hoop_ref_h:.1f}px"
                    )

            # ---------------------------------------------------------------
            # ÉTAPE 5 — Calcul des triggers spatiaux
            # ---------------------------------------------------------------

            # Trigger A : Balle près du panier (Logique "Sticky" pour l'occlusion)
            current_ball_near = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px)
            ball_detected = state.ball_bbox_px is not None

            if current_ball_near:
                # 1. La balle est VUE PRÈS du panier → on active et on met à jour le chrono
                state.is_ball_near_hoop_sticky = True
                state.last_ball_near_hoop_frame = state.frame_idx
                
            elif ball_detected:
                # 2. La balle est VUE AILLEURS (hors de la zone) → on désactive immédiatement
                state.is_ball_near_hoop_sticky = False
                
            elif (state.frame_idx - state.last_ball_near_hoop_frame) > BALL_LOST_TIMEOUT_FRAMES:
                # 3. La balle est INVISIBLE depuis trop longtemps → désactivation par timeout
                state.is_ball_near_hoop_sticky = False

            # On utilise notre variable collante pour la suite du code (SAM et Shots)
            ball_near = state.is_ball_near_hoop_sticky 
            state.active_triggers["ball_near_hoop"] = ball_near

            # Trigger B : Logo AR possible (utilise la variable locale permissive !)
            ar_possible = (
                enable_ar
                and cam_stable_ar  # <-- On utilise la variable calculée à l'étape 3
                and state.camera.H_matrix is not None
            )
            state.active_triggers["ar_active"] = ar_possible
            
            # Trigger C : Joueurs qui chevauchent la zone des logos → active SAM joueurs
            players_in_ar = []
            if ar_possible:
                for corners in [LOGO_CORNERS_LEFT, LOGO_CORNERS_RIGHT]:
                    ar_bbox = _get_logo_bbox_px(
                        corners, state.camera.H_matrix, vid_w, vid_h
                    )
                    if ar_bbox is not None:
                        players_in_ar += get_players_in_ar_zone(state.players, ar_bbox)
                players_in_ar = list(set(players_in_ar))

            # On active le masque joueur dès qu'un joueur est sur le logo (indépendant de SAM)
            mask_players_needed = len(players_in_ar) > 0
            sam_net_needed      = enable_sam and ball_near and state.hoop_bbox_px is not None

            state.active_triggers["sam_net_active"]     = sam_net_needed
            state.active_triggers["sam_players_active"] = mask_players_needed

            # ---------------------------------------------------------------
            # ÉTAPE 6 — Masquage & Segmentation (Joueurs et Filet)
            # ---------------------------------------------------------------
            
            # Faut-il encoder l'image pour SAM ? (Seulement si SAM est requis)
            need_sam_encoding = sam_net_needed or (mask_players_needed and MASK_METHOD == "sam")
            
            if need_sam_encoding and sam_predictor is not None:
                encode_frame(sam_predictor, frame, seg_config)

            # 1. Masques des joueurs qui gênent le logo (Capsule ou SAM)
            if mask_players_needed:
                boxes_ar = [
                    state.players[tid].bbox_px
                    for tid in players_in_ar
                    if tid in state.players
                ]
                if boxes_ar:
                    # Appel de la fonction avec les bons arguments nommés
                    state.player_masks = get_players_masks(
                        player_boxes=boxes_ar,
                        config=seg_config,
                        predictor=sam_predictor,
                        method=MASK_METHOD,
                        frame_shape=frame.shape
                    )

            # 2. Masque du filet (Toujours SAM)
            if sam_net_needed and sam_predictor is not None:
                state.net_mask = get_net_mask(
                    sam_predictor,
                    state.hoop_bbox_px,
                    vid_w, vid_h,
                    seg_config
                )
                if state.net_mask is not None:
                    state.net_area_history.append(float(np.sum(state.net_mask)))

            # ---------------------------------------------------------------
            # ÉTAPE 7 — Incrustation du logo AR
            # On applique le logo avec les masques SAM joueurs pour l'occlusion.
            # ---------------------------------------------------------------
            if ar_possible:
                for logo_cfg in [logo_left, logo_right]:
                    if logo_cfg is not None:
                        frame = apply_virtual_logo(frame, state, logo_cfg)

            # ---------------------------------------------------------------
            # ÉTAPE 8 — Détection de tir (6 métriques)
            # Calculées uniquement quand la balle est près du panier.
            # ---------------------------------------------------------------
            if ball_near and state.hoop_bbox_px is not None:

                ball_detected  = state.ball_bbox_px is not None
                last_ball_pos  = None
                if ball_detected:
                    bx1, by1, bx2, by2 = state.ball_bbox_px
                    last_ball_pos = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

                scores = {}

                # 1. Traversée géométrique de l'arceau
                scores["geometry"] = check_geometric_crossing(
                    list(state.ball_history), state.hoop_bbox_px
                )

                # 2. Variation de l'aire du filet (SAM)
                scores["net_area"] = check_net_area_variation(
                    state.net_mask, list(state.net_area_history)
                )

                # 3. Déformation de la bbox du panier (étirement vertical)
                if state.is_hoop_calibrated:
                    _, v_h = check_hoop_deformation(
                        state.hoop_bbox_px,
                        (state.hoop_ref_w, state.hoop_ref_h)
                    )
                    scores["deform"] = float(max(0.0, v_h))
                else:
                    scores["deform"] = 0.0

                # 4. Flux optique dans la zone du panier
                scores["optical"] = get_hoop_optical_flow(
                    state.prev_frame_bgr, frame, state.hoop_bbox_px
                )

                # 5. Occlusion : balle disparaît dans le filet
                scores["occlusion"] = check_ball_occlusion(
                    ball_detected, last_ball_pos, state.hoop_bbox_px
                )

                # 6. Profil de vélocité (décélération dans le filet)
                scores["velocity"] = check_ball_velocity_profile(
                    list(state.ball_history), state.hoop_bbox_px, fps
                )

                state.shot_scores = scores

                # --- Validation du tir ---
                # Score de confiance = moyenne des métriques actives (> 0)
                active_scores = [v for v in scores.values() if v > 0.0]
                if active_scores:
                    confidence = float(np.mean(active_scores))
                    cooldown_ok = (
                        state.frame_idx - last_shot_frame
                    ) >= SHOT_COOLDOWN_FRAMES

                    if confidence >= SHOT_CONFIDENCE_THRESHOLD and cooldown_ok:
                        state.events.append("SHOT_DETECTED")
                        last_shot_frame = state.frame_idx
                        logger.info(
                            f"  [TIR DÉTECTÉ] Frame {state.frame_idx} "
                            f"| Confiance : {confidence:.2f}"
                        )

            # ---------------------------------------------------------------
            # ÉTAPE 9 — Mise à jour de l'historique de la balle
            # (Toujours à jour, pas seulement quand ball_near)
            # ---------------------------------------------------------------
            if state.ball_bbox_px is not None:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                state.ball_history.append((
                    state.frame_idx,
                    (bx1 + bx2) / 2.0,
                    (by1 + by2) / 2.0
                ))

            # ---------------------------------------------------------------
            # ÉTAPE 10 — Sifflet audio
            # On compare le timestamp courant aux événements pré-calculés.
            # ---------------------------------------------------------------
            current_ts = state.frame_idx / fps
            whistle_now = any(
                abs(current_ts - e.timestamp) <= (e.duration / 2.0)
                for e in audio_events
            )

            if whistle_now:
                whistle_frames_left = WHISTLE_DISPLAY_FRAMES

            state.is_whistle_active = (whistle_frames_left > 0)
            if whistle_frames_left > 0:
                whistle_frames_left -= 1

            # ---------------------------------------------------------------
            # ÉTAPE 11 — Mémorisation de la frame actuelle pour le flux optique
            # ---------------------------------------------------------------
            state.prev_frame_bgr = frame.copy()

            # ---------------------------------------------------------------
            # ÉTAPE 12 — Rendu et écriture
            # ---------------------------------------------------------------
            output_frame = render_debug_frame(frame, state, sidebar_w=SIDEBAR_W)
            writer.write(output_frame)

            pbar.update(1)

    # =======================================================================
    # FIN
    # =======================================================================
    cap.release()
    writer.release()

    # --- AJOUT DE L'AUDIO (Muxing avec FFmpeg) ---
    if enable_audio:
        logger.info("Fusion de la piste audio d'origine avec la vidéo annotée...")
        
        # On renomme temporairement le fichier muet qu'on vient de créer
        temp_video = output_path.with_name(f"temp_muet_{output_path.name}")
        output_path.rename(temp_video)
        
        # Commande FFmpeg pour coller la vidéo muette et l'audio source
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),         # Input 0: Vidéo annotée muette
            "-i", str(video_path),         # Input 1: Vidéo source originale
            "-c:v", "copy",                # Copie l'image sans re-compresser (Ultra rapide)
            "-c:a", "aac",                 # Encode/Copie l'audio
            "-map", "0:v:0",               # Garde le flux vidéo de l'input 0
            "-map", "1:a:0",               # Garde le flux audio de l'input 1
            "-shortest",                   # Coupe à la fin de la vidéo
            "-loglevel", "error",
            str(output_path)               # Fichier final
        ]
        
        # Exécution de la commande
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            logger.info("Audio ajouté avec succès !")
            # On nettoie le fichier temporaire
            if temp_video.exists():
                temp_video.unlink()
        else:
            logger.error("Échec de l'ajout audio via FFmpeg. La vidéo muette est conservée.")
            temp_video.rename(output_path) # Restauration en cas d'erreur
            
    logger.info(f"Vidéo de sortie sauvegardée : {output_path}")



# ===========================================================================
# ENTRÉE CLI
# ===========================================================================

if __name__ == "__main__":

    SOURCE_VIDEO_PATH = Path("data/demos/videos_raw/video_cergy_layup.mp4")
    OUTPUT_PATH       = Path("data/demos/videos_annotated/demo_test.mp4")

    parser = argparse.ArgumentParser(
        description="Pipeline de démonstration basket V0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",  type=Path, default=SOURCE_VIDEO_PATH,
        help="Chemin de la vidéo d'entrée (mp4)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Chemin de la vidéo de sortie (mp4). "
             "Par défaut : <nom_video>_demo_v0.mp4"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Index du GPU (0 par défaut, -1 pour CPU)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Désactiver l'analyse audio (détection sifflets)"
    )
    parser.add_argument(
        "--no-ar",    action="store_true",
        help="Désactiver l'incrustation du logo AR"
    )
    parser.add_argument(
        "--no-sam",   action="store_true",
        help="Désactiver la segmentation SAM2"
    )

    parser.add_argument(
        "--no-veto",  action="store_true",
        help="Désactiver le superviseur de tracking (Veto GMM)"
    )

    args = parser.parse_args()

    output = args.output or args.video.parent / (args.video.stem + "_demo_v0.mp4")

    process_video(
        video_path=args.video,
        output_path=output,
        device=args.device,
        enable_audio=not args.no_audio,
        enable_ar=not args.no_ar,
        enable_sam=not args.no_sam,
    )