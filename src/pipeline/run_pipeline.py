"""
run_pipeline.py
---------------
Pipeline de démonstration V2 — Analyse d'un match de basket en broadcast.

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
from collections import deque

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Modules core
# ---------------------------------------------------------------------------
from core.state import MatchState, CameraState
from core.detect_objects import DetectionConfig, load_object_detector, run_object_detection
from core.detect_court import CourtPoseConfig, load_court_detector, run_court_detection, compute_homography, smooth_homography
from core.tracking import load_tracker, update_players_tracking
from core.segmentation import SegmentationConfig, load_segmentation_model, encode_frame, get_players_masks, get_net_mask
from core.detect_shots import check_geometric_crossing, check_net_area_variation, get_hoop_optical_flow, check_optical_flow_signature
from core.detect_audio import YamnetConfig, get_match_audio_events
from core.incrust_logo import LogoConfig, load_ar_assets, apply_virtual_logo
from core.spatial_triggers import is_ball_near_hoop, get_players_in_ar_zone, is_camera_stable, is_ball_falling
from core.render import render_debug_frame
from core.metrics import compute_kinematics
from core.detect_team import TeamDetector
from core.filters import filter_top_10_players, filter_best_ball, OneEuroFilter, filter_isolated_players
from core.track_supervisor import TrackSupervisor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTES
# ===========================================================================
AR_FADEOUT_FRAMES = 6  # Durée exacte du fondu de disparition

# --- Look-Ahead Buffer ---
LOOK_AHEAD_FRAMES = 15          # Le rendu aura 15 frames de retard sur l'analyse

# --- Réalité Augmentée ---
AR_COOLDOWN_FRAMES = 15         # ~1 sec de stabilité requise avant d'afficher
AR_FADE_FRAMES = 6             # ~0.5 sec pour le fondu (fade-in)

MASK_METHOD = "capsule"
COURT_L, COURT_W = 28.0, 15.0

# --- Position des deux logos sur le terrain (symétrie) ---
_CX, _CY, _SIZE = 2.285, 12.0, 3.0
_HALF = _SIZE / 2.0

LOGO_CORNERS_LEFT = np.array([
    [_CX - _HALF, _CY - _HALF],
    [_CX + _HALF, _CY - _HALF],
    [_CX + _HALF, _CY + _HALF],
    [_CX - _HALF, _CY + _HALF],
], dtype=np.float32)

LOGO_CORNERS_RIGHT = np.array([
    [COURT_L - _CX - _HALF, _CY - _HALF],
    [COURT_L - _CX + _HALF, _CY - _HALF],
    [COURT_L - _CX + _HALF, _CY + _HALF],
    [COURT_L - _CX - _HALF, _CY + _HALF],
], dtype=np.float32)

# --- Calibration du panier ---
# --- Cooldown entre deux tirs validés ---
SHOT_COOLDOWN_FRAMES = 45       # ~1.5 sec à 30fps
BALL_LOST_TIMEOUT_FRAMES = 45   # ~1.5 sec à 30fps

# --- Seuil de confiance pour valider un tir (moyenne des scores) ---
SHOT_CONFIDENCE_THRESHOLD = 0.4

# --- Durée d'affichage du sifflet à l'écran ---
WHISTLE_DISPLAY_FRAMES = 15     # ~0.5 sec à 30fps


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
    enable_audio: bool = False,
    enable_ar: bool = True,
    enable_sam: bool = True,
    enable_supervisor: bool = False,
    enable_h_smooth: bool = True,
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
            center_y_m=_CY, 
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
        audio_events = get_match_audio_events(video_path, YamnetConfig())
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

    # --- ARCHITECTURE RESPONSIVE (Calcul parfait du Ratio FIBA) ---
    HUD_H = min(80, max(45, int(vid_h * 0.06)))
    sidebar_h_target = vid_h + HUD_H
    
    # Objectif : On veut que le terrain prenne exactement la moitié de la hauteur (50%)
    half_h = sidebar_h_target / 2.0
    
    # On retire les marges Y (environ 15% d'espace réservé pour le titre et l'aération)
    avail_h_for_court = half_h * 0.85
    
    # Le terrain fait 28m x 15m (Ratio = 1.866)
    # Pour remplir 'avail_h_for_court', la largeur idéale doit être :
    ideal_court_w = avail_h_for_court * (28.0 / 15.0)
    
    # On rajoute les marges horizontales de la sidebar (environ 8% de vide au total)
    SIDEBAR_W = int(ideal_court_w / 0.92)
    
    # Sécurité minimale
    SIDEBAR_W = max(400, SIDEBAR_W)

    # Dimensions finales de la vidéo exportée
    out_w = vid_w + SIDEBAR_W
    out_h = sidebar_h_target

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
    state.camera.kp_filter = OneEuroFilter(mincutoff=0.2, beta=0.5)

    # --- Variables de contrôle inter-frames ---
    prev_hoop_bbox      = None    # Dernière bbox panier connue (pour is_camera_stable)
    last_shot_frame     = -9999   # Frame du dernier tir validé (cooldown)
    whistle_frames_left = 0       # Compteur d'affichage du sifflet à l'écran

    logger.info(f"Lancement du traitement : {total} frames → {output_path}")

    # =======================================================================
    # BOUCLE PRINCIPALE
    # =======================================================================

    # =======================================================================
    # PRE-FLIGHT : Calibration des équipes (Optimisé V2)
    # =======================================================================
    SAMPLES_NEEDED = 200 # On vise 100 frames réparties sur tout le match
    team_detector = TeamDetector(calibration_frames=SAMPLES_NEEDED, history_size=75)
    
    # Calcul du saut en nombre de frames
    stride = max(1, total // SAMPLES_NEEDED)
    
    logger.info(f"Pre-flight : Échantillonnage de {SAMPLES_NEEDED} frames (1 frame toutes les {stride} frames)...")
    
    for i in tqdm(range(0, total, stride), desc="Calibration GMM"):
        # On saute directement à la frame ciblée
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, pf_frame = cap.read()
        if not ret: break
        
        # 1. Détection des objets sur cette frame isolée
        det_res = run_object_detection(det_model, pf_frame, det_config)
        
        # 2. On filtre pour ne garder que les joueurs isolés (tolérance de 10% d'occlusion)
        # Note: on n'utilise que det_res.players, les arbitres sont naturellement ignorés
        isolated_players = filter_isolated_players(det_res.players, max_overlap_ratio=0.10)
        
        # 3. On extrait les couleurs de ces joueurs parfaits
        team_detector.collect_from_raw_boxes(pf_frame, isolated_players)

    # On force l'entraînement du modèle avec les données collectées
    team_detector._run_calibration()
    
    # ON REMBOBINE LA VIDÉO À LA FRAME 0 POUR LE VRAI TRAITEMENT
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- INITIALISATION DU BUFFER ---
    look_ahead_buffer = deque()

    with tqdm(total=total, unit="frame", desc="Pipeline V2") as pbar:
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
            best_ball = filter_best_ball(det_result.ball, state)
            
            # On met à jour le state (pour la balle et le panier)
            state.ball_bbox_px = best_ball[:4] if best_ball else None
            state.hoop_bbox_px = det_result.hoops[0][:4] if det_result.hoops else None

            # ---------------------------------------------------------------
            # NOUVELLE ÉTAPE 2 (Ex-Etape 3) — Stabilité caméra & Homographie
            # On met à jour le repère du monde AVANT de placer les joueurs dedans
            # ---------------------------------------------------------------

            # Calcul de la stabilité (basé sur le panier)
            cam_stable_strict = is_camera_stable(state.hoop_bbox_px, prev_hoop_bbox, threshold_ratio=0.05)
            cam_stable_ar = is_camera_stable(state.hoop_bbox_px, prev_hoop_bbox, threshold_ratio=0.15)
            
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
                        # --- AMORTISSEUR D'HOMOGRAPHIE ---
                        if enable_h_smooth and state.camera.H_matrix is not None:
                            # alpha=0.10 => Lissage fort. On absorbe le choc du changement de matrice.
                            state.camera.H_matrix = smooth_homography(H_new, state.camera.H_matrix, alpha=0.1)
                        else:
                            state.camera.H_matrix = H_new
                        
                        # ---> SUPPRESSION DE _reprojet_all_players(state) ICI <---

            # Mémorisation pour la frame suivante
            if state.hoop_bbox_px is not None:
                prev_hoop_bbox = state.hoop_bbox_px

            # ---------------------------------------------------------------
            # NOUVELLE ÉTAPE 3 (Ex-Etape 2) — Tracking des joueurs
            # ---------------------------------------------------------------
            # Maintenant, l'homographie H_matrix est DÉJÀ à jour.
            # L'EMA du tracking fera son travail en douceur, sans jamais être court-circuité.
            state = update_players_tracking(
                tracker,
                filtered_players,
                frame,
                state,
                current_t=current_t,
                team_detector=team_detector,
                supervisor=supervisor
            )

            # ---------------------------------------------------------------
            # ÉTAPE 3.5 — Détection des équipes
            # ---------------------------------------------------------------
            team_detector.update(state, frame)

            # ---------------------------------------------------------------
            # ÉTAPE 4 — Calcul de la cinématique (Vitesse des joueurs)
            # ---------------------------------------------------------------
            compute_kinematics(state, fps)

            # ---------------------------------------------------------------
            # ÉTAPE 5 — Calcul des triggers spatiaux
            # ---------------------------------------------------------------

            # 1. Vérification de la présence dans la GRANDE zone (Trigger de base)
            current_ball_near = is_ball_near_hoop(state.ball_bbox_px, state.hoop_bbox_px)
            ball_detected = state.ball_bbox_px is not None
            
            # 2. Analyse du vecteur de chute (True si la balle descend vers le bas de l'écran)
            ball_falling = is_ball_falling(list(state.ball_history), window=5)
            
            # 3. EXCEPTION LAYUP / DUNK (Intersection stricte avec le panier)
            ball_very_close = False
            if ball_detected and state.hoop_bbox_px is not None:
                bx1, by1, bx2, by2 = state.ball_bbox_px
                hx1, hy1, hx2, hy2 = state.hoop_bbox_px
                
                # Tolérance : On élargit légèrement la boîte du panier pour capter le contact
                margin_x = (hx2 - hx1) * 0.05
                margin_y = (hy2 - hy1) * 0.05
                
                # Vérification d'intersection AABB (Axis-Aligned Bounding Box)
                intersect_x = (bx1 <= hx2 + margin_x) and (bx2 >= hx1 - margin_x)
                intersect_y = (by1 <= hy2 + margin_y) and (by2 >= hy1 - margin_y)
                
                if intersect_x and intersect_y:
                    ball_very_close = True

            # 4. LA MACHINE À ÉTATS (Le Gatekeeper intelligent)
            if current_ball_near:
                if not state.is_ball_near_hoop_sticky:
                    # La balle est dans la zone, mais l'analyse SAM n'est pas encore active.
                    # On l'active SI elle descend (Tir classique) OU SI elle frôle l'arceau (Layup en montée)
                    if ball_falling or ball_very_close:
                        state.is_ball_near_hoop_sticky = True
                        state.last_ball_near_hoop_frame = state.frame_idx
                else:
                    # DÉJÀ DANS LA ZONE (Mode Sticky Actif)
                    # On maintient l'état ACTIF même si la balle rebondit (ball_falling = False)
                    state.last_ball_near_hoop_frame = state.frame_idx
                    
            elif ball_detected:
                # La balle est VUE AILLEURS (hors de la grande zone) → on désactive tout immédiatement
                state.is_ball_near_hoop_sticky = False
                
            elif (state.frame_idx - state.last_ball_near_hoop_frame) > BALL_LOST_TIMEOUT_FRAMES:
                # La balle est INVISIBLE depuis trop longtemps → désactivation par timeout
                state.is_ball_near_hoop_sticky = False

            # On met à jour le dictionnaire global
            ball_near = state.is_ball_near_hoop_sticky 
            state.active_triggers["ball_near_hoop"] = ball_near

            # -------------------------------------------------------------------
            # Trigger B : Logo AR possible (Machine à états : Cooldown + Fade-in)
            # -------------------------------------------------------------------
            # 1. Si les conditions de stabilité sont réunies, on incrémente le compteur
            if enable_ar and cam_stable_ar and state.camera.H_matrix is not None:
                state.ar_stable_frames += 1
            else:
                # Disparition IMMÉDIATE au moindre mouvement brusque
                state.ar_stable_frames = 0
                state.ar_alpha_multiplier = 0.0 

            # 2. Calcul du fondu (Fade-in)
            if state.ar_stable_frames >= AR_COOLDOWN_FRAMES:
                # Le logo a le droit de s'afficher, on calcule son opacité progressive
                frames_visible = state.ar_stable_frames - AR_COOLDOWN_FRAMES
                # L'alpha monte progressivement de 0.0 à 1.0
                state.ar_alpha_multiplier = min(1.0, frames_visible / AR_FADE_FRAMES)
            else:
                state.ar_alpha_multiplier = 0.0

            # L'AR est considéré "actif" (pour déclencher SAM sur les joueurs) dès qu'il commence à apparaître
            ar_possible = state.ar_alpha_multiplier > 0.0
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

            # On active le masque joueur dès qu'un joueur est sur le logo
            mask_players_needed = len(players_in_ar) > 0
            
            # SAM Filet ne s'active que si la logique intelligente (Tir ou Layup) a validé !
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
            # -> SUPPRIMÉ D'ICI ! Le logo sera dessiné dans le passé (Étape 12)
            # On applique le logo avec les masques SAM joueurs pour l'occlusion.
            # ---------------------------------------------------------------
            #if ar_possible:
            #    for logo_cfg in [logo_left, logo_right]:
            #        if logo_cfg is not None:
            #            frame = apply_virtual_logo(frame, state, logo_cfg)

            # ---------------------------------------------------------------
            # ÉTAPE 8 — Détection de tir (3 Métriques Robustes)
            # ---------------------------------------------------------------
            if ball_near and state.hoop_bbox_px is not None:
                scores = {}

                # 1. Traversée géométrique
                scores["geometry"] = check_geometric_crossing(
                    list(state.ball_history), state.hoop_bbox_px
                )

                # 2. Variation de l'aire du filet (La fameuse cloche)
                # Note: On ne passe que l'historique, plus besoin du masque courant
                scores["net_area"] = check_net_area_variation(
                    list(state.net_area_history)
                )

                # 3. Flux optique masqué (On calcule d'abord la frame T)
                current_flow = get_hoop_optical_flow(
                    state.prev_frame_bgr, frame, state.hoop_bbox_px, state.net_mask
                )
                state.optical_flow_history.append(current_flow)
                
                # Puis on évalue la signature temporelle
                scores["optical"] = check_optical_flow_signature(
                    list(state.optical_flow_history)
                )

                state.shot_scores = scores

                # --- Validation du tir (Logique Veto + Poids) ---
                if scores["geometry"] > 0.1: # La balle a physiquement traversé l'arceau
                    
                    # SAM pèse 70%, le mouvement continu du filet 30%
                    physical_score = (scores["net_area"] * 0.70) + (scores["optical"] * 0.30)
                    
                    if physical_score >= SHOT_CONFIDENCE_THRESHOLD:
                        cooldown_ok = (state.frame_idx - last_shot_frame) >= SHOT_COOLDOWN_FRAMES

                        if cooldown_ok:
                            state.events.append("SHOT_DETECTED")
                            last_shot_frame = state.frame_idx
                            logger.info(f"  [TIR DÉTECTÉ] Frame {state.frame_idx} | Geom: {scores['geometry']:.2f} | Phys: {physical_score:.2f}")

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
            # ÉTAPE 12 — Rendu Différé (Look-Ahead Buffer)
            # ---------------------------------------------------------------
            # 1. On prend la photo de l'instant Présent (T) avec une frame vierge de logo
            snapshot = state.take_snapshot(frame)
            look_ahead_buffer.append(snapshot)

            # 2. Si le buffer est "plein", on dessine la frame du Passé (T - 15)
            if len(look_ahead_buffer) >= LOOK_AHEAD_FRAMES:
                past_snapshot = look_ahead_buffer.popleft()

                # ===============================================================
                # 🔮 LES ANTICIPATIONS VISUELLES (MAGIE DU LOOK-AHEAD)
                # ===============================================================
                
                # --- A. ANTICIPATION DU MOUVEMENT (Fade-out Doux du Logo) ---
                frames_before_crash = LOOK_AHEAD_FRAMES
                
                for i, future_snap in enumerate(look_ahead_buffer):
                    if not future_snap.camera_stable:
                        frames_before_crash = i
                        break
                
                if frames_before_crash == LOOK_AHEAD_FRAMES and not state.camera.is_stable:
                    frames_before_crash = LOOK_AHEAD_FRAMES - 1

                # Si le crash arrive dans la fenêtre de Fade-out prévue
                if frames_before_crash < AR_FADEOUT_FRAMES:
                    # Ratio linéaire de 1.0 (loin du crash) à 0.0 (crash imminent)
                    ratio = frames_before_crash / float(AR_FADEOUT_FRAMES)
                    
                    # Courbe d'Easing (Quadratique) pour un rendu "Doux"
                    # L'opacité baisse doucement au début, puis chute vite à la fin.
                    anticipated_alpha = ratio * ratio 
                    
                    past_snapshot.ar_alpha_multiplier = min(past_snapshot.ar_alpha_multiplier, anticipated_alpha)

                # --- B. ANTICIPATION DU TIR (Synchro parfaite) ---
                # Si l'état présent (T) vient tout juste de valider un tir...
                if "SHOT_DETECTED" in state.events:
                    # ... Alors à l'instant passé (T-15), la balle est exactement dans le filet !
                    past_snapshot.is_perfect_shot = True

                # ===============================================================
                
                # 3. Dessin final
                draw_frame = past_snapshot.frame_bgr.copy()
                
                if past_snapshot.active_triggers.get("ar_active", False) and past_snapshot.ar_alpha_multiplier > 0.0:
                    for logo_cfg in [logo_left, logo_right]:
                        if logo_cfg is not None:
                            draw_frame = apply_virtual_logo(draw_frame, past_snapshot, logo_cfg)

                output_frame = render_debug_frame(draw_frame, past_snapshot, sidebar_w=SIDEBAR_W, hud_h=HUD_H)
                writer.write(output_frame)

    # Vidage du Look-Ahead Buffer à la fin de la vidéo
    logger.info("Vidage final du buffer de rendu...")
    while len(look_ahead_buffer) > 0:
        past_snapshot = look_ahead_buffer.popleft()
        
        draw_frame = past_snapshot.frame_bgr.copy()
        if past_snapshot.active_triggers.get("ar_active", False):
            for logo_cfg in [logo_left, logo_right]:
                if logo_cfg is not None:
                    draw_frame = apply_virtual_logo(draw_frame, past_snapshot, logo_cfg)

        output_frame = render_debug_frame(draw_frame, past_snapshot, sidebar_w=SIDEBAR_W, hud_h=HUD_H)
        writer.write(output_frame)

    # =======================================================================
    # FIN
    # =======================================================================
    cap.release()     
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()  # Sécurité : ferme bien toutes les fenêtres d'OpenCV

    # --- AJOUT DE L'AUDIO (Muxing avec FFmpeg) ---
    if enable_audio:
        logger.info("Fusion de la piste audio d'origine avec la vidéo annotée...")
        
        # On crée le nom du fichier temporaire
        temp_video = output_path.with_name(f"temp_muet_{output_path.name}")
        
        # CRUCIAL (Surtout sur Windows) : On supprime le fichier temp s'il existait déjà 
        # pour éviter le crash de la fonction rename()
        if temp_video.exists():
            temp_video.unlink()
            
        # On renomme le fichier muet généré par OpenCV
        output_path.rename(temp_video)
        
        import subprocess
        # Commande FFmpeg ultra-robuste avec conversion H.264 universelle
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),         # Input 0: Vidéo annotée muette
            "-i", str(video_path),         # Input 1: Vidéo source originale
            
            # --- LES 3 LIGNES MAGIQUES QUI RÉGLENT TON PROBLÈME ---
            "-c:v", "libx264",             # Convertit la vidéo au standard universel H.264
            "-preset", "fast",             # Accélère la conversion
            "-pix_fmt", "yuv420p",         # Format de pixels obligatoire pour que Windows/QuickTime puisse lire la vidéo
            # ------------------------------------------------------
            
            "-map", "0:v:0",               # Garde le flux vidéo de l'input 0
            "-map", "1:a:0?",              # Le '?' évite le crash si la source n'a pas d'audio standard
            "-c:a", "aac",                 # Force un format audio universel et lisible partout
            "-shortest",                   # Coupe proprement à la fin de la vidéo
            "-loglevel", "error",
            str(output_path)               # Fichier final
        ]
        
        # Exécution de la commande avec capture d'erreurs
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Audio ajouté avec succès !")
            # On nettoie le fichier temporaire car tout s'est bien passé
            if temp_video.exists():
                temp_video.unlink()
        else:
            logger.error(f"Échec de l'ajout audio via FFmpeg. Erreur : {result.stderr}")
            logger.warning("Restauration de la vidéo muette.")
            # Restauration propre en cas de crash de FFmpeg (replace gère l'écrasement sur Windows)
            if temp_video.exists():
                temp_video.replace(output_path)
                
    logger.info(f"Vidéo de sortie sauvegardée : {output_path}")



# ===========================================================================
# ENTRÉE CLI
# ===========================================================================

if __name__ == "__main__":

    SOURCE_VIDEO_PATH = Path("data/demos/videos_raw/video_cergy_layup.mp4")
    OUTPUT_PATH       = Path("data/demos/videos_annotated/demo_test_2.mp4")

    parser = argparse.ArgumentParser(
        description="Pipeline de démonstration basket V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",  type=Path, default=SOURCE_VIDEO_PATH,
        help="Chemin de la vidéo d'entrée (mp4)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Chemin de la vidéo de sortie (mp4). "
             "Par défaut : <nom_video>_demo_v2.mp4"
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

    parser.add_argument(
        "--no-h-smooth", action="store_true",
        help="Désactiver l'amortisseur d'homographie (Technique des 4 coins)"
    )

    args = parser.parse_args()

    output = args.output or args.video.parent / (args.video.stem + "_demo_v2.mp4")

    process_video(
        video_path=args.video,
        output_path=output,
        device=args.device,
        enable_audio=not args.no_audio,
        enable_ar=not args.no_ar,
        enable_sam=not args.no_sam,
        enable_h_smooth=not args.no_h_smooth,
    )