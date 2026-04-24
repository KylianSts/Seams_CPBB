import argparse
import sys
import warnings
import subprocess
import shutil
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from boxmot import BotSort
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES PERSONNALISABLES
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ID_PLAYER      = 0
CLASS_ID_BALL        = 1
CLASS_ID_HOOP        = 3

HOOP_EPSILON         = 5.0
MAX_LOST_FRAMES      = 6

# Paramètres Filet (SAM)
SAM_NET_MARGIN       = 0.03
NET_AREA_VAR_THRESH  = 0.15

# Paramètres Effet Feu
ZONE_HIGHLIGHT_DURATION  = 3.0
TIME_TRAVEL_SECONDS      = 0.10

# 🕒 COOLDOWN : Empêche physiquement le système de recompter un tir avant 2.5s
SHOT_COOLDOWN_SECONDS    = 2.5 

# Cap joueurs & Anti-switch
MAX_PLAYERS_ON_COURT   = 10
MAX_SPEED_PX_PER_FRAME = 120
SWITCH_COOLDOWN_FRAMES = 10

# Lissage de la caméra verticale
CAMERA_SMOOTHING_SIGMA = 25

# 🌟 LOGOS & INTÉGRATION 🌟
MOTION_PIXEL_THRESHOLD = 100   
MOTION_COOLDOWN_FRAMES = 24    
LOGO_OPACITY    = 0.70
LOGO_BRIGHTNESS = 0.70
LOGO_WARMTH     = 1.15
LOGO_WIDTH_METERS = 3.0
CENTER_X_1, CENTER_Y_1 = 2.75, 11.5
COURT_L, COURT_W = 28.0, 15.0
CENTER_X_2, CENTER_Y_2 = COURT_L - CENTER_X_1, CENTER_Y_1

CHECKPOINT_PLAYER     = Path("models/runs/object_detection/rfdetr-medium_1280px_100ep_v1/checkpoint_best_ema.pth")
CHECKPOINT_COURT      = Path("models/runs/keypoint_detection/yolo11m-pose_1000ep_v1/weights/best.pt")
CHECKPOINT_SAM        = Path("models/weights/sam2.1_hiera_small.pt")
CONFIG_SAM            = "configs/sam2.1/sam2.1_hiera_s.yaml"
SOURCE_VIDEO_PATH     = Path("data/videos_raw/first_quarter_10s.mp4")
OUTPUT_PATH           = Path("data/demos/videos_annotated/demo_cergy_10s_vertical.mp4")
REID_WEIGHTS          = Path("models/weights/osnet_x0_25_msmt17.pt")

LOGO_1_PATH = Path("data/demos/assets/veolia.png")
LOGO_2_PATH = Path("data/demos/assets/cenergy.png")

# ─────────────────────────────────────────────────────────────────────────────
# GESTION DES LOGOS
# ─────────────────────────────────────────────────────────────────────────────
def _prepare_logo_data(path: Path, cx: float, cy: float, width_m: float):
    if not path.exists(): return None, None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: return None, None
        
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    b, g, r, a = cv2.split(img.astype(np.float32))
    b = b * LOGO_BRIGHTNESS
    g = g * (LOGO_BRIGHTNESS * (1.0 + (LOGO_WARMTH - 1.0) / 2)) 
    r = r * (LOGO_BRIGHTNESS * LOGO_WARMTH)                     
    img = cv2.merge((np.clip(b, 0, 255).astype(np.uint8),
                     np.clip(g, 0, 255).astype(np.uint8),
                     np.clip(r, 0, 255).astype(np.uint8),
                     a.astype(np.uint8)))

    h_px, w_px = img.shape[:2]
    aspect_ratio = h_px / w_px
    height_m = width_m * aspect_ratio
    half_w, half_h = width_m / 2.0, height_m / 2.0

    corners = np.array([
        [cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]
    ], dtype=np.float32)
    return corners, img

_REF_WORLD_PTS = np.array([
    [0.0, 0.0], [COURT_L, 0.0], [COURT_L, COURT_W], [0.0, COURT_W],
], dtype=np.float32).reshape(-1, 1, 2)

def _measure_camera_motion(H_prev, H_curr):
    if H_prev is None or H_curr is None: return 0.0
    try:
        pts_prev = cv2.perspectiveTransform(_REF_WORLD_PTS, np.linalg.inv(H_prev))
        pts_curr = cv2.perspectiveTransform(_REF_WORLD_PTS, np.linalg.inv(H_curr))
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.linalg.norm(pts_curr - pts_prev, axis=2).mean())

# ─────────────────────────────────────────────────────────────────────────────
# TERRAIN FIBA & MAPPING
# ─────────────────────────────────────────────────────────────────────────────
X_BASKET, X_FT, X_3PT_START = 1.575, 5.8, 2.99
Y_CENTER, Y_KEY_HALF, Y_3PT_OFFSET = 7.5, 2.45, 0.9
R_3PT, R_FT, R_DS = 6.75, 1.8, 1.25

FIBA_M_COORDS = {
    1:  (0.0,  15.0), 2:  (0.0,  15.0 - Y_3PT_OFFSET), 3:  (0.0,  Y_CENTER + Y_KEY_HALF),
    4:  (0.0,  Y_CENTER - Y_KEY_HALF), 5:  (0.0,  Y_3PT_OFFSET), 6:  (0.0,  0.0),
    7:  (X_BASKET, Y_CENTER + 1.25), 8:  (X_BASKET, Y_CENTER - 1.25),
    9:  (X_3PT_START, 15.0 - Y_3PT_OFFSET), 10: (X_3PT_START, Y_3PT_OFFSET),
    11: (X_FT, Y_CENTER + Y_KEY_HALF), 12: (X_FT, Y_CENTER), 13: (X_FT, Y_CENTER - Y_KEY_HALF),
    14: (X_BASKET + R_3PT, Y_CENTER),
    17: (14.0, 15.0), 18: (14.0, 7.5), 19: (14.0, 0.0),
}
SYMMETRY_PAIRS_1BASED = [
    (1, 30), (2, 31), (3, 32), (4, 33), (5, 34), (6, 35),
    (7, 28), (8, 29), (9, 26), (10, 27),
    (11, 23), (12, 24), (13, 25), (14, 22)
]
YOLO_TO_CUSTOM_ID = [i for i in range(1, 36) if i not in (15, 16, 17, 18)]
YOLO_TO_CUSTOM_ID[YOLO_TO_CUSTOM_ID.index(19)], YOLO_TO_CUSTOM_ID[YOLO_TO_CUSTOM_ID.index(21)] = 21, 19

def _build_world_coords():
    coords = {k: v for k, v in FIBA_M_COORDS.items()}
    for l_id, r_id in SYMMETRY_PAIRS_1BASED:
        if l_id in FIBA_M_COORDS:
            coords[r_id] = (COURT_L - FIBA_M_COORDS[l_id][0], FIBA_M_COORDS[l_id][1])
    return {idx: coords[cid] for idx, cid in enumerate(YOLO_TO_CUSTOM_ID) if cid in coords}

WORLD_COORDS = _build_world_coords()

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class CoupledCalibrator:
    def __init__(self, img_w, img_h):
        self.img_w, self.img_h = img_w, img_h
        self.cx, self.cy = img_w / 2.0, img_h / 2.0
        self.f_init = img_w * 0.8
        self.src_pts, self.dst_pts = [], []
        self.is_calibrated = False

    def add_points(self, kp_xy, kp_conf):
        if self.is_calibrated: return
        src, dst = [], []
        for yolo_idx, world_xy in WORLD_COORDS.items():
            if yolo_idx < len(kp_conf) and kp_conf[yolo_idx] > CONF_KP:
                px, py = kp_xy[yolo_idx]
                if px == 0.0 and py == 0.0: continue
                src.append([px, py])
                dst.append(world_xy)
        if len(src) >= 4:
            self.src_pts.extend(src)
            self.dst_pts.extend(dst)
        if len(self.src_pts) >= 150:
            self._optimize()

    def _optimize(self):
        src_arr, dst_arr = np.array(self.src_pts, dtype=np.float32), np.array(self.dst_pts, dtype=np.float32)
        H0, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        H0 = H0 / H0[2, 2] if H0 is not None else np.eye(3)
        p0 = [H0[0,0], H0[0,1], H0[0,2], H0[1,0], H0[1,1], H0[1,2], H0[2,0], H0[2,1], 0.0, 0.0, self.f_init]

        def cost_fn(params):
            H = np.array([[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]])
            K = np.array([[params[10], 0, self.cx], [0, params[10], self.cy], [0, 0, 1]], dtype=np.float32)
            D = np.array([params[8], params[9], 0, 0, 0], dtype=np.float32)
            pts_reshaped = src_arr.reshape(-1, 1, 2)
            undistorted  = cv2.undistortPoints(pts_reshaped, K, D)
            u_un, v_un = undistorted[:, 0, 0] * params[10] + self.cx, undistorted[:, 0, 1] * params[10] + self.cy
            pts_hom = np.stack([u_un, v_un, np.ones_like(u_un)], axis=1)
            transformed = pts_hom @ H.T
            X_pred = transformed[:, 0] / (transformed[:, 2] + 1e-7)
            Y_pred = transformed[:, 1] / (transformed[:, 2] + 1e-7)
            return np.concatenate([X_pred - dst_arr[:, 0], Y_pred - dst_arr[:, 1]])

        res = opt.least_squares(cost_fn, p0, method='lm')
        K = np.array([[res.x[10], 0, self.cx], [0, res.x[10], self.cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([res.x[8], res.x[9], 0, 0, 0], dtype=np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (self.img_w, self.img_h), 1, (self.img_w, self.img_h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (self.img_w, self.img_h), cv2.CV_32FC1)
        self.is_calibrated = True

    def undistort_frame(self, frame):
        if not self.is_calibrated: return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-ID-SWITCH & UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
CONF_PLAYER, CONF_BALL, CONF_HOOP, CONF_KP = 0.40, 0.40, 0.30, 0.50
INFERENCE_RESOLUTION, HOMOGRAPHY_SMOOTHING = 1280, 0.15

class TrackPositionGuard:
    def __init__(self, max_speed: float = MAX_SPEED_PX_PER_FRAME, cooldown: int = SWITCH_COOLDOWN_FRAMES):
        self.max_speed, self.cooldown = max_speed, cooldown
        self._last_pos, self._banned_until = {}, {}

    def filter(self, tracks: np.ndarray, frame_idx: int) -> np.ndarray:
        if len(tracks) == 0: return tracks
        keep = []
        for t in tracks:
            tid = int(t[4])
            if tid < 0: keep.append(True); continue
            if frame_idx <= self._banned_until.get(tid, -1): keep.append(False); continue
            cx, cy = ((t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0)
            if tid in self._last_pos and np.hypot(cx - self._last_pos[tid][0], cy - self._last_pos[tid][1]) > self.max_speed:
                self._banned_until[tid] = frame_idx + self.cooldown
                keep.append(False); continue
            keep.append(True)
        kept_tracks = tracks[np.array(keep, dtype=bool)]
        for t in kept_tracks:
            if int(t[4]) >= 0: self._last_pos[int(t[4])] = ((t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0)
        active_ids = {int(t[4]) for t in kept_tracks if t[4] >= 0}
        stale = [tid for tid in list(self._last_pos.keys()) if tid not in active_ids and frame_idx > self._banned_until.get(tid, -1) + 30]
        for tid in stale: self._last_pos.pop(tid, None); self._banned_until.pop(tid, None)
        return kept_tracks

def cap_to_max_players(tracks: np.ndarray, max_players: int = MAX_PLAYERS_ON_COURT) -> np.ndarray:
    if len(tracks) <= max_players: return tracks
    active = tracks[tracks[:, 5] > 0]
    gap_filled = tracks[tracks[:, 5] == 0]
    if len(active) > 0: active = active[np.argsort(-active[:, 5])]
    if len(gap_filled) > 0:
        areas = (gap_filled[:, 2] - gap_filled[:, 0]) * (gap_filled[:, 3] - gap_filled[:, 1])
        gap_filled = gap_filled[np.argsort(-areas)]
    combined = np.concatenate([active, gap_filled], axis=0) if len(active) > 0 else gap_filled
    return combined[:max_players]

def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A, B, C, D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE IA
# ─────────────────────────────────────────────────────────────────────────────
def run_rfdetr_detection(model, frame, conf_player, conf_ball, conf_hoop):
    if model is None: return [], None, []
    try:
        preds = model.predict(frame, threshold=min(conf_player, conf_ball, conf_hoop))
        dets  = preds[0] if isinstance(preds, list) else preds
        players, balls, hoops = [], [], []
        if hasattr(dets, 'xyxy') and dets.xyxy is not None:
            for i, box in enumerate(dets.xyxy):
                x1,y1,x2,y2 = map(int, box)
                conf = float(dets.confidence[i]) if hasattr(dets,'confidence') else 1.0
                cid  = int(dets.class_id[i])     if hasattr(dets,'class_id')   else 0
                if   cid == CLASS_ID_PLAYER and conf >= conf_player: players.append((x1,y1,x2,y2,conf,cid))
                elif cid == CLASS_ID_BALL   and conf >= conf_ball:   balls.append((x1,y1,x2,y2,conf,cid))
                elif cid == CLASS_ID_HOOP   and conf >= conf_hoop:   hoops.append((x1,y1,x2,y2,conf,cid))
        return (sorted(players, key=lambda d: d[4], reverse=True)[:12],
                sorted(balls, key=lambda d: d[4], reverse=True)[0] if balls else None,
                sorted(hoops, key=lambda d: d[4], reverse=True))
    except: return [], None, []

def run_court_pose(model, frame, device):
    res = model(frame, device=device, verbose=False)
    if not res or res[0].keypoints is None or res[0].keypoints.xy.shape[0] == 0: return None, None
    return res[0].keypoints.xy[0].cpu().numpy(), res[0].keypoints.conf[0].cpu().numpy()

def run_tracking_with_gap_filling(tracker: BotSort, frame, player_dets) -> np.ndarray:
    active_tracks = tracker.update(np.array(player_dets, dtype=np.float32)[:,:6] if player_dets else np.empty((0,6), dtype=np.float32), frame)
    aug = list(active_tracks) if active_tracks is not None else []
    active_ids = {int(t[4]) for t in aug} if aug else set()
    if hasattr(tracker, 'lost_stracks'):
        current_frame = getattr(tracker, 'frame_id', 0)
        img_h, img_w  = frame.shape[:2]
        for t in tracker.lost_stracks:
            tid = int(t.track_id)
            if tid not in active_ids and 0 < (current_frame - getattr(t,'frame_id',current_frame)) <= MAX_LOST_FRAMES:
                x1,y1,x2,y2 = getattr(t,'tlbr',(0,0,0,0))
                if ((np.clip(x2,0,img_w)-np.clip(x1,0,img_w)) > 10 and (np.clip(y2,0,img_h)-np.clip(y1,0,img_h)) > 10):
                    aug.append([np.clip(x1,0,img_w), np.clip(y1,0,img_h), np.clip(x2,0,img_w), np.clip(y2,0,img_h), tid, 0.0, 0, -1])
    return np.array(aug, dtype=np.float32) if aug else np.empty((0, 8))

def run_sam2_segmentation(sam_predictor, tracks):
    if len(tracks) == 0: return []
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        masks, _, _ = sam_predictor.predict(box=np.array([t[:4] for t in tracks]), multimask_output=False)
    return [mask.squeeze().astype(bool) for mask in masks]

def run_sam2_net_segmentation(sam_predictor, hoop_box, img_w, img_h, margin=0.03):
    x1, y1, x2, y2 = hoop_box
    w, h = x2 - x1, y2 - y1
    nx1, ny1 = max(0, x1 - w * margin), max(0, y1 - h * margin)
    nx2, ny2 = min(img_w, x2 + w * margin), min(img_h, y2 + h * margin)
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        masks, _, _ = sam_predictor.predict(box=np.array([[nx1, ny1, nx2, ny2]]), multimask_output=False)
    return masks[0].squeeze().astype(bool)

def compute_homography(kp_xy, kp_conf):
    if kp_xy is None or kp_conf is None: return None
    src, dst = [], []
    for yolo_idx, world_xy in WORLD_COORDS.items():
        if yolo_idx >= len(kp_conf) or kp_conf[yolo_idx] < CONF_KP: continue
        px, py = kp_xy[yolo_idx]
        if px == 0.0 and py == 0.0: continue
        src.append([float(px), float(py)])
        dst.append(list(world_xy))
    if len(src) < 5: return None
    H, _ = cv2.findHomography(np.array(src,np.float32), np.array(dst,np.float32), cv2.RANSAC, 5.0)
    return H

def apply_virtual_logo(frame, logos_data, H_frame_to_world, all_masks):
    if H_frame_to_world is None: return frame
    h_frame, w_frame = frame.shape[:2]
    try: H_world_to_frame = np.linalg.inv(H_frame_to_world)
    except: return frame
    out_frame = frame.copy()
    for target_world_corners, logo_img in logos_data:
        if logo_img is None: continue
        h_logo, w_logo = logo_img.shape[:2]
        H_logo_to_world, _ = cv2.findHomography(np.array([[0,0],[w_logo,0],[w_logo,h_logo],[0,h_logo]], dtype=np.float32), target_world_corners)
        warped_logo = cv2.warpPerspective(logo_img, H_world_to_frame @ H_logo_to_world, (w_frame, h_frame), flags=cv2.INTER_LINEAR)
        warped_rgb, warped_alpha = warped_logo[...,:3], (warped_logo[...,3].astype(np.float32) / 255.0) * LOGO_OPACITY
        for mask in all_masks: warped_alpha[mask] = 0.0
        for c in range(3): out_frame[...,c] = out_frame[...,c]*(1.0-warped_alpha) + warped_rgb[...,c]*warped_alpha
    return out_frame.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# RENDU VISUEL : L'EMBRASEMENT DU FILET (SAM)
# ─────────────────────────────────────────────────────────────────────────────
def annotate_frame(frame, sam_masks, highlight_frames, total_highlight_frames, net_mask) -> np.ndarray:
    out = frame.copy()
    
    # Masques SAM pour passer l'effet DERRIÈRE les joueurs
    h_img, w_img = out.shape[:2]
    combined_sam_mask = np.zeros((h_img, w_img), dtype=bool)
    for mask in sam_masks:
        if mask.shape == combined_sam_mask.shape: 
            combined_sam_mask |= mask

    # 🔥 EFFET EMBRASEMENT DU FILET (FEU) 🔥
    if highlight_frames > 0 and net_mask is not None:
        
        # Calcul du fondu temporel
        fade = highlight_frames / float(total_highlight_frames)
        
        if fade > 0.05:
            fire_layer = np.zeros_like(out, dtype=np.float32)
            
            # 1. Couche Extérieure (Halo Rouge profond/électrique - Grand flou)
            red_glow = np.zeros_like(out)
            red_glow[net_mask] = (0, 0, 255) # BGR (Rouge pur)
            red_glow = cv2.GaussianBlur(red_glow, (45, 45), 0)
            
            # 2. Couche Médiane (Orange Flamboyant - Flou moyen)
            orange_glow = np.zeros_like(out)
            orange_glow[net_mask] = (0, 120, 255) # BGR (Orange)
            orange_glow = cv2.GaussianBlur(orange_glow, (21, 21), 0)
            
            # 3. Cœur du filet (Jaune/Blanc incandescent - Très peu flou)
            yellow_core = np.zeros_like(out)
            yellow_core[net_mask] = (150, 255, 255) # BGR (Jaune très clair)
            yellow_core = cv2.GaussianBlur(yellow_core, (7, 7), 0)

            # Addition des couches avec des multiplicateurs d'intensité
            fire_layer += red_glow.astype(np.float32) * 1.5
            fire_layer += orange_glow.astype(np.float32) * 1.2
            fire_layer += yellow_core.astype(np.float32) * 1.0
            
            # Effet "Flamme qui vacille" (Flicker)
            flicker = np.random.uniform(0.85, 1.15)
            
            # Application de l'intensité globale (fade temporel + flicker)
            fire_layer = fire_layer * fade * flicker
            
            # Effacement de l'effet là où se trouvent les joueurs
            fire_layer[combined_sam_mask] = 0
            
            # Application du calque avec fusion additive
            fire_layer_uint8 = np.clip(fire_layer, 0, 255).astype(np.uint8)
            out = cv2.addWeighted(out, 1.0, fire_layer_uint8, 1.0, 0)
                
    return out

# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(video_path: Path, output_path: Path, conf_player: float, conf_ball: float, conf_kp: float, device: int) -> None:
    global CONF_KP
    CONF_KP = conf_kp

    torch_device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    print("[INFO] Chargement des modèles...")
    from ultralytics import YOLO
    from rfdetr import RFDETRMedium
    
    court_pose_model = YOLO(str(CHECKPOINT_COURT))
    det_model        = RFDETRMedium(pretrain_weights=str(CHECKPOINT_PLAYER), resolution=INFERENCE_RESOLUTION)
    player_tracker   = BotSort(reid_weights=REID_WEIGHTS, device=device, half=False, track_high_thresh=0.45, track_low_thresh=0.15, new_track_thresh=0.55, track_buffer=60, match_thresh=0.80)
    sam_predictor    = SAM2ImagePredictor(build_sam2(CONFIG_SAM, str(CHECKPOINT_SAM), device=torch_device_str))

    raw_logos_data = [
        _prepare_logo_data(LOGO_1_PATH, CENTER_X_1, CENTER_Y_1, LOGO_WIDTH_METERS),
        _prepare_logo_data(LOGO_2_PATH, CENTER_X_2, CENTER_Y_2, LOGO_WIDTH_METERS),
    ]
    logos_data = [d for d in raw_logos_data if d[0] is not None]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): sys.exit(f"[ERREUR] Impossible d'ouvrir : {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # --- PASSE 1 : Caméra ---
    print(f"\n[INFO] PASSE 1/2 : Analyse de la trajectoire pour caméra fluide...")
    ball_xs = []
    with tqdm(total=total, unit="frame", desc="Pass 1 (Tracking Camera)") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            _, ball_det, _ = run_rfdetr_detection(det_model, frame, conf_player, conf_ball, CONF_HOOP)
            ball_xs.append((ball_det[0] + ball_det[2]) / 2.0 if ball_det is not None else np.nan)
            pbar.update(1)
    cap.release()

    ball_xs = np.array(ball_xs, dtype=np.float64)
    nans, x = np.isnan(ball_xs), lambda z: z.nonzero()[0]
    if np.all(nans): ball_xs[:] = vid_w / 2.0
    else: ball_xs[nans] = np.interp(x(nans), x(~nans), ball_xs[~nans])
    smoothed_cx = gaussian_filter1d(ball_xs, sigma=CAMERA_SMOOTHING_SIGMA)

    # --- PASSE 2 : Rendu ---
    print(f"\n[INFO] PASSE 2/2 : Rendu Final (Format Vertical + Embrasement Filet)...")
    cap = cv2.VideoCapture(str(video_path))
    
    # 🚨 Fichier temp pour l'audio
    temp_output_path = output_path.with_name(output_path.stem + "_temp.mp4")
    target_w = min(int(vid_h * 9 / 16), vid_w)
    writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, vid_h))

    H_smooth, H_smooth_prev = None, None
    motion_cooldown = 0

    calibrator       = CoupledCalibrator(vid_w, vid_h)
    position_guard   = TrackPositionGuard() 

    prev_hoop_box, prev_court_kp_xy, prev_court_kp_conf = None, None, None
    ball_history, net_area_history = deque(maxlen=40), deque(maxlen=5) 
    
    frame_count, geom_flag_frames, net_flag_frames, last_shot_frame = 0, 0, 0, -9999 

    delay_frames           = int(fps * TIME_TRAVEL_SECONDS)
    frame_queue            = deque()
    highlight_events       = {}
    current_highlight_frames = 0
    total_highlight_frames   = int(fps * ZONE_HIGHLIGHT_DURATION)
    
    last_valid_net_mask = None

    def render_buffered_frame(pop_data, pbar_ref):
        nonlocal current_highlight_frames

        f_idx = pop_data["f_idx"]
        if f_idx in highlight_events:
            current_highlight_frames = total_highlight_frames
            del highlight_events[f_idx]

        annotated_frame = annotate_frame(
            pop_data["augmented"], 
            pop_data["sam_masks"], 
            current_highlight_frames, 
            total_highlight_frames, 
            pop_data["net_mask"]
        )

        arr_idx = f_idx - 1
        center_x = smoothed_cx[arr_idx] if 0 <= arr_idx < len(smoothed_cx) else vid_w / 2.0
        
        # Recadrage vertical
        x1 = int(center_x - target_w / 2.0)
        x2 = x1 + target_w
        if x1 < 0: x1, x2 = 0, target_w
        elif x2 > vid_w: x2, x1 = vid_w, vid_w - target_w

        cropped_frame = np.ascontiguousarray(annotated_frame[:, x1:x2])
        writer.write(cropped_frame)
        pbar_ref.update(1)

        if current_highlight_frames > 0: 
            current_highlight_frames -= 1

    with tqdm(total=total, unit="frame", desc="Pass 2 (Pipeline & Render)") as pbar:
        while True:
            ret, raw_frame = cap.read()
            if not ret: break
            frame_count += 1

            frame = calibrator.undistort_frame(raw_frame) if calibrator.is_calibrated else raw_frame.copy()
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            player_dets, ball_det, hoop_dets = run_rfdetr_detection(det_model, frame, conf_player, conf_ball, CONF_HOOP)
            court_kp_xy, court_kp_conf = run_court_pose(court_pose_model, frame, device)
            if not calibrator.is_calibrated and court_kp_xy is not None: calibrator.add_points(court_kp_xy, court_kp_conf)

            ball_pos = ((ball_det[0]+ball_det[2])/2.0, (ball_det[1]+ball_det[3])/2.0) if ball_det is not None else None
            if ball_pos: ball_history.append((frame_count, ball_pos[0], ball_pos[1]))

            raw_tracks = run_tracking_with_gap_filling(player_tracker, frame, player_dets)
            clean_tracks = position_guard.filter(raw_tracks, frame_count)
            tracks = cap_to_max_players(clean_tracks, MAX_PLAYERS_ON_COURT)
            sam_masks = run_sam2_segmentation(sam_predictor, tracks)

            current_hoop_box = hoop_dets[0][:4] if hoop_dets else None
            net_ok_now = False
            current_net_mask = None

            if current_hoop_box is not None:
                current_net_mask = run_sam2_net_segmentation(sam_predictor, current_hoop_box, vid_w, vid_h, margin=SAM_NET_MARGIN)
                last_valid_net_mask = current_net_mask
                
                current_net_area = np.sum(current_net_mask)
                if len(net_area_history) > 0 and current_net_area > 50 and sum(net_area_history)/len(net_area_history) > 50:
                    if abs(current_net_area - sum(net_area_history)/len(net_area_history)) / (sum(net_area_history)/len(net_area_history)) >= NET_AREA_VAR_THRESH:
                        hx1, hy1, hx2, hy2 = current_hoop_box
                        if any((frame_count - b[0]) < fps and (hx1 - (hx2-hx1)*1.5) <= b[1] <= (hx2 + (hx2-hx1)*1.5) and (hy1 - (hy2-hy1)*2.0) <= b[2] <= (hy2 + (hy2-hy1)*3.0) for b in list(ball_history)[-10:]): net_ok_now = True
                net_area_history.append(current_net_area)
            else:
                current_net_mask = last_valid_net_mask
                net_area_history.clear()

            geom_ok_now = False
            if current_hoop_box is not None and len(ball_history) >= 3:
                hx1, hy1, hx2, hy2 = current_hoop_box
                b_frame, bx, by = ball_history[-1] 
                if by > hy1:
                    for p_frame, px, py in reversed(list(ball_history)[:-1]):
                        if py < hy1:
                            if (b_frame - p_frame) < int(fps * 0.8) and intersect((px, py), (bx, by), (hx1 - (hx2-hx1) * 0.1, hy1), (hx2 + (hx2-hx1) * 0.1, hy1)): geom_ok_now = True
                            break 

            if net_ok_now:  net_flag_frames = int(fps * 1.5)
            if geom_ok_now: geom_flag_frames = int(fps * 1.5)

            H_new = compute_homography(court_kp_xy, court_kp_conf)
            if H_new is not None:
                H_new = H_new / H_new[2,2]
                H_smooth_prev = H_smooth
                H_smooth = H_new if H_smooth is None else (HOMOGRAPHY_SMOOTHING*H_new + (1-HOMOGRAPHY_SMOOTHING)*H_smooth)
                H_smooth = H_smooth / H_smooth[2,2]
            
            motion = _measure_camera_motion(H_smooth_prev, H_smooth)
            if motion > MOTION_PIXEL_THRESHOLD: motion_cooldown = MOTION_COOLDOWN_FRAMES
            elif motion_cooldown > 0: motion_cooldown -= 1
            logos_visible = (motion_cooldown == 0)

            geom_ok, net_ok  = geom_flag_frames > 0, net_flag_frames > 0

            if geom_ok and net_ok and current_hoop_box and H_smooth is not None and (frame_count - last_shot_frame) >= int(fps * SHOT_COOLDOWN_SECONDS):
                try:
                    highlight_events[max(1, frame_count - delay_frames)] = True
                    last_shot_frame = frame_count
                    geom_flag_frames = 0 
                    net_flag_frames = 0
                    ball_history.clear()      
                    net_area_history.clear()  
                except: pass

            if geom_flag_frames > 0: geom_flag_frames -= 1
            if net_flag_frames > 0:  net_flag_frames -= 1

            frame_data = {
                "f_idx":      frame_count,
                "augmented":  apply_virtual_logo(frame, logos_data if logos_visible else [], H_smooth, sam_masks),
                "sam_masks":  sam_masks,
                "net_mask":   current_net_mask.copy() if current_net_mask is not None else None,
                "H_smooth":   H_smooth.copy() if H_smooth is not None else None,
            }
            frame_queue.append(frame_data)
            if len(frame_queue) > delay_frames: render_buffered_frame(frame_queue.popleft(), pbar)

        while frame_queue: render_buffered_frame(frame_queue.popleft(), pbar)

    cap.release()
    writer.release()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 🎧 FUSION AUDIO AVEC FFMPEG
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[INFO] Ajout de la piste audio originale (Fusion FFmpeg)...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_output_path),
            "-i", str(video_path),
            "-c:v", "copy",       
            "-c:a", "aac",        
            "-map", "0:v:0",      
            "-map", "1:a:0?",     
            "-shortest",          
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if temp_output_path.exists(): temp_output_path.unlink()
        print(f"[OK] 🎥 Vidéo finale avec audio sauvegardée avec succès : {output_path}")
        
    except FileNotFoundError:
        print(f"[ATTENTION] FFmpeg n'est pas installé. Vidéo muette : {output_path}")
        shutil.move(str(temp_output_path), str(output_path))
    except Exception as e:
        print(f"[ERREUR] Échec de la fusion audio : {e}. Vidéo muette : {output_path}")
        shutil.move(str(temp_output_path), str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=Path,  default=SOURCE_VIDEO_PATH)
    parser.add_argument("--output",      type=Path,  default=OUTPUT_PATH)
    parser.add_argument("--conf-player", type=float, default=CONF_PLAYER)
    parser.add_argument("--conf-ball",   type=float, default=CONF_BALL)
    parser.add_argument("--conf-kp",     type=float, default=CONF_KP)
    parser.add_argument("--device",      type=int,   default=0)
    args = parser.parse_args()
    out_path = args.output if args.output else args.video.with_name(args.video.stem + "_vertical_feu_filet.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    process_video(args.video, out_path, args.conf_player, args.conf_ball, args.conf_kp, args.device)