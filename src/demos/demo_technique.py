"""
basketball_demo.py  –  v19.1 (UI Épurée Cyan, Détection Filet SAM & Anti-Faux Positifs Temporel)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import scipy.optimize as opt
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
SMOOTHING_2D         = 0.20
MAX_LOST_FRAMES      = 6
DASHBOARD_PERSISTENCE_FRAMES = 15

# Paramètres Filet (SAM)
SAM_NET_MARGIN       = 0.03      # +3% de marge autour de la box du panier pour SAM
NET_AREA_VAR_THRESH  = 0.15      # 15% de variation de l'aire du filet pour valider

# 🌟 Paramètres Lignes, Onde de Choc & Logique de Tir 🌟
ZONE_HIGHLIGHT_DURATION  = 2.0
TIME_TRAVEL_SECONDS      = 0.5
ZONE_HIGHLIGHT_THICKNESS = 4
SHOT_COOLDOWN_SECONDS    = 0.75   # 🕒 Cooldown entre deux tirs réussis

# 👥 Cap joueurs & Anti-switch
MAX_PLAYERS_ON_COURT   = 10      # Maximum de joueurs affichés simultanément
MAX_SPEED_PX_PER_FRAME = 120     # Distance pixel max plausible entre deux frames consécutives
SWITCH_COOLDOWN_FRAMES = 10      # Frames pendant lesquelles un ID suspect est ignoré

CHECKPOINT_PLAYER     = Path("models/runs/object_detection/rfdetr-medium_1280px_100ep_v1/checkpoint_best_ema.pth")
CHECKPOINT_COURT      = Path("models/runs/keypoint_detection/yolo11m-pose_1000ep_v1/weights/best.pt")
CHECKPOINT_SAM        = Path("models/weights/sam2.1_hiera_small.pt")
CONFIG_SAM            = "configs/sam2.1/sam2.1_hiera_s.yaml"
SOURCE_VIDEO_PATH     = Path("data/demos/videos_raw/video_cergy_3pts.mp4")
OUTPUT_PATH           = Path("data/demos/videos_annotated/demo_cergy_3pts_technique.mp4")
REID_WEIGHTS          = Path("models/weights/osnet_x0_25_msmt17.pt")
LOGO_PATH             = Path("data/demos/assets/cenergy.png.png")

# Configuration du Logo
center_x, center_y, size = 2.285, 12.0, 4.0
half = size / 2
LOGO_1_CORNERS = [[center_x - half, center_y - half], [center_x + half, center_y - half],
                  [center_x + half, center_y + half], [center_x - half, center_y + half]]
COURT_L, COURT_W = 28.0, 15.0
center_x_mirror, center_y_mirror = COURT_L - center_x, COURT_W - center_y
LOGO_2_CORNERS = [[center_x_mirror - half, center_y_mirror - half], [center_x_mirror + half, center_y_mirror - half],
                  [center_x_mirror + half, center_y_mirror + half], [center_x_mirror - half, center_y_mirror + half]]
ALL_LOGOS_CORNERS = [np.array(LOGO_1_CORNERS, dtype=np.float32), np.array(LOGO_2_CORNERS, dtype=np.float32)]

# ─────────────────────────────────────────────────────────────────────────────
# TERRAIN FIBA (mètres) & MAPPING
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
_idx_19, _idx_21 = YOLO_TO_CUSTOM_ID.index(19), YOLO_TO_CUSTOM_ID.index(21)
YOLO_TO_CUSTOM_ID[_idx_19], YOLO_TO_CUSTOM_ID[_idx_21] = 21, 19

def _build_world_coords() -> dict:
    custom_id_to_xy = {k: v for k, v in FIBA_M_COORDS.items()}
    for left_id, right_id in SYMMETRY_PAIRS_1BASED:
        if left_id in FIBA_M_COORDS:
            custom_id_to_xy[right_id] = (COURT_L - FIBA_M_COORDS[left_id][0], FIBA_M_COORDS[left_id][1])
    return {yolo_idx: custom_id_to_xy[custom_id]
            for yolo_idx, custom_id in enumerate(YOLO_TO_CUSTOM_ID) if custom_id in custom_id_to_xy}

WORLD_COORDS = _build_world_coords()

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION COUPLÉE & TPS
# ─────────────────────────────────────────────────────────────────────────────
class CoupledCalibrator:
    def __init__(self, img_w, img_h):
        self.img_w, self.img_h = img_w, img_h
        self.cx, self.cy = img_w / 2.0, img_h / 2.0
        self.f_init = img_w * 0.8
        self.src_pts, self.dst_pts = [], []
        self.is_calibrated = False
        self.k1, self.k2, self.f_found = 0.0, 0.0, img_w * 0.8
        self.map1, self.map2 = None, None

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
        src_arr = np.array(self.src_pts, dtype=np.float32)
        dst_arr = np.array(self.dst_pts, dtype=np.float32)
        H0, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        H0 = H0 / H0[2, 2] if H0 is not None else np.eye(3)
        p0 = [H0[0,0], H0[0,1], H0[0,2], H0[1,0], H0[1,1], H0[1,2],
              H0[2,0], H0[2,1], 0.0, 0.0, self.f_init]

        def cost_fn(params):
            H = np.array([[params[0], params[1], params[2]],
                          [params[3], params[4], params[5]],
                          [params[6], params[7], 1.0]])
            k1, k2, f = params[8], params[9], params[10]
            K = np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1]], dtype=np.float32)
            D = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
            pts_reshaped = src_arr.reshape(-1, 1, 2)
            undistorted  = cv2.undistortPoints(pts_reshaped, K, D)
            u_un = undistorted[:, 0, 0] * f + self.cx
            v_un = undistorted[:, 0, 1] * f + self.cy
            pts_hom      = np.stack([u_un, v_un, np.ones_like(u_un)], axis=1)
            transformed = pts_hom @ H.T
            X_pred = transformed[:, 0] / (transformed[:, 2] + 1e-7)
            Y_pred = transformed[:, 1] / (transformed[:, 2] + 1e-7)
            return np.concatenate([X_pred - dst_arr[:, 0], Y_pred - dst_arr[:, 1]])

        res = opt.least_squares(cost_fn, p0, method='lm')
        self.k1, self.k2, self.f_found = res.x[8], res.x[9], res.x[10]
        K = np.array([[self.f_found, 0, self.cx], [0, self.f_found, self.cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([self.k1, self.k2, 0, 0, 0], dtype=np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (self.img_w, self.img_h), 1, (self.img_w, self.img_h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (self.img_w, self.img_h), cv2.CV_32FC1)
        self.is_calibrated = True

    def undistort_frame(self, frame):
        if not self.is_calibrated: return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


class TPSWarper:
    def __init__(self):
        self.is_fitted, self._img_pts, self._world_pts = False, None, None

    def fit(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> bool:
        img_pts, world_pts = [], []
        for yolo_idx, world_xy in WORLD_COORDS.items():
            if yolo_idx >= len(kp_conf) or kp_conf[yolo_idx] < CONF_KP: continue
            px, py = kp_xy[yolo_idx]
            if px == 0.0 and py == 0.0: continue
            img_pts.append([float(px), float(py)])
            world_pts.append(list(world_xy))
        if len(img_pts) < 5: return False
        self._img_pts  = np.array(img_pts,   dtype=np.float32)
        self._world_pts = np.array(world_pts, dtype=np.float32)
        self.is_fitted = True
        return True

    def _make_tps(self, src, dst):
        tps = cv2.createThinPlateSplineShapeTransformer(regularizationParameter=0)
        tps.estimateTransformation(
            src.reshape(1, -1, 2), dst.reshape(1, -1, 2),
            [cv2.DMatch(i, i, 0) for i in range(len(src))])
        return tps

    def image_to_world(self, pts: np.ndarray) -> np.ndarray | None:
        if not self.is_fitted: return None
        _, result = self._make_tps(self._img_pts, self._world_pts).applyTransformation(
            np.array(pts, dtype=np.float32).reshape(1, -1, 2))
        return result.reshape(-1, 2)

    def project_foot(self, px: float, py: float) -> tuple | None:
        res = self.image_to_world(np.array([[px, py]]))
        if res is None: return None
        xm, ym = float(res[0, 0]), float(res[0, 1])
        return (xm, ym) if (-2 < xm < COURT_L + 2 and -2 < ym < COURT_W + 2) else None


# ─────────────────────────────────────────────────────────────────────────────
# SEUILS VISUELS & RÉGLAGES DE BASE
# ─────────────────────────────────────────────────────────────────────────────
CONF_PLAYER, CONF_BALL, CONF_HOOP, CONF_KP = 0.40, 0.40, 0.30, 0.50
INFERENCE_RESOLUTION, HOMOGRAPHY_SMOOTHING = 1280, 0.15
SAM_MASK_ALPHA = 0.35  

C_BG, C_COURT, C_LINE, C_SEP = (15, 18, 22), (28, 32, 38), (180, 180, 190), (15, 18, 22)
C_HEADER_BG, C_HEADER_TXT = (10, 12, 16), (240, 240, 245)
C_KP_HIGH, C_KP_MED, C_KP_LOW, C_KP_GREY = (0, 230, 80), (0, 200, 255), (0, 80, 255), (150, 150, 150)
C_BALL_REAL, C_HOOP, C_OK = (30, 200, 255), (80, 130, 210), (50, 255, 50)
C_PLAYER_CYAN = (255, 255, 0) 

FONT, FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-ID-SWITCH : gestionnaire de positions
# ─────────────────────────────────────────────────────────────────────────────
class TrackPositionGuard:
    def __init__(self, max_speed: float = MAX_SPEED_PX_PER_FRAME, cooldown: int = SWITCH_COOLDOWN_FRAMES):
        self.max_speed   = max_speed
        self.cooldown    = cooldown
        self._last_pos   : dict[int, tuple[float, float]] = {}
        self._banned_until: dict[int, int] = {}

    def _center(self, t) -> tuple[float, float]:
        return ((t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0)

    def filter(self, tracks: np.ndarray, frame_idx: int) -> np.ndarray:
        if len(tracks) == 0: return tracks

        keep = []
        for t in tracks:
            tid = int(t[4])
            if tid < 0:
                keep.append(True)
                continue

            if frame_idx <= self._banned_until.get(tid, -1):
                keep.append(False)
                continue

            cx, cy = self._center(t)

            if tid in self._last_pos:
                lx, ly = self._last_pos[tid]
                dist = np.hypot(cx - lx, cy - ly)
                if dist > self.max_speed:
                    self._banned_until[tid] = frame_idx + self.cooldown
                    keep.append(False)
                    continue

            keep.append(True)

        kept_tracks = tracks[np.array(keep, dtype=bool)]

        for t in kept_tracks:
            tid = int(t[4])
            if tid >= 0:
                self._last_pos[tid] = self._center(t)

        active_ids = {int(t[4]) for t in kept_tracks if t[4] >= 0}
        stale = [tid for tid in list(self._last_pos.keys())
                 if tid not in active_ids and frame_idx > self._banned_until.get(tid, -1) + 30]
        for tid in stale:
            self._last_pos.pop(tid, None)
            self._banned_until.pop(tid, None)

        return kept_tracks

# ─────────────────────────────────────────────────────────────────────────────
# CAP 10 JOUEURS
# ─────────────────────────────────────────────────────────────────────────────
def cap_to_max_players(tracks: np.ndarray, max_players: int = MAX_PLAYERS_ON_COURT) -> np.ndarray:
    if len(tracks) <= max_players: return tracks
    active    = tracks[tracks[:, 5] > 0]
    gap_filled = tracks[tracks[:, 5] == 0]

    if len(active) > 0: active = active[np.argsort(-active[:, 5])]
    if len(gap_filled) > 0:
        areas = (gap_filled[:, 2] - gap_filled[:, 0]) * (gap_filled[:, 3] - gap_filled[:, 1])
        gap_filled = gap_filled[np.argsort(-areas)]

    combined = np.concatenate([active, gap_filled], axis=0) if len(active) > 0 else gap_filled
    return combined[:max_players]

# ─────────────────────────────────────────────────────────────────────────────
# GÉOMÉTRIE & UTILS
# ─────────────────────────────────────────────────────────────────────────────
def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A, B, C, D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# ─────────────────────────────────────────────────────────────────────────────
# HQ TERRAIN
# ─────────────────────────────────────────────────────────────────────────────
def _m2px(xm, ym, sc, mg): return int(ym * sc + mg), int(xm * sc + mg)
def _arc_pts(cx, cy, r, a0, a1, n=72):
    return np.stack([cx + r * np.cos(np.radians(np.linspace(a0, a1, n))),
                     cy + r * np.sin(np.radians(np.linspace(a0, a1, n)))], axis=1)
def _pts2px(pts, sc, mg):
    return np.stack([(pts[:,1]*sc+mg).astype(np.int32), (pts[:,0]*sc+mg).astype(np.int32)], axis=1)
def _list2px(pts, sc, mg): return _pts2px(np.array(pts, np.float32), sc, mg)

def build_court_bg_hq(target_h: int) -> tuple:
    sc, mg = (target_h - 80) / COURT_L, 40
    img = np.full((target_h, int(COURT_W * sc + 2 * mg), 3), C_BG, dtype=np.uint8)
    p0, p1 = _m2px(0, 0, sc, mg), _m2px(COURT_L, COURT_W, sc, mg)
    cv2.rectangle(img, (min(p0[0],p1[0]), min(p0[1],p1[1])),
                  (max(p0[0],p1[0]), max(p0[1],p1[1])), C_COURT, -1)
    lw = max(1, round(sc * 0.035))

    def pline(pts, closed=False):
        cv2.polylines(img, [_list2px(pts, sc, mg).reshape(-1,1,2)], closed, C_LINE, lw, cv2.LINE_AA)
    def arc(cx, cy, r, a0, a1, n=128):
        cv2.polylines(img, [_pts2px(_arc_pts(cx, cy, r, a0, a1, n), sc, mg).reshape(-1,1,2)],
                      False, C_LINE, lw, cv2.LINE_AA)

    _a3 = float(np.degrees(np.arcsin((Y_CENTER - Y_3PT_OFFSET) / R_3PT)))
    for sign in (1, -1):
        bx, fx = (0.0, X_FT) if sign == 1 else (COURT_L, COURT_L - X_FT)
        pline([(bx, Y_CENTER-Y_KEY_HALF),(fx, Y_CENTER-Y_KEY_HALF),
               (fx, Y_CENTER+Y_KEY_HALF),(bx, Y_CENTER+Y_KEY_HALF)], True)
        arc(fx, Y_CENTER, 1.8, 0, 360, 256)
        pline([(bx, Y_3PT_OFFSET), (bx + sign*X_3PT_START, Y_3PT_OFFSET)])
        pline([(bx, COURT_W-Y_3PT_OFFSET), (bx + sign*X_3PT_START, COURT_W-Y_3PT_OFFSET)])
        if sign == 1:
            arc(X_BASKET, Y_CENTER, R_3PT, -_a3, _a3, 256)
            arc(X_BASKET, Y_CENTER, 1.25, -90, 90, 128)
            cv2.circle(img, _m2px(X_BASKET, Y_CENTER, sc, mg), max(2, round(0.23*sc)), C_LINE, lw, cv2.LINE_AA)
        else:
            arc(COURT_L-X_BASKET, Y_CENTER, R_3PT, 180-_a3, 180+_a3, 256)
            arc(COURT_L-X_BASKET, Y_CENTER, 1.25, 90, 270, 128)
            cv2.circle(img, _m2px(COURT_L-X_BASKET, Y_CENTER, sc, mg), max(2, round(0.23*sc)), C_LINE, lw, cv2.LINE_AA)
    pline([(14.0, 0.0), (14.0, COURT_W)])
    arc(14.0, Y_CENTER, 1.8, 0, 360, 256)
    pline([(0,0),(COURT_L,0),(COURT_L,COURT_W),(0,COURT_W)], True)
    return img, sc, mg

def draw_player_dot_hq(img, cx, cy, radius):
    cv2.circle(img, (cx+1, cy+1), radius, (20,20,20), -1, cv2.LINE_AA) 
    cv2.circle(img, (cx,   cy),   radius, C_PLAYER_CYAN, -1, cv2.LINE_AA) 
    cv2.circle(img, (cx,   cy),   radius, (255,255,255),  1, cv2.LINE_AA) 

HEADER_H = 40
def draw_header(img, text):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, HEADER_H), C_HEADER_BG, -1)
    cv2.line(img, (0, HEADER_H), (w, HEADER_H), (40, 45, 55), 1)
    (tw, th), _ = cv2.getTextSize(text, FONT_MONO, 0.44, 1)
    cv2.putText(img, text, ((w-tw)//2, (HEADER_H+th)//2), FONT_MONO, 0.44, C_HEADER_TXT, 1, cv2.LINE_AA)

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
    except Exception:
        return [], None, []

def run_court_pose(model, frame, device):
    res = model(frame, device=device, verbose=False)
    if not res or res[0].keypoints is None or res[0].keypoints.xy.shape[0] == 0:
        return None, None
    return res[0].keypoints.xy[0].cpu().numpy(), res[0].keypoints.conf[0].cpu().numpy()

def run_tracking_with_gap_filling(tracker: BotSort, frame, player_dets) -> np.ndarray:
    active_tracks = tracker.update(
        np.array(player_dets, dtype=np.float32)[:,:6] if player_dets else np.empty((0,6), dtype=np.float32), frame)
    aug = list(active_tracks) if active_tracks is not None else []
    active_ids = {int(t[4]) for t in aug} if aug else set()

    if hasattr(tracker, 'lost_stracks'):
        current_frame = getattr(tracker, 'frame_id', 0)
        img_h, img_w  = frame.shape[0], frame.shape[1]
        for t in tracker.lost_stracks:
            tid = int(t.track_id)
            if tid not in active_ids and 0 < (current_frame - getattr(t,'frame_id',current_frame)) <= MAX_LOST_FRAMES:
                x1,y1,x2,y2 = getattr(t,'tlbr',(0,0,0,0))
                if ((np.clip(x2,0,img_w)-np.clip(x1,0,img_w)) > 10 and (np.clip(y2,0,img_h)-np.clip(y1,0,img_h)) > 10):
                    aug.append([np.clip(x1,0,img_w), np.clip(y1,0,img_h), np.clip(x2,0,img_w), np.clip(y2,0,img_h), tid, 0.0, 0, -1])
    return np.array(aug, dtype=np.float32) if aug else np.empty((0, 8))

def run_sam2_segmentation(sam_predictor, tracks):
    """ Exécute SAM sur les joueurs. L'image doit avoir été set_image() au préalable. """
    if len(tracks) == 0: return []
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        masks, _, _ = sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array([t[:4] for t in tracks]),
            multimask_output=False)
    return [mask.squeeze().astype(bool) for mask in masks]

def run_sam2_net_segmentation(sam_predictor, hoop_box, img_w, img_h, margin=0.03):
    """ Exécute SAM pour cibler et détourez le filet sous le panier. """
    x1, y1, x2, y2 = hoop_box
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, x1 - w * margin)
    ny1 = max(0, y1 - h * margin)
    nx2 = min(img_w, x2 + w * margin)
    ny2 = min(img_h, y2 + h * margin)
    
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        masks, _, _ = sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array([[nx1, ny1, nx2, ny2]]),
            multimask_output=False)
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

def apply_virtual_logo(frame, logo_img, H_frame_to_world, all_masks):
    if H_frame_to_world is None or logo_img is None: return frame
    h_logo, w_logo = logo_img.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    try: H_world_to_frame = np.linalg.inv(H_frame_to_world)
    except: return frame
    out_frame = frame.copy()
    for target_world_corners in ALL_LOGOS_CORNERS:
        H_logo_to_world, _ = cv2.findHomography(np.array([[0,0],[w_logo,0],[w_logo,h_logo],[0,h_logo]], dtype=np.float32), target_world_corners)
        warped_logo  = cv2.warpPerspective(logo_img, H_world_to_frame @ H_logo_to_world, (w_frame, h_frame), flags=cv2.INTER_LINEAR)
        warped_rgb   = warped_logo[...,:3]
        warped_alpha = warped_logo[...,3].astype(np.float32) / 255.0
        for mask in all_masks: warped_alpha[mask] = 0.0
        for c in range(3): out_frame[...,c] = out_frame[...,c]*(1.0-warped_alpha) + warped_rgb[...,c]*warped_alpha
    return out_frame.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION & ONDE DE CHOC GOUTTE D'EAU
# ─────────────────────────────────────────────────────────────────────────────
def annotate_frame(frame, tracks, sam_masks, net_mask, ball_pos,
                   court_kp_xy, court_kp_conf,
                   reused_homography, hoop_dets,
                   geom_ok, net_ok,
                   H_matrix, tps_fitted,
                   highlight_frames, total_highlight_frames,
                   active_basket, lens_status) -> np.ndarray:

    out     = frame.copy()
    overlay = np.zeros_like(out)
    h_img, w_img = out.shape[:2]

    # ── 🌟 Onde de Choc Pure ───────────────────────────────────
    if highlight_frames > 0 and active_basket is not None and H_matrix is not None:
        try:
            is_left = (active_basket == "left")
            try: H_inv = np.linalg.inv(H_matrix)
            except np.linalg.LinAlgError: H_inv = np.eye(3)

            court_components = []
            if is_left:
                court_components.append((np.array([[0.0,0.0],[0.0,15.0]]), False))
                court_components.append((np.array([[0.0,Y_CENTER-Y_KEY_HALF],[X_FT,Y_CENTER-Y_KEY_HALF],
                                                   [X_FT,Y_CENTER+Y_KEY_HALF],[0.0,Y_CENTER+Y_KEY_HALF]]), True))
                pts_3pt = [[0.0, Y_3PT_OFFSET], [X_3PT_START, Y_3PT_OFFSET]]
                for a in np.linspace(np.arctan2(Y_3PT_OFFSET - Y_CENTER, X_3PT_START - X_BASKET),
                                     np.arctan2((15.0-Y_3PT_OFFSET) - Y_CENTER, X_3PT_START - X_BASKET), 60):
                    pts_3pt.append([X_BASKET + R_3PT*np.cos(a), Y_CENTER + R_3PT*np.sin(a)])
                pts_3pt.extend([[X_3PT_START, 15.0-Y_3PT_OFFSET],[0.0, 15.0-Y_3PT_OFFSET]])
                court_components.append((np.array(pts_3pt), False))
                pts_ft = [[X_FT + R_FT*np.cos(a), Y_CENTER + R_FT*np.sin(a)] for a in np.linspace(-np.pi/2, np.pi/2, 30)]
                court_components.append((np.array(pts_ft), False))
            else:
                court_components.append((np.array([[COURT_L,0.0],[COURT_L,15.0]]), False))
                court_components.append((np.array([[COURT_L,Y_CENTER-Y_KEY_HALF],[COURT_L-X_FT,Y_CENTER-Y_KEY_HALF],
                                                   [COURT_L-X_FT,Y_CENTER+Y_KEY_HALF],[COURT_L,Y_CENTER+Y_KEY_HALF]]), True))
                pts_3pt = [[COURT_L, Y_3PT_OFFSET],[COURT_L-X_3PT_START, Y_3PT_OFFSET]]
                a_bot = np.arctan2(Y_3PT_OFFSET-Y_CENTER, (COURT_L-X_3PT_START)-(COURT_L-X_BASKET))
                a_top = np.arctan2((15.0-Y_3PT_OFFSET)-Y_CENTER, (COURT_L-X_3PT_START)-(COURT_L-X_BASKET))
                if a_bot < 0: a_bot += 2*np.pi
                for a in np.linspace(a_bot, a_top, 60):
                    pts_3pt.append([(COURT_L-X_BASKET)+R_3PT*np.cos(a), Y_CENTER+R_3PT*np.sin(a)])
                pts_3pt.extend([[COURT_L-X_3PT_START, 15.0-Y_3PT_OFFSET],[COURT_L, 15.0-Y_3PT_OFFSET]])
                court_components.append((np.array(pts_3pt), False))
                pts_ft = [[(COURT_L-X_FT)+R_FT*np.cos(a), Y_CENTER+R_FT*np.sin(a)] for a in np.linspace(np.pi/2, 3*np.pi/2, 30)]
                court_components.append((np.array(pts_ft), False))

            lines_layer = np.zeros_like(out)
            COLOR_OUTER, COLOR_MID, COLOR_INNER = (50,255,50), (150,255,150), (255,255,255)

            for arr_3d, is_closed in court_components:
                px_arr = np.int32(cv2.perspectiveTransform(arr_3d.astype(np.float32).reshape(-1,1,2), H_inv).reshape(-1,1,2))
                cv2.polylines(lines_layer,[px_arr],isClosed=is_closed, color=COLOR_OUTER,thickness=ZONE_HIGHLIGHT_THICKNESS+5,lineType=cv2.LINE_AA)
                cv2.polylines(lines_layer,[px_arr],isClosed=is_closed, color=COLOR_MID,  thickness=ZONE_HIGHLIGHT_THICKNESS+2,lineType=cv2.LINE_AA)
                cv2.polylines(lines_layer,[px_arr],isClosed=is_closed, color=COLOR_INNER,thickness=ZONE_HIGHLIGHT_THICKNESS,  lineType=cv2.LINE_AA)

            basket_3d = np.array([[[X_BASKET if is_left else (COURT_L-X_BASKET), Y_CENTER]]], dtype=np.float32)
            try:
                b_px = cv2.perspectiveTransform(basket_3d, H_inv)
                basket_px = (int(b_px[0,0,0]), int(b_px[0,0,1])) if not np.isnan(b_px).any() else (w_img//2, h_img//2)
            except:
                basket_px = (w_img//2, h_img//2)

            t_ratio = 1.0 - (highlight_frames / float(total_highlight_frames))
            wave_mask = np.zeros((h_img, w_img), dtype=np.float32)
            max_radius = int(w_img * 1.2)

            if t_ratio <= 1.0:
                r = int(max_radius * t_ratio)
                thickness_front = max(8,  int(w_img * 0.02))
                thickness_trail = max(25, int(w_img * 0.06))
                if r > thickness_trail:
                    cv2.circle(wave_mask, basket_px, r-thickness_trail//2, 0.6, thickness_trail, cv2.LINE_AA)
                cv2.circle(wave_mask, basket_px, r, 4.0, thickness_front, cv2.LINE_AA)
                k_size = (int(w_img*0.03)//2)*2+1
                wave_mask = cv2.GaussianBlur(wave_mask, (k_size, k_size), 0)

            lines_float = lines_layer.astype(np.float32)
            for c in range(3): lines_float[...,c] *= wave_mask
            lines_layer = np.clip(lines_float, 0, 255).astype(np.uint8)

            combined_sam_mask = np.zeros((h_img, w_img), dtype=bool)
            for mask in sam_masks:
                if mask.shape == combined_sam_mask.shape:
                    combined_sam_mask |= mask
            lines_layer[combined_sam_mask] = 0

            out = cv2.addWeighted(out, 1.0, lines_layer, 1.0, 0)
        except Exception:
            pass

    # ── Masques SAM (Joueurs et Filet) ─────────────────────────────────────────
    for idx, t in enumerate(tracks):
        if idx < len(sam_masks):
            overlay[sam_masks[idx]] = C_PLAYER_CYAN
    
    if net_mask is not None:
        overlay[net_mask] = C_HOOP # Mise en évidence subtile du filet

    cv2.addWeighted(overlay, SAM_MASK_ALPHA, out, 1.0, 0, out)

    for hoop in hoop_dets:
        cv2.rectangle(out,(int(hoop[0]),int(hoop[1])),(int(hoop[2]),int(hoop[3])),C_HOOP,1,cv2.LINE_AA)

    if ball_pos is not None:
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(out, (cx,cy), 11, (*C_BALL_REAL,80),  3,  cv2.LINE_AA)
        cv2.circle(out, (cx,cy),  7, C_BALL_REAL,       -1,  cv2.LINE_AA)
        cv2.circle(out, (cx,cy),  7, (255,255,255),      2,  cv2.LINE_AA)

    if court_kp_xy is not None:
        for i, (px, py) in enumerate(court_kp_xy):
            if i >= len(court_kp_conf) or court_kp_conf[i] < CONF_KP or (px==0 and py==0): continue
            color = C_KP_GREY if reused_homography else (
                C_KP_HIGH if court_kp_conf[i] > 0.70 else
                C_KP_MED  if court_kp_conf[i] > 0.55 else C_KP_LOW)
            cv2.circle(out,(int(px),int(py)),6,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(out,(int(px),int(py)),5,color,  -1,cv2.LINE_AA)

    # Header
    cv2.rectangle(out,(0,0),(w_img,HEADER_H),C_HEADER_BG,-1)
    cv2.line(out,(0,HEADER_H),(w_img,HEADER_H),(40,45,55),1)
    tps_str = "TPS:OK" if tps_fitted else "TPS:--"
    hom_str = "H:RECYCLE" if reused_homography else ("H:OK" if H_matrix is not None else "H:--")
    geom_text = "GEOM:[OK]" if geom_ok else "GEOM:[--]"
    net_text  = "FILET:[OK]" if net_ok else "FILET:[--]"
    n_kp = int((court_kp_conf >= CONF_KP).sum()) if court_kp_conf is not None else 0

    def put(text, color, cursor):
        sz = cv2.getTextSize(text, FONT_MONO, 0.44, 1)[0]
        cv2.putText(out, text, (cursor, (HEADER_H+sz[1])//2), FONT_MONO, 0.44, color, 1, cv2.LINE_AA)
        return cursor + sz[0]

    cur = (w_img - cv2.getTextSize(
        f"LENS:{lens_status} | {tps_str} | {hom_str} | Joueurs:{len(tracks)}/{MAX_PLAYERS_ON_COURT} | KP:{n_kp}/31 | Tir -> {geom_text} | {net_text}",
        FONT_MONO, 0.44, 1)[0][0]) // 2
    cur = put(f"LENS:{lens_status} | ", C_OK if lens_status=="CORRECTED" else (0,165,255), cur)
    cur = put(f"{tps_str} | ", C_OK if tps_fitted else C_HEADER_TXT, cur)
    cur = put(f"{hom_str} | Joueurs:{len(tracks)}/{MAX_PLAYERS_ON_COURT} | KP:{n_kp}/31 | Tir -> ", C_HEADER_TXT, cur)
    cur = put(f"{geom_text} | ", C_OK if geom_ok else C_HEADER_TXT, cur)
    cur = put(net_text,           C_OK if net_ok  else C_HEADER_TXT, cur)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(video_path: Path, output_path: Path, conf_player: float, conf_ball: float, conf_kp: float, device: int) -> None:
    global CONF_KP
    CONF_KP = conf_kp

    torch_device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True

    print("[INFO] Chargement des modèles...")
    from ultralytics import YOLO
    from rfdetr import RFDETRMedium
    court_pose_model = YOLO(str(CHECKPOINT_COURT))
    det_model        = RFDETRMedium(pretrain_weights=str(CHECKPOINT_PLAYER), resolution=INFERENCE_RESOLUTION)
    player_tracker   = BotSort(reid_weights=REID_WEIGHTS, device=device, half=False,
                               track_high_thresh=0.45, track_low_thresh=0.15,
                               new_track_thresh=0.55, track_buffer=60, match_thresh=0.80)
    sam_predictor    = SAM2ImagePredictor(build_sam2(CONFIG_SAM, str(CHECKPOINT_SAM), device=torch_device_str))

    logo_img = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED) if LOGO_PATH.exists() else None
    if logo_img is not None and logo_img.ndim == 3 and logo_img.shape[2] == 3: logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): sys.exit(f"[ERREUR] Impossible d'ouvrir : {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    court_bg_hq, sc_hq, mg_hq = build_court_bg_hq(vid_h)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w + 3 + court_bg_hq.shape[1], vid_h))
    sep = np.full((vid_h, 3, 3), C_SEP, dtype=np.uint8)

    H_smooth         = None
    tps_warper       = TPSWarper()
    calibrator       = CoupledCalibrator(vid_w, vid_h)
    position_guard   = TrackPositionGuard() 

    prev_hoop_box       = None
    prev_court_kp_xy    = None
    prev_court_kp_conf  = None

    persistent_2d_state = {}
    ball_history        = deque(maxlen=40) 
    net_area_history    = deque(maxlen=5) 
    
    frame_count         = 0
    geom_flag_frames    = 0
    net_flag_frames     = 0
    last_shot_frame     = -9999 

    # Buffer voyage temporel
    delay_frames           = int(fps * TIME_TRAVEL_SECONDS)
    frame_queue            = deque()
    highlight_events       = {}
    current_highlight_frames = 0
    total_highlight_frames   = int(fps * ZONE_HIGHLIGHT_DURATION)
    current_active_basket    = None

    print(f"[INFO] Traitement de {total} frames → {output_path}")
    print(f"       Anti-switch actif (max {MAX_SPEED_PX_PER_FRAME}px/frame) | Cap joueurs : {MAX_PLAYERS_ON_COURT}")

    def render_buffered_frame(pop_data, pbar_ref):
        nonlocal current_highlight_frames, current_active_basket

        f_idx = pop_data["f_idx"]
        if f_idx in highlight_events:
            current_highlight_frames = total_highlight_frames
            current_active_basket    = highlight_events[f_idx]
            del highlight_events[f_idx]

        left_panel = annotate_frame(
            pop_data["augmented"],
            pop_data["tracks"],
            pop_data["sam_masks"],
            pop_data["net_mask"],
            pop_data["ball_pos"],
            pop_data["court_kp_xy"],
            pop_data["court_kp_conf"],
            pop_data["reused_homography"],
            pop_data["hoop_dets"],
            pop_data["geom_ok"],
            pop_data["net_ok"],
            pop_data["H_smooth"],
            pop_data["tps_fitted"],
            current_highlight_frames,
            total_highlight_frames,
            current_active_basket,
            pop_data["lens_status"],
        )

        right_panel = court_bg_hq.copy()
        court_h_hq  = right_panel.shape[0] - 30
        dot_r_hq    = max(9, int(sc_hq * 0.6))
        for tid, state in pop_data["persistent_2d_state"].items():
            px_ = int(max(0.0, min(COURT_W, COURT_W - state["pos"][1])) * sc_hq + mg_hq)
            py_ = min(int(max(0.0, min(COURT_L, state["pos"][0])) * sc_hq + mg_hq), court_h_hq - dot_r_hq - 1)
            draw_player_dot_hq(right_panel, px_, py_, dot_r_hq)

        draw_header(right_panel, "DASHBOARD TACTIQUE")
        y0 = right_panel.shape[0] - 30
        cv2.rectangle(right_panel,(0,y0),(right_panel.shape[1],right_panel.shape[0]),C_HEADER_BG,-1)
        cv2.line(right_panel,(0,y0),(right_panel.shape[1],y0),(40,45,55),1)
        n_active = sum(1 for s in pop_data["persistent_2d_state"].values() if s["lost_frames"] == 0)
        cv2.putText(right_panel, f"Joueurs (Actifs: {n_active} | Total: {len(pop_data['persistent_2d_state'])})",
                    (15, y0+19), FONT_MONO, 0.38, C_HEADER_TXT, 1, cv2.LINE_AA)

        writer.write(np.hstack([left_panel, sep, right_panel]))
        pbar_ref.update(1)

        if current_highlight_frames > 0: current_highlight_frames -= 1

    # ── BOUCLE DE TRAITEMENT ─────────────────────────────────────────────────
    with tqdm(total=total, unit="frame", desc="Processing") as pbar:
        while True:
            ret, raw_frame = cap.read()
            if not ret: break
            frame_count += 1

            frame = calibrator.undistort_frame(raw_frame) if calibrator.is_calibrated else raw_frame.copy()
            lens_status = "CORRECTED" if calibrator.is_calibrated else "CALIBRATING..."
            frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 🔥 Optimisation SAM : On passe l'image dans l'encodeur UNE SEULE FOIS par frame
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                sam_predictor.set_image(frame_rgb)

            player_dets, ball_det, hoop_dets = run_rfdetr_detection(det_model, frame, conf_player, conf_ball, CONF_HOOP)

            court_kp_xy, court_kp_conf = run_court_pose(court_pose_model, frame, device)
            if not calibrator.is_calibrated and court_kp_xy is not None: calibrator.add_points(court_kp_xy, court_kp_conf)

            ball_pos = ((ball_det[0]+ball_det[2])/2.0, (ball_det[1]+ball_det[3])/2.0) if ball_det is not None else None
            
            # ── Enregistrement temporel de la balle ──────────────────────────
            if ball_pos: 
                ball_history.append((frame_count, ball_pos[0], ball_pos[1]))

            # ── Tracking + Cap + SAM Joueurs ─────────────────────────────────
            raw_tracks = run_tracking_with_gap_filling(player_tracker, frame, player_dets)
            clean_tracks = position_guard.filter(raw_tracks, frame_count)
            tracks = cap_to_max_players(clean_tracks, MAX_PLAYERS_ON_COURT)
            
            # L'image est déjà encodée, on récupère les masques rapidement
            sam_masks = run_sam2_segmentation(sam_predictor, tracks)

            # ── 1. Analyse SAM du Filet (Déformation) ────────────────────────
            current_hoop_box = hoop_dets[0][:4] if hoop_dets else None
            net_mask = None
            net_ok_now = False

            if current_hoop_box is not None:
                net_mask = run_sam2_net_segmentation(sam_predictor, current_hoop_box, vid_w, vid_h, margin=SAM_NET_MARGIN)
                current_net_area = np.sum(net_mask)
                
                if len(net_area_history) > 0 and current_net_area > 50:
                    avg_prev_area = sum(net_area_history) / len(net_area_history)
                    if avg_prev_area > 50:
                        variation = abs(current_net_area - avg_prev_area) / avg_prev_area
                        if variation >= NET_AREA_VAR_THRESH:
                            # Vérif que la balle était près du panier physiquement ET temporellement (max 1 sec)
                            hx1, hy1, hx2, hy2 = current_hoop_box
                            w_h, h_h = hx2 - hx1, hy2 - hy1
                            ball_is_near = any(
                                (frame_count - b_frame) < fps and 
                                (hx1 - w_h*1.5) <= bcx <= (hx2 + w_h*1.5) and
                                (hy1 - h_h*2.0) <= bcy <= (hy2 + h_h*3.0)
                                for b_frame, bcx, bcy in list(ball_history)[-10:]
                            )
                            if ball_is_near:
                                net_ok_now = True
                
                net_area_history.append(current_net_area)
            else:
                net_area_history.clear()

            # ── 2. Analyse géométrique (Balle passant dans l'arceau) ─────────
            geom_ok_now = False
            if current_hoop_box is not None and len(ball_history) >= 3:
                hx1, hy1, hx2, hy2 = current_hoop_box
                w_h = hx2 - hx1
                b_frame, bx, by = ball_history[-1] # Position actuelle la plus récente

                # Si la balle actuelle est EN DESSOUS du haut de l'arceau
                if by > hy1:
                    # Chercher le point le plus récent EN AU-DESSUS de l'arceau
                    for past_b in reversed(list(ball_history)[:-1]):
                        p_frame, px, py = past_b
                        if py < hy1:
                            # 🛡️ ANTI-TELEPORTATION : Ne relier les points que si l'écart de temps est court (ex: max 0.8 seconde)
                            if (b_frame - p_frame) < int(fps * 0.8):
                                # Vérifier que le segment passe STRICTEMENT au milieu de l'arceau (+/- petite marge)
                                A = (px, py)
                                B = (bx, by)
                                C = (hx1 - w_h * 0.1, hy1) # Bord gauche du panier
                                D = (hx2 + w_h * 0.1, hy1) # Bord droit du panier
                                
                                if intersect(A, B, C, D):
                                    geom_ok_now = True
                            break # On arrête la recherche au premier point au-dessus

            # ── 3. Mise à jour des drapeaux (Fenêtre de tolérance) ───────────
            # On laisse 1.5 seconde de tolérance pour que la déformation et le passage géométrique se "croisent"
            if net_ok_now:  net_flag_frames = int(fps * 1.5)
            if geom_ok_now: geom_flag_frames = int(fps * 1.5)

            camera_moved = (current_hoop_box is None or prev_hoop_box is None or
                            max(abs(current_hoop_box[i]-prev_hoop_box[i]) for i in range(4)) > HOOP_EPSILON)

            if camera_moved or prev_court_kp_xy is None:
                H_new = compute_homography(court_kp_xy, court_kp_conf)
                if H_new is not None:
                    H_new = H_new / H_new[2,2]
                    H_smooth = H_new if H_smooth is None else (HOMOGRAPHY_SMOOTHING*H_new + (1-HOMOGRAPHY_SMOOTHING)*H_smooth)
                    H_smooth = H_smooth / H_smooth[2,2]
                if court_kp_xy is not None: tps_warper.fit(court_kp_xy, court_kp_conf)
                prev_court_kp_xy, prev_court_kp_conf = court_kp_xy, court_kp_conf
                if current_hoop_box is not None: prev_hoop_box = current_hoop_box
                reused_homography = False
            else:
                court_kp_xy, court_kp_conf, reused_homography = prev_court_kp_xy, prev_court_kp_conf, True

            geom_ok = geom_flag_frames > 0
            net_ok  = net_flag_frames > 0

            # ── 4. Validation finale avec COOLDOWN ───────────────────────────
            if geom_ok and net_ok and current_hoop_box and H_smooth is not None:
                if (frame_count - last_shot_frame) >= int(fps * SHOT_COOLDOWN_SECONDS):
                    try:
                        pt = cv2.perspectiveTransform(np.array([[[(hx1+hx2)/2.0, float(hy2)]]], dtype=np.float32), H_smooth)
                        active_basket = "left" if pt[0,0,0] < COURT_L/2 else "right"
                        start_frame   = max(1, frame_count - delay_frames)
                        
                        highlight_events[start_frame] = active_basket
                        last_shot_frame  = frame_count # 🕒 Enregistre le tir pour le cooldown
                        geom_flag_frames = 0           # Reset immédiat
                        net_flag_frames  = 0           # Reset immédiat
                    except Exception:
                        pass

            if geom_flag_frames > 0: geom_flag_frames -= 1
            if net_flag_frames > 0:  net_flag_frames -= 1

            # Mise à jour dashboard 2D
            current_frame_tids = []
            for t in tracks:
                tid = int(t[4]) if len(t) > 4 else -1
                if tid == -1: continue
                current_frame_tids.append(tid)
                foot = tps_warper.project_foot((t[0]+t[2])/2.0, float(t[3]))
                if foot is not None:
                    if tid in persistent_2d_state:
                        px_, py_ = persistent_2d_state[tid]["pos"]
                        persistent_2d_state[tid] = {"pos": (SMOOTHING_2D*foot[0]+(1-SMOOTHING_2D)*px_, SMOOTHING_2D*foot[1]+(1-SMOOTHING_2D)*py_), "lost_frames": 0}
                    else: persistent_2d_state[tid] = {"pos": foot, "lost_frames": 0}

            for tid in list(persistent_2d_state.keys()):
                if tid not in current_frame_tids:
                    persistent_2d_state[tid]["lost_frames"] += 1
                    if persistent_2d_state[tid]["lost_frames"] > DASHBOARD_PERSISTENCE_FRAMES: del persistent_2d_state[tid]

            # 📦 Mise en buffer
            frame_data = {
                "f_idx":             frame_count,
                "augmented":         apply_virtual_logo(frame, logo_img, H_smooth, sam_masks),
                "tracks":            [t.copy() for t in tracks] if len(tracks) > 0 else [],
                "sam_masks":         sam_masks,
                "net_mask":          net_mask,
                "ball_pos":          ball_pos,
                "court_kp_xy":       court_kp_xy.copy() if court_kp_xy is not None else None,
                "court_kp_conf":     court_kp_conf.copy() if court_kp_conf is not None else None,
                "reused_homography": reused_homography,
                "hoop_dets":         hoop_dets.copy(),
                "geom_ok":           geom_ok,
                "net_ok":            net_ok,
                "H_smooth":          H_smooth.copy() if H_smooth is not None else None,
                "tps_fitted":        tps_warper.is_fitted,
                "lens_status":       lens_status,
                "persistent_2d_state": {k: {"pos": v["pos"], "lost_frames": v["lost_frames"]} for k, v in persistent_2d_state.items()},
            }
            frame_queue.append(frame_data)

            # 📤 Dépilage
            if len(frame_queue) > delay_frames:
                render_buffered_frame(frame_queue.popleft(), pbar)

        # 🧹 Vidage
        while frame_queue: render_buffered_frame(frame_queue.popleft(), pbar)

    cap.release()
    writer.release()
    print(f"\n[OK] Vidéo sauvegardée : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=Path,  default=SOURCE_VIDEO_PATH)
    parser.add_argument("--output",      type=Path,  default=OUTPUT_PATH)
    parser.add_argument("--conf-player", type=float, default=CONF_PLAYER)
    parser.add_argument("--conf-ball",   type=float, default=CONF_BALL)
    parser.add_argument("--conf-kp",     type=float, default=CONF_KP)
    parser.add_argument("--device",      type=int,   default=0)
    args = parser.parse_args()

    out_path = args.output if args.output else args.video.with_name(args.video.stem + "_demo_v19.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    process_video(args.video, out_path, args.conf_player, args.conf_ball, args.conf_kp, args.device)