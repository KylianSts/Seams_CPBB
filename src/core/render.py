"""
render.py
---------
Moteur de rendu graphique et d'interface utilisateur (UI).
Assemble les détections, la minimap et le tableau de bord tactique
en une seule image composite pour l'export vidéo.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from core.filters import get_torso_box, FiltersConfig

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.state import MatchState, FrameSnapshot

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. CONFIGURATION VISUELLE (Design System)
# ===========================================================================

@dataclass
class RenderConfig:
    """Charte graphique et paramètres de rendu globaux (Couleurs BGR)."""
    
    # --- Couleurs des Entités ---
    color_team_a: Tuple[int, int, int] = (50, 50, 255)     # Rouge
    color_team_b: Tuple[int, int, int] = (255, 255, 50)    # Cyan
    color_no_team: Tuple[int, int, int] = (150, 150, 150)  # Gris
    color_ball: Tuple[int, int, int] = (30, 200, 255)      # Jaune/Orange
    color_hoop: Tuple[int, int, int] = (130, 60, 110)      # Violet foncé
    
    # --- Couleurs de l'Interface (HUD / Sidebar) ---
    color_bg_dark: Tuple[int, int, int] = (15, 18, 22)
    color_court_bg: Tuple[int, int, int] = (28, 32, 38)
    color_lines: Tuple[int, int, int] = (180, 180, 190)
    color_text_main: Tuple[int, int, int] = (240, 240, 245)
    color_text_sub: Tuple[int, int, int] = (160, 160, 170)
    
    # --- Couleurs de Statut ---
    color_ok: Tuple[int, int, int] = (100, 255, 100)       # Vert
    color_warn: Tuple[int, int, int] = (50, 150, 255)      # Orange
    color_off: Tuple[int, int, int] = (80, 80, 80)         # Gris sombre
    color_kp_active: Tuple[int, int, int] = (0, 255, 0)
    color_kp_recycled: Tuple[int, int, int] = (150, 150, 150)
    
    # --- Polices et Transparence ---
    font_main: int = cv2.FONT_HERSHEY_SIMPLEX
    font_mono: int = cv2.FONT_HERSHEY_DUPLEX
    alpha_zone: float = 0.20
    alpha_mask: float = 0.35
    
    # --- Géométrie FIBA ---
    court_length_m: float = 28.0
    court_width_m: float = 15.0
    min_kp_confidence: float = 0.50

    # --- Options d'affichage (Toggles) ---
    show_player_text: bool = True
    show_player_masks: bool = False
    show_torso_debug: bool = True


# ===========================================================================
# 2. ANNOTATEUR VIDÉO (Surcouche Terrain)
# ===========================================================================

class VideoAnnotator:
    """Gère le dessin direct sur les pixels de la caméra (BBoxes, Masques, Keypoints)."""
    
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def draw_detections(self, frame: np.ndarray, state: 'FrameSnapshot') -> np.ndarray:
        """Trace les Bounding Boxes des joueurs, de la balle et du panier."""
        out = frame.copy()

        # 1. Panier (avec effet de feedback visuel si tir parfait)
        if state.hoop_bbox_px:
            x1, y1, x2, y2 = map(int, state.hoop_bbox_px)
            color = self.cfg.color_ok if state.is_perfect_shot else self.cfg.color_hoop
            thickness = 3 if state.is_perfect_shot else 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # 2. Balle
        if state.ball_bbox_px:
            x1, y1, x2, y2 = map(int, state.ball_bbox_px)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), 11, self.cfg.color_ball, 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 7, self.cfg.color_ball, -1, cv2.LINE_AA)

        # 3. Joueurs
        for track_id, player in state.players.items():
            if player.team_id == 0:
                color = self.cfg.color_team_a
            elif player.team_id == 1:
                color = self.cfg.color_team_b
            else:
                color = self.cfg.color_no_team
            
            x1, y1, x2, y2 = map(int, player.bbox_px)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            if getattr(self.cfg, 'show_torso_debug', False):
                    try:
                        tx1, ty1, tx2, ty2 = get_torso_box(player.bbox_px, FiltersConfig())
                        cv2.rectangle(out, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Erreur dessin torse ID {track_id} : {e}")

            # Label ID et Debug GMM
            if self.cfg.show_player_text:
                lines = [f"ID:{track_id}"]
                
                # Extraction des données GMM de la frame courante
                if player.gmm_history:
                    # On prend la dernière prédiction instantanée de la frame
                    _, pA, pB, occ = player.gmm_history[-1]
                    lines.append(f"A:{pA*100:.0f}% | B:{pB*100:.0f}% | Occ:{occ*100:.0f}%")
                    
                    # Affichage de la VRAIE moyenne bidirectionnelle calculée par le pipeline
                    if hasattr(player, '_debug_bidi_avg'):
                        avg_A, avg_B, w_len = player._debug_bidi_avg
                        lines.append(f"Moy: A:{avg_A*100:.0f}% | B:{avg_B*100:.0f}%")

                # Dessin du texte multi-lignes (de bas en haut)
                y_offset = int(y1) - 4
                for line in reversed(lines):
                    (tw, th), _ = cv2.getTextSize(line, self.cfg.font_main, 0.4, 1)
                    
                    cv2.rectangle(out, (int(x1), y_offset - th - 4), (int(x1) + tw + 4, y_offset + 2), color, -1)
                    
                    txt_color = (0, 0, 0) if player.team_id == 1 else (255, 255, 255)
                    if player.team_id is None: txt_color = (0, 0, 0)
                        
                    cv2.putText(out, line, (int(x1) + 2, y_offset - 2), self.cfg.font_main, 0.4, txt_color, 1, cv2.LINE_AA)
                    
                    y_offset -= (th + 6)
        return out

    def draw_overlays(self, frame: np.ndarray, state: 'FrameSnapshot') -> np.ndarray:
        """Applique les calques semi-transparents (Masques SAM) et les keypoints."""
        out = frame.copy()
        overlay = out.copy()
        masks_drawn = False

        # 1. Masques des joueurs
        if state.player_masks and self.cfg.show_player_masks:
            combined = np.zeros(frame.shape[:2], dtype=bool)
            for mask in state.player_masks:
                if mask.shape == combined.shape:
                    # Conversion à la volée si le masque est un float (méthode capsule)
                    # On considère comme True tout pixel ayant un peu d'opacité (> 0.05)
                    current_mask = mask if mask.dtype == bool else (mask > 0.05)
                    combined |= current_mask
                    
            overlay[combined] = self.cfg.color_no_team
            masks_drawn = True

        # 2. Masque du filet
        if state.net_mask is not None and state.net_mask.shape == frame.shape[:2]:
            overlay[state.net_mask] = self.cfg.color_hoop
            masks_drawn = True

        if masks_drawn:
            cv2.addWeighted(overlay, self.cfg.alpha_mask, out, 1.0 - self.cfg.alpha_mask, 0, out)

        # 3. Keypoints du terrain (YOLO-Pose)
        if state.court_keypoints_px is not None:
            conf = state.court_keypoints_conf
            color = self.cfg.color_kp_recycled if state.camera_stable else self.cfg.color_kp_active

            for i, kp in enumerate(state.court_keypoints_px):
                kx, ky = int(kp[0]), int(kp[1])
                if kx == 0 and ky == 0:
                    continue
                if conf is not None and i < len(conf) and conf[i] < self.cfg.min_kp_confidence:
                    continue
                cv2.circle(out, (kx, ky), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(out, (kx, ky), 4, color, -1, cv2.LINE_AA)

        return out


# ===========================================================================
# 3. MOTEUR DU HEAD-UP DISPLAY (Barre Supérieure)
# ===========================================================================

class HudRenderer:
    """Génère le bandeau supérieur de télémétrie et statuts de l'IA."""
    
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def _draw_sparkline(self, img: np.ndarray, history: List[float], x: int, y: int, w: int, h: int, color: Tuple[int, int, int]) -> None:
        """Trace un graphique linéaire temporel miniature."""
        if len(history) < 2:
            return
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 25), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 70), 1)

        max_val = max(history) if max(history) > 0 else 1.0
        max_val *= 1.2 

        points = []
        for i, v in enumerate(history):
            px = x + int(i * (w / (len(history) - 1)))
            py = y + h - int((v / max_val) * h)
            points.append((px, py))

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color, 1, cv2.LINE_AA)

    def render(self, width: int, height: int, state: 'FrameSnapshot') -> np.ndarray:
        """Construit le HUD avec un layout responsive."""
        hud = np.full((height, width, 3), self.cfg.color_bg_dark, dtype=np.uint8)
        cv2.line(hud, (0, height - 1), (width, height - 1), (60, 60, 70), 1)

        # Échelle responsive
        sf = height / 55.0  
        f_main, f_sub = 0.45 * sf, 0.40 * sf
        y_center = int(34 * sf)
        gap = int(15 * sf)

        def put_txt(x_pos: int, text: str, color: Tuple[int, int, int]) -> int:
            cv2.putText(hud, text, (int(x_pos), y_center), self.cfg.font_mono, f_main, color, 1, cv2.LINE_AA)
            return x_pos + cv2.getTextSize(text, self.cfg.font_mono, f_main, 1)[0][0] + int(12 * sf)

        # --- GAUCHE : État des Triggers ---
        cx = gap
        trig = state.active_triggers
        
        cx = put_txt(cx, "CAM:STABLE" if state.camera_stable else "CAM:MOVING", self.cfg.color_ok if state.camera_stable else self.cfg.color_warn)
        cx = put_txt(cx, "SAM:ON" if trig.get("sam_net_active") or trig.get("sam_players_active") else "SAM:OFF", self.cfg.color_ok if (trig.get("sam_net_active") or trig.get("sam_players_active")) else self.cfg.color_off)
        cx = put_txt(cx, "AR:ON" if trig.get("ar_active") else "AR:OFF", self.cfg.color_ok if trig.get("ar_active") else self.cfg.color_off)

        # --- DROITE : Télémétrie de Tir (Alignée à droite) ---
        scores = state.shot_scores
        y_txt, y_graph = int(22 * sf), int(28 * sf)
        gw, gh = int(60 * sf), int(20 * sf)
        
        cx_right = width - gap

        def put_metric(label: str, val: float, history: List[float]):
            nonlocal cx_right
            color = self.cfg.color_off if val == 0.0 else (self.cfg.color_ok if val >= 0.40 else (200, 200, 200))
            cx_right -= (gw + gap)
            
            txt = f"{label}:{val:.2f}"
            cv2.putText(hud, txt, (cx_right, y_txt), self.cfg.font_mono, f_sub, color, 1, cv2.LINE_AA)
            if history:
                self._draw_sparkline(hud, history, cx_right, y_graph, gw, gh, color)

        # Tracé des métriques de tir en partant de la droite
        put_metric("OPTI", scores.get("optical", 0.0), state.optical_flow_history)
        put_metric("NET_", scores.get("net_area", 0.0), state.net_area_history)
        
        # Geometrie (Sans graphe)
        val_geom = scores.get("geometry", 0.0)
        c_geom = self.cfg.color_ok if val_geom >= 0.40 else (self.cfg.color_off if val_geom == 0.0 else (200, 200, 200))
        txt_geom = f"GEOM:{val_geom:.2f}"
        tw_geom = cv2.getTextSize(txt_geom, self.cfg.font_mono, f_sub, 1)[0][0]
        
        cx_right -= (tw_geom + gap)
        cv2.putText(hud, txt_geom, (cx_right, y_center), self.cfg.font_mono, f_sub, c_geom, 1, cv2.LINE_AA)

        # Label de section
        shot_lbl = "SHOT TELEMETRY"
        tw_lbl = cv2.getTextSize(shot_lbl, self.cfg.font_mono, f_main, 1)[0][0]
        cx_right -= (tw_lbl + gap * 2)
        cv2.putText(hud, shot_lbl, (cx_right, y_center), self.cfg.font_mono, f_main, self.cfg.color_text_main, 1, cv2.LINE_AA)

        return hud


# ===========================================================================
# 4. MOTEUR DE LA SIDEBAR (Minimap & Dashboard)
# ===========================================================================

class SidebarRenderer:
    """Génère le panneau latéral contenant la projection 2D et les statistiques tactiques."""
    
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def render(self, width: int, height: int, state: 'FrameSnapshot') -> np.ndarray:
        sidebar = np.full((height, width, 3), self.cfg.color_bg_dark, dtype=np.uint8)
        sf = width / 900.0  
        
        mg_x, mg_y = int(width * 0.04), int(height * 0.03)
        half_h = height // 2

        # 1. Construction de la Minimap (Moitié supérieure)
        self._draw_minimap(sidebar, state, width, half_h, mg_x, mg_y, sf)

        # 2. Construction du Dashboard (Moitié inférieure)
        dash_y = half_h + int(20 * sf)
        self._draw_dashboard(sidebar, state, width, dash_y, mg_x, sf)

        # Ligne de séparation verticale
        cv2.line(sidebar, (0, 0), (0, height), (60, 60, 70), max(1, int(2 * sf)))
        
        return sidebar

    def _draw_minimap(self, canvas: np.ndarray, state: 'FrameSnapshot', w: int, h: int, mg_x: int, mg_y: int, sf: float) -> None:
        title_space = int(40 * sf)
        avail_w, avail_h = w - (mg_x * 2), h - (mg_y * 2) - title_space

        sc_w = avail_w / self.cfg.court_length_m
        sc_h = avail_h / self.cfg.court_width_m
        sc = min(sc_w, sc_h)

        court_w_px = int(self.cfg.court_length_m * sc)
        court_h_px = int(self.cfg.court_width_m * sc)

        off_x = mg_x + (avail_w - court_w_px) // 2
        off_y = mg_y + title_space

        cv2.rectangle(canvas, (off_x, off_y), (off_x + court_w_px, off_y + court_h_px), self.cfg.color_court_bg, -1)

        def c2px(x_m: float, y_m: float) -> Tuple[int, int]:
            return int(x_m * sc) + off_x, int(y_m * sc) + off_y

        def pline(pts, closed=False):
            pts_px = np.array([c2px(p[0], p[1]) for p in pts], dtype=np.int32)
            cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], closed, self.cfg.color_lines, lw, cv2.LINE_AA)

        lw = max(1, round(sc * 0.035))
        
        # --- LIGNES DU TERRAIN FIBA COMPLET ---
        pline([[0, 0], [28, 0], [28, 15], [0, 15]], closed=True) # Bordures
        pline([[14.0, 0.0], [14.0, 15.0]]) # Ligne médiane
        cv2.circle(canvas, c2px(14.0, 7.5), int(1.8 * sc), self.cfg.color_lines, lw, cv2.LINE_AA) # Rond central
        
        pline([[0, 5.05], [5.8, 5.05], [5.8, 9.95], [0, 9.95]], closed=True) # Raquette Gauche
        pline([[28.0, 5.05], [22.2, 5.05], [22.2, 9.95], [28.0, 9.95]], closed=True) # Raquette Droite
        
        cv2.circle(canvas, c2px(5.8, 7.5), int(1.8 * sc), self.cfg.color_lines, lw, cv2.LINE_AA) # Lancer franc gauche
        cv2.circle(canvas, c2px(22.2, 7.5), int(1.8 * sc), self.cfg.color_lines, lw, cv2.LINE_AA) # Lancer franc droite
        
        # Lignes à 3 points
        arc_angles = np.linspace(-1.36, 1.36, 25)
        arc_left = [[1.575 + 6.75 * np.cos(a), 7.5 + 6.75 * np.sin(a)] for a in arc_angles]
        line_3pt_left = [[0.0, 0.9], [2.99, 0.9]] + arc_left + [[2.99, 14.1], [0.0, 14.1]]
        pline(line_3pt_left)
        line_3pt_right = [[28.0 - p[0], p[1]] for p in line_3pt_left]
        pline(line_3pt_right)
        
        # Arceaux et Planches
        pline([[1.2, 6.9], [1.2, 8.1]])
        pline([[26.8, 6.9], [26.8, 8.1]])
        cv2.circle(canvas, c2px(1.575, 7.5), max(2, int(0.225 * sc)), self.cfg.color_lines, lw, cv2.LINE_AA)
        cv2.circle(canvas, c2px(26.425, 7.5), max(2, int(0.225 * sc)), self.cfg.color_lines, lw, cv2.LINE_AA)

        # --- FLÈCHE D'ATTAQUE ---
        if state.attacking_team_id is not None and state.target_hoop is not None:
            center_x = off_x + (court_w_px // 2)
            arrow_y = off_y + court_h_px + int(18 * sf)
            
            is_attacking_right = (state.target_hoop[0] > 14.0)
            dir_sign = 1 if is_attacking_right else -1
            color = self.cfg.color_team_a if state.attacking_team_id == 0 else self.cfg.color_team_b
            
            s_4, s_6, s_10, s_18, s_24 = int(4*sf), int(6*sf), int(10*sf), int(18*sf), int(24*sf)
            
            pts = np.array([
                [center_x - s_18 * dir_sign, arrow_y - s_4], 
                [center_x + s_6 * dir_sign,  arrow_y - s_4], 
                [center_x + s_6 * dir_sign,  arrow_y - s_10],
                [center_x + s_24 * dir_sign, arrow_y],       
                [center_x + s_6 * dir_sign,  arrow_y + s_10],
                [center_x + s_6 * dir_sign,  arrow_y + s_4], 
                [center_x - s_18 * dir_sign, arrow_y + s_4]  
            ], np.int32)
            
            pts_shadow = pts.copy()
            pts_shadow[:, 1] += max(1, int(2 * sf)) 
            cv2.fillPoly(canvas, [pts_shadow], (20, 20, 25), cv2.LINE_AA)
            cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)

        # --- JOUEURS ---
        dot_r = max(4, int(sc * 0.45))
        halo_th = max(1, int(2 * sf))
        
        for tid, player in state.players.items():
            if player.court_pos_m is None:
                continue
            
            px, py = c2px(max(0.0, min(self.cfg.court_length_m, player.court_pos_m[0])), 
                          max(0.0, min(self.cfg.court_width_m, player.court_pos_m[1])))

            color = self.cfg.color_team_a if player.team_id == 0 else (self.cfg.color_team_b if player.team_id == 1 else self.cfg.color_no_team)

            cv2.circle(canvas, (px, py), dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), dot_r, (255, 255, 255), halo_th, cv2.LINE_AA)

            if player.is_open:
                angle_start = (state.frame_idx * 10) % 360
                cv2.ellipse(canvas, (px, py), (dot_r + int(5*sf), dot_r + int(5*sf)), 
                            0, angle_start, angle_start + 180, self.cfg.color_ok, halo_th, cv2.LINE_AA)

    def _draw_dashboard(self, canvas: np.ndarray, state: 'FrameSnapshot', w: int, start_y: int, mg_x: int, sf: float) -> None:
        cv2.line(canvas, (0, start_y), (w, start_y), (80, 80, 90), max(1, int(2 * sf)))
        
        # Colonnes ajustées pour faire de la place à la colonne DIFF.
        col_lbl = mg_x
        col_a   = int(w * 0.32)
        col_b   = int(w * 0.50)
        col_all = int(w * 0.68)
        col_dif = int(w * 0.85)
        
        f_sub = 0.40 * sf
        y_curr = start_y + int(30 * sf)
        
        cv2.putText(canvas, "METRIQUE", (col_lbl, y_curr), self.cfg.font_mono, f_sub, self.cfg.color_text_sub, 1, cv2.LINE_AA)
        cv2.putText(canvas, "EQUIPE A", (col_a, y_curr), self.cfg.font_mono, f_sub, self.cfg.color_team_a, 1, cv2.LINE_AA)
        cv2.putText(canvas, "EQUIPE B", (col_b, y_curr), self.cfg.font_mono, f_sub, self.cfg.color_team_b, 1, cv2.LINE_AA)
        cv2.putText(canvas, "MATCH", (col_all, y_curr), self.cfg.font_mono, f_sub, self.cfg.color_text_main, 1, cv2.LINE_AA)
        cv2.putText(canvas, "DIFF.", (col_dif, y_curr), self.cfg.font_mono, f_sub, self.cfg.color_warn, 1, cv2.LINE_AA)
        
        cv2.line(canvas, (mg_x, y_curr + int(15 * sf)), (w - mg_x, y_curr + int(15 * sf)), (50, 50, 60), 1)
        y_curr += int(40 * sf)

        def draw_section(title: str, y: int) -> int:
            cv2.putText(canvas, title.upper(), (mg_x, y), self.cfg.font_mono, 0.45 * sf, self.cfg.color_text_sub, 1, cv2.LINE_AA)
            cv2.line(canvas, (mg_x, y + int(10 * sf)), (mg_x + int(100 * sf), y + int(10 * sf)), self.cfg.color_warn, max(1, int(2 * sf)))
            return y + int(35 * sf)

        def row(label: str, v_a: float, v_b: float, v_all: float, y: int, unit: str = "", sub: bool = False) -> int:
            f_scale = 0.35 * sf if sub else f_sub
            c_lbl = (160, 160, 170) if sub else (230, 230, 240)
            c_val = (130, 130, 140) if sub else (255, 255, 255)
            ind = int(25 * sf) if sub else 0
            
            fmt = "{:.1f}" if "m/s2" in unit else "{:.0f}" 
            diff_val = (v_a - v_b)
            diff_fmt = round(diff_val, 1) if "m/s2" in unit else round(diff_val)
            diff_color = (80, 80, 90) if abs(diff_fmt) == 0 else self.cfg.color_warn
            
            cv2.putText(canvas, label, (col_lbl + ind, y), self.cfg.font_mono, f_scale, c_lbl, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{fmt.format(v_a)}{unit}", (col_a, y), self.cfg.font_mono, f_scale, c_val, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{fmt.format(v_b)}{unit}", (col_b, y), self.cfg.font_mono, f_scale, c_val, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{fmt.format(v_all)}{unit}", (col_all, y), self.cfg.font_mono, f_scale, c_val, 1, cv2.LINE_AA)
            
            if not sub:
                cv2.putText(canvas, f"{'+' if diff_fmt > 0 else ''}{diff_fmt}", (col_dif, y), self.cfg.font_mono, f_scale, diff_color, 1, cv2.LINE_AA)

            return y + int((25 if sub else 40) * sf)

        m_a, m_b = state.team_metrics.get(0, {}), state.team_metrics.get(1, {})

        y_curr = draw_section("Performance Athletique", y_curr)
        y_curr = row("Vitesse Moy.", m_a.get("avg_speed", 0), m_b.get("avg_speed", 0), state.avg_speed_kmh, y_curr, " km/h")
        y_curr = row("Ecart-type", m_a.get("std_speed", 0), m_b.get("std_speed", 0), state.std_speed_kmh, y_curr, " km/h", True)
        y_curr = row("Vitesse Min", m_a.get("min_speed", 0), m_b.get("min_speed", 0), state.min_speed_kmh, y_curr, " km/h", True)
        y_curr = row("Vitesse Max", m_a.get("max_speed", 0), m_b.get("max_speed", 0), state.max_speed_kmh, y_curr, " km/h", True)
        
        y_curr += int(10 * sf)
        y_curr = row("Accel. Moy.", m_a.get("avg_accel", 0), m_b.get("avg_accel", 0), state.avg_accel_ms2, y_curr, " m/s2")
        y_curr = row("Ecart-type", m_a.get("std_accel", 0), m_b.get("std_accel", 0), state.std_accel_ms2, y_curr, " m/s2", True)
        y_curr = row("Accel. Min", m_a.get("min_accel", 0), m_b.get("min_accel", 0), state.min_accel_ms2, y_curr, " m/s2", True)
        y_curr = row("Accel. Max", m_a.get("max_accel", 0), m_b.get("max_accel", 0), state.max_accel_ms2, y_curr, " m/s2", True)
        
        y_curr += int(20 * sf)
        y_curr = draw_section("Placement & Spatialisation", y_curr)
        y_curr = row("Spacing (Aire)", m_a.get("spacing", 0), m_b.get("spacing", 0), 0, y_curr, " m2")
        y_curr = row("Dans Raquette", m_a.get("paint_count", 0), m_b.get("paint_count", 0), m_a.get("paint_count", 0) + m_b.get("paint_count", 0), y_curr, " jou.")

        y_curr += int(25 * sf)

        total_paint = m_a.get("paint_count", 0) + m_b.get("paint_count", 0)
        alert_active = total_paint > 4
        
        alert_bg = (20, 20, 25)
        alert_border = self.cfg.color_warn if alert_active else (60, 60, 70)
        
        cv2.rectangle(canvas, (mg_x, y_curr), (w - mg_x, y_curr + int(45 * sf)), alert_bg, -1)
        cv2.rectangle(canvas, (mg_x, y_curr), (w - mg_x, y_curr + int(45 * sf)), alert_border, 1)

        alert_txt = f"CHARGES RAQUETTES : {int(total_paint)} JOUEURS"
        if alert_active:
            alert_txt += " | ALERTE REBOND"
            
        text_color = self.cfg.color_ok if not alert_active else self.cfg.color_warn
        cv2.putText(canvas, alert_txt, (mg_x + int(20 * sf), y_curr + int(28 * sf)), self.cfg.font_mono, f_sub, text_color, 1, cv2.LINE_AA)

# ===========================================================================
# 5. ORCHESTRATEUR PRINCIPAL
# ===========================================================================

class MatchRenderer:
    """Chef d'orchestre assemblant les différentes couches visuelles de l'application."""
    
    def __init__(self, config: RenderConfig = RenderConfig()):
        self.config = config
        self.video_annotator = VideoAnnotator(config)
        self.hud_renderer = HudRenderer(config)
        self.sidebar_renderer = SidebarRenderer(config)

    def render_frame(self, frame: np.ndarray, state: 'FrameSnapshot', sidebar_w: int, hud_h: int) -> np.ndarray:
        """Assemble la vidéo annotée, le HUD et la Sidebar en une seule frame de sortie."""
        h, w = frame.shape[:2]

        # 1. Traitement de la vidéo principale
        main_view = self.video_annotator.draw_overlays(frame, state)
        main_view = self.video_annotator.draw_detections(main_view, state)

        # 2. Rendu des panneaux d'interface
        hud_view = self.hud_renderer.render(w, hud_h, state)
        sidebar_view = self.sidebar_renderer.render(sidebar_w, h + hud_h, state)

        # 3. Assemblage spatial (Stack vertical pour le bloc gauche, horizontal pour le final)
        left_pane = np.vstack([hud_view, main_view])
        final_frame = np.hstack([left_pane, sidebar_view])

        return final_frame

# Rétro-compatibilité pour un usage fonctionnel direct si nécessaire
def render_debug_frame(frame: np.ndarray, state: 'FrameSnapshot', sidebar_w: int, hud_h: int) -> np.ndarray:
    renderer = MatchRenderer()
    return renderer.render_frame(frame, state, sidebar_w, hud_h)