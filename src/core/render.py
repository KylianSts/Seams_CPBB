"""
render.py
---------
Moteur de rendu graphique (Debug V2).
Prend le MatchState et dessine toutes les informations visuelles.
"""

import logging
import cv2
import numpy as np
from core.state import MatchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Charte graphique
# ---------------------------------------------------------------------------
C_PLAYER      = (255, 255, 0)      # Joueurs
C_TEAM_A      = (50, 50, 255)
C_TEAM_B      = (255, 255, 50)
C_NO_TEAM     = (150, 150, 150)
C_BALL        = (30, 200, 255)     # Balle
C_HOOP        = (130, 60, 110)     # Panier
C_KP_ACTIVE   = (0, 255, 0)        # Keypoints recalculés
C_KP_RECYCLED = (150, 150, 150)    # Keypoints recyclés (cam stable)
C_ZONE_ALPHA = 0.20 
C_MASK_ALPHA = 0.35
C_BG_DARK     = (15, 18, 22)
C_COURT       = (28, 32, 38)
C_LINE        = (180, 180, 190)
C_OK          = (100, 255, 100)
C_WARN        = (50, 150, 255)
C_OFF         = (80, 80, 80)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX

COURT_L, COURT_W = 28.0, 15.0   # Dimensions terrain FIBA en mètres
CONF_KP_DISPLAY  = 0.50          # Seuil minimum pour afficher un keypoint


# ===========================================================================
# 1. ANNOTATION DE LA VIDÉO PRINCIPALE
# ===========================================================================

def draw_detections(frame: np.ndarray, state: MatchState, txt_hoop: bool = False, txt_player: bool = False, show_team_crop: bool = False) -> np.ndarray:
    """Dessine les BBoxes des joueurs, de la balle et du panier."""
    out = frame.copy()

    # Panier
    if state.hoop_bbox_px:
        x1, y1, x2, y2 = map(int, state.hoop_bbox_px)
        cv2.rectangle(out, (x1, y1), (x2, y2), C_HOOP, 1, cv2.LINE_AA)
        if txt_hoop:
            cv2.putText(out, "HOOP", (x1, y1 - 5), FONT, 0.45, C_HOOP, 1, cv2.LINE_AA)

    # Balle
    if state.ball_bbox_px:
        x1, y1, x2, y2 = map(int, state.ball_bbox_px)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 11, C_BALL, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy),  7, C_BALL, -1, cv2.LINE_AA)

    # Joueurs
    for track_id, player in state.players.items():
        # Sélection de la couleur selon l'équipe
        if player.team_id == 0:
            color = C_TEAM_A   # Rouge
        elif player.team_id == 1:
            color = C_TEAM_B  # Cyan
        else:
            color = C_NO_TEAM # Gris (calibration)
        
        x1, y1, x2, y2 = map(int, player.bbox_px)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        if show_team_crop:
            w, h = x2 - x1, y2 - y1
            # Ratios identiques à ceux de detect_team.py
            cx1 = x1 + int(w * 0.25)
            cy1 = y1 + int(h * 0.20)
            cx2 = x2 - int(w * 0.25)
            cy2 = y2 - int(h * 0.40)
            # On dessine un rectangle blanc très fin à l'intérieur
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1, cv2.LINE_4)

        if txt_player:
            label = f"ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - 2), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return out

def draw_zones_and_masks(frame: np.ndarray, state: MatchState, margin_ratio: float, mask_palyer: bool = True, mask_net: bool = True) -> np.ndarray:
    """
    Dessine :
    - La zone de sécurité (trigger) avec sa propre transparence.
    - Les masques SAM avec leur propre transparence.
    - Les keypoints du terrain.
    """
    out = frame.copy()
    h_img, w_img = frame.shape[:2]

    # ---------------------------------------------------------
    # 1. COUCHE ZONE DE SÉCURITÉ (Trigger Panier)
    # ---------------------------------------------------------
    if state.hoop_bbox_px:
        overlay_zone = out.copy()
        hx1, hy1, hx2, hy2 = state.hoop_bbox_px
        w_h, h_h = hx2 - hx1, hy2 - hy1
        sx1 = int(max(0, hx1 - (w_h * margin_ratio)))
        sy1 = int(max(0, hy1 - (h_h * margin_ratio)))
        sx2 = int(min(w_img, hx2 + (w_h * margin_ratio)))
        sy2 = int(min(h_img, hy2 + (h_h * margin_ratio)))
        
        cv2.rectangle(overlay_zone, (sx1, sy1), (sx2, sy2), C_HOOP, -1)
        
        # Fusion immédiate avec la transparence dédiée à la zone (C_ZONE_ALPHA)
        cv2.addWeighted(overlay_zone, C_ZONE_ALPHA, out, 1.0 - C_ZONE_ALPHA, 0, out)

    # ---------------------------------------------------------
    # 2. COUCHE MASQUES SAM (Joueurs & Filet)
    # ---------------------------------------------------------
    overlay_masks = out.copy()
    masks_drawn = False

    # Masques joueurs
    if mask_palyer:
        if state.player_masks:
            combined = np.zeros(frame.shape[:2], dtype=bool)
            for mask in state.player_masks:
                if mask.shape == combined.shape:
                    combined |= mask
            overlay_masks[combined] = C_PLAYER
            masks_drawn = True

    # Masque filet
    if mask_net:
        if state.net_mask is not None and state.net_mask.shape == frame.shape[:2]:
            overlay_masks[state.net_mask] = C_HOOP
            masks_drawn = True

    # Fusion avec la transparence dédiée aux masques (C_MASK_ALPHA)
    if masks_drawn:
        cv2.addWeighted(overlay_masks, C_MASK_ALPHA, out, 1.0 - C_MASK_ALPHA, 0, out)

    # ---------------------------------------------------------
    # 3. COUCHE KEYPOINTS (Pas de transparence)
    # ---------------------------------------------------------
    if state.court_keypoints_px is not None:
        conf = state.court_keypoints_conf
        
        # Utilisation de la bonne variable de stabilité selon les modifs précédentes
        cam_stable = getattr(state.camera, 'is_stable_strict', getattr(state.camera, 'is_stable', False))
        color = C_KP_RECYCLED if cam_stable else C_KP_ACTIVE

        for i, kp in enumerate(state.court_keypoints_px):
            kx, ky = int(kp[0]), int(kp[1])
            if kx == 0 and ky == 0:
                continue
            if conf is not None and i < len(conf) and conf[i] < CONF_KP_DISPLAY:
                continue
            cv2.circle(out, (kx, ky), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (kx, ky), 4, color,     -1, cv2.LINE_AA)

    return out


# ===========================================================================
# 2. BANDEAU SUPÉRIEUR (HUD)
# ===========================================================================

def draw_sparkline(img: np.ndarray, history: list, x: int, y: int, w: int, h: int, color: tuple):
    """Dessine un mini-graphique temporel (sparkline) d'un signal."""
    if len(history) < 2:
        return
    
    # Fond du graphe (semi-transparent sombre) avec bordure
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 25), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 70), 1)

    vals = list(history)
    max_val = max(vals) if max(vals) > 0 else 1.0
    # On ajoute une marge de 20% pour ne pas que la courbe touche le plafond
    max_val *= 1.2 

    points = []
    for i, v in enumerate(vals):
        px = x + int(i * (w / (len(history) - 1)))
        # Calcul Y (inversé car 0 est en haut sur l'image)
        py = y + h - int((v / max_val) * h)
        points.append((px, py))

    # Dessin de la ligne reliant les points
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, 1, cv2.LINE_AA)

def build_top_hud(total_width: int, hud_h: int, state: MatchState) -> np.ndarray:
    """
    Génère le bandeau de debug supérieur en 3 zones (Responsive) :
    Gauche : Indicateurs d'activation (CAM, SAM, AR)
    Milieu : Métriques de tir (SHOT) + Graphes
    Droite : Détections audio (WHISTLE, CROWD)
    """
    hud = np.full((hud_h, total_width, 3), C_BG_DARK, dtype=np.uint8)
    cv2.line(hud, (0, hud_h - 1), (total_width, hud_h - 1), (60, 60, 70), 1)

    # --- MISE À L'ÉCHELLE DYNAMIQUE ---
    scale_factor = hud_h / 55.0  

    f_main = 0.45 * scale_factor
    f_sub = 0.40 * scale_factor
    
    y_center = int(34 * scale_factor) # Ligne de base du texte pour les zones Gauche et Droite
    y_text = int(22 * scale_factor)   # On remonte le texte pour faire la place au graphe
    y_graph = int(28 * scale_factor)  # Position Y du coin haut-gauche du graphe
    graph_w = int(60 * scale_factor)  # Largeur de la fenêtre du graphe
    graph_h = int(20 * scale_factor)  # Hauteur de la fenêtre du graphe

    # Marges adaptatives
    gap_main = int(15 * scale_factor)
    gap_small = int(8 * scale_factor)
    gap_tiny = int(6 * scale_factor)
    # -------------------------------------------------------------

    trig = state.active_triggers

    def put(img, text, x, color):
        """Écrit du texte et retourne la nouvelle position X (dessin de gauche à droite)."""
        cv2.putText(img, text, (int(x), y_center), FONT_MONO, f_main, color, 1, cv2.LINE_AA)
        return x + cv2.getTextSize(text, FONT_MONO, f_main, 1)[0][0] + int(12 * scale_factor)

    # ==========================================
    # 1. INDICATEURS D'ACTIVATION (Zone Gauche)
    # ==========================================
    cursor = gap_main

    # Stabilité caméra
    cam_ok = getattr(state.camera, 'is_stable_strict', getattr(state.camera, 'is_stable', False))
    cursor = put(hud, "CAM:STABLE" if cam_ok else "CAM:MOVING", cursor, C_OK if cam_ok else C_WARN)
    put(hud, "|", cursor - gap_tiny, C_OFF)
    cursor += gap_small

    # SAM Filet
    sam_net = trig.get("sam_net_active", False)
    cursor = put(hud, "SAM-NET:ON" if sam_net else "SAM-NET:OFF", cursor, C_OK if sam_net else C_OFF)
    put(hud, "|", cursor - gap_tiny, C_OFF)
    cursor += gap_small

    # SAM Joueurs
    sam_ply = trig.get("sam_players_active", False)
    cursor = put(hud, "SAM-PLY:ON" if sam_ply else "SAM-PLY:OFF", cursor, C_OK if sam_ply else C_OFF)
    put(hud, "|", cursor - gap_tiny, C_OFF)
    cursor += gap_small

    # Logo AR
    ar_ok = trig.get("ar_active", False)
    put(hud, "AR:ON" if ar_ok else "AR:OFF", cursor, C_OK if ar_ok else C_OFF)

    # ==========================================
    # 2. MÉTRIQUES DE TIR (Zone Milieu)
    # ==========================================
    scores = state.shot_scores if state.shot_scores else {}
    
    # On positionne ce bloc pour qu'il finisse aux 2/3 de l'écran (dessin de droite à gauche)
    cursor_mid = int((2 * total_width / 3) + (20 * scale_factor))
    
    def put_metric_with_graph(label: str, val: float, history: list):
        """Helper pour dessiner le texte et le graphe en reculant le curseur."""
        nonlocal cursor_mid
        
        # Détermination de la couleur selon le score
        if val == 0.0:
            color = C_OFF
        elif val >= 0.40:
            color = C_OK
        else:
            color = (200, 200, 200)

        # On recule le curseur de la largeur de l'élément
        cursor_mid -= (graph_w + gap_main)
        
        # Dessin du texte en haut
        txt = f"{label}:{val:.2f}"
        cv2.putText(hud, txt, (cursor_mid, y_text), FONT_MONO, f_sub, color, 1, cv2.LINE_AA)
        
        # Dessin du graphe en bas
        if history:
            draw_sparkline(hud, list(history), cursor_mid, y_graph, graph_w, graph_h, color)

    # --- 1. OPTICAL FLOW (Avec Graphe) ---
    put_metric_with_graph("OPTI", scores.get("optical", 0.0), getattr(state, 'optical_flow_history', []))

    # --- 2. NET AREA (Avec Graphe) ---
    put_metric_with_graph("NET_", scores.get("net_area", 0.0), getattr(state, 'net_area_history', []))

    # --- 3. GEOMETRY (Sans Graphe, juste du texte) ---
    val_geom = scores.get("geometry", 0.0)
    c_geom = C_OK if val_geom >= 0.40 else (C_OFF if val_geom == 0.0 else (200, 200, 200))
    txt_geom = f"GEOM:{val_geom:.2f}"
    
    tw_geom = cv2.getTextSize(txt_geom, FONT_MONO, f_sub, 1)[0][0]
    cursor_mid -= (tw_geom + gap_main)
    # On le centre verticalement car il n'a pas de graphe en dessous
    cv2.putText(hud, txt_geom, (cursor_mid, y_center), FONT_MONO, f_sub, c_geom, 1, cv2.LINE_AA)

    # --- 4. Label Global "SHOT" ---
    shot_label = "SHOT  "
    tw_shot = cv2.getTextSize(shot_label, FONT_MONO, f_main, 1)[0][0]
    cursor_mid -= (tw_shot + gap_main)
    cv2.putText(hud, shot_label, (cursor_mid, y_center), FONT_MONO, f_main, (240, 240, 240), 1, cv2.LINE_AA)

    # ==========================================
    # 3. DÉTECTIONS AUDIO (Zone Droite)
    # ==========================================
    # On commence à quelques pixels du bord droit de la vidéo et on recule
    cursor_right = total_width - gap_main

    # --- Foule (CROWD) ---
    crowd_on = getattr(state, 'is_crowd_active', False)
    c_crowd = (50, 100, 255) if crowd_on else C_OFF
    crowd_text = "CROWD:ON" if crowd_on else "CROWD:OFF"
    
    tw = cv2.getTextSize(crowd_text, FONT_MONO, f_main, 1)[0][0]
    cursor_right -= tw
    cv2.putText(hud, crowd_text, (int(cursor_right), y_center), FONT_MONO, f_main, c_crowd, 1, cv2.LINE_AA)
    
    # Séparateur
    cursor_right -= gap_main
    cv2.putText(hud, "|", (int(cursor_right), y_center), FONT_MONO, f_main, C_OFF, 1, cv2.LINE_AA)
    cursor_right -= gap_small

    # --- Sifflet (WHISTLE) ---
    whistle_on = getattr(state, 'is_whistle_active', False)
    c_whistle = (0, 215, 255) if whistle_on else C_OFF
    whistle_text = "WHISTLE:ON" if whistle_on else "WHISTLE:OFF"
    
    tw = cv2.getTextSize(whistle_text, FONT_MONO, f_main, 1)[0][0]
    cursor_right -= tw
    cv2.putText(hud, whistle_text, (int(cursor_right), y_center), FONT_MONO, f_main, c_whistle, 1, cv2.LINE_AA)

    # Label global "AUDIO"
    audio_label = "AUDIO  "
    tw = cv2.getTextSize(audio_label, FONT_MONO, f_main, 1)[0][0]
    cursor_right -= tw
    cv2.putText(hud, audio_label, (int(cursor_right), y_center), FONT_MONO, f_main, (240, 240, 240), 1, cv2.LINE_AA)

    return hud

# ===========================================================================
# 3. SIDEBAR DROITE (Minimap Horizontale + Futur Dashboard)
# ===========================================================================

def build_sidebar(sidebar_h: int, sidebar_w: int, state: MatchState) -> np.ndarray:
    """
    Génère la barre latérale droite (Responsive & 50/50).
    Moitié Haute : Minimap 2D (centrée et auto-scalée).
    Moitié Basse : Dashboard tactique adaptatif.
    """
    sidebar = np.full((sidebar_h, sidebar_w, 3), C_BG_DARK, dtype=np.uint8)

    # --- FACTEUR D'ÉCHELLE GLOBAL UI ---
    # On utilise 400px comme largeur de référence pour le texte
    scale_f = sidebar_w / 400.0  

    # Marges adaptatives
    mg_x = int(sidebar_w * 0.04)
    mg_y = int(sidebar_h * 0.03)
    half_h = sidebar_h // 2

    # ==========================================
    # MOITIÉ HAUTE : MINIMAP
    # ==========================================
    # On garantit que le terrain rentre parfaitement dans la moitié haute
    title_space = int(30 * scale_f)
    avail_w = sidebar_w - (mg_x * 2)
    avail_h = half_h - (mg_y * 2) - title_space

    # On calcule les échelles possibles en X et en Y
    sc_w = avail_w / COURT_L
    sc_h = avail_h / COURT_W
    sc = min(sc_w, sc_h) # La contrainte la plus forte l'emporte pour ne pas déborder

    court_px_w = int(COURT_L * sc)
    court_px_h = int(COURT_W * sc)

    # Centrage automatique de la minimap dans l'espace disponible
    offset_x = mg_x + (avail_w - court_px_w) // 2
    offset_y = mg_y + title_space

    # Fond du terrain
    cv2.rectangle(sidebar, (offset_x, offset_y), (offset_x + court_px_w, offset_y + court_px_h), C_COURT, -1)

    # Tailles de polices dynamiques
    f_title = 0.50 * scale_f
    f_main = 0.45 * scale_f
    f_sub = 0.40 * scale_f

    # --- En-tête : Titre + Statistiques ---
    title_y = mg_y + int(10 * scale_f)
    title_txt = "MINIMAP & DASHBOARD"
    cv2.putText(sidebar, title_txt, (offset_x, title_y), FONT_MONO, f_title, (240, 240, 245), 1, cv2.LINE_AA)
    
    stats_txt = (
        f"| PLYRS: {len(state.players)} "
        f"| AVG_SPD: {state.avg_speed_kmh:.1f} KM/H "
        f"| STD_DEV: {state.std_speed_kmh:.1f}"
    )
    tw_title = cv2.getTextSize(title_txt, FONT_MONO, f_title, 1)[0][0]
    cv2.putText(sidebar, stats_txt, (offset_x + tw_title + int(15 * scale_f), title_y),
                FONT_MONO, f_sub, C_WARN, 1, cv2.LINE_AA)

    lw = max(1, round(sc * 0.035))

    # --- PROJECTION HORIZONTALE ---
    def court_to_px(x_m, y_m):
        # On utilise le ratio calculé et les offsets de centrage
        px = int(x_m * sc) + offset_x
        py = int(y_m * sc) + offset_y
        return px, py

    def pline(pts, closed=False):
        pts_px = np.array([court_to_px(p[0], p[1]) for p in pts], dtype=np.int32)
        cv2.polylines(sidebar, [pts_px.reshape(-1, 1, 2)], closed, C_LINE, lw, cv2.LINE_AA)

    # ==========================================
    # DESSIN DU TERRAIN FIBA
    # ==========================================
    pline([[0, 0], [COURT_L, 0], [COURT_L, COURT_W], [0, COURT_W]], closed=True)
    pline([[14.0, 0.0], [14.0, COURT_W]])
    cx, cy = court_to_px(14.0, 7.5)
    cv2.circle(sidebar, (cx, cy), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)
    
    pline([[0, 5.05], [5.8, 5.05], [5.8, 9.95], [0, 9.95]], closed=True)
    pline([[28.0, 5.05], [22.2, 5.05], [22.2, 9.95], [28.0, 9.95]], closed=True)
    
    cx_lf_l, cy_lf_l = court_to_px(5.8, 7.5)
    cv2.circle(sidebar, (cx_lf_l, cy_lf_l), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)
    cx_lf_r, cy_lf_r = court_to_px(22.2, 7.5)
    cv2.circle(sidebar, (cx_lf_r, cy_lf_r), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)
    
    arc_angles = np.linspace(-1.36, 1.36, 25)
    arc_left = [[1.575 + 6.75 * np.cos(a), 7.5 + 6.75 * np.sin(a)] for a in arc_angles]
    line_3pt_left = [[0.0, 0.9], [2.99, 0.9]] + arc_left + [[2.99, 14.1], [0.0, 14.1]]
    pline(line_3pt_left, closed=False)
    line_3pt_right = [[28.0 - p[0], p[1]] for p in line_3pt_left]
    pline(line_3pt_right, closed=False)
    
    pline([[1.2, 6.9], [1.2, 8.1]])
    pline([[26.8, 6.9], [26.8, 8.1]])
    cx_b_l, cy_b_l = court_to_px(1.575, 7.5)
    cv2.circle(sidebar, (cx_b_l, cy_b_l), max(2, int(0.225 * sc)), C_LINE, lw, cv2.LINE_AA)
    cx_b_r, cy_b_r = court_to_px(26.425, 7.5)
    cv2.circle(sidebar, (cx_b_r, cy_b_r), max(2, int(0.225 * sc)), C_LINE, lw, cv2.LINE_AA)
    
    # ==========================================
    # INDICATEUR DE DIRECTION D'ATTAQUE
    # ==========================================
    target_hoop = getattr(state, 'target_hoop', None)
    
    if state.attacking_team_id is not None and target_hoop is not None:
        center_x = offset_x + (court_px_w // 2)
        arrow_y = offset_y + court_px_h + int(18 * scale_f)
        
        is_attacking_right = (target_hoop[0] > 14.0)
        dir_sign = 1 if is_attacking_right else -1
        color = C_TEAM_A if state.attacking_team_id == 0 else C_TEAM_B
        
        # Coordonnées mises à l'échelle (scale_f)
        s_4 = int(4 * scale_f)
        s_6 = int(6 * scale_f)
        s_10 = int(10 * scale_f)
        s_18 = int(18 * scale_f)
        s_24 = int(24 * scale_f)
        
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
        pts_shadow[:, 1] += max(1, int(2 * scale_f)) 
        cv2.fillPoly(sidebar, [pts_shadow], (20, 20, 25), cv2.LINE_AA)
        cv2.fillPoly(sidebar, [pts], color, cv2.LINE_AA)

    # ==========================================
    # AFFICHAGE DES JOUEURS
    # ==========================================
    dot_r = max(4, int(sc * 0.45))
    halo_th = max(1, int(2 * scale_f))
    
    for track_id, player in state.players.items():
        if player.court_pos_m is None:
            continue
        X_m, Y_m = player.court_pos_m
        
        X_m = max(0.0, min(COURT_L, X_m))
        Y_m = max(0.0, min(COURT_W, Y_m))
        px, py = court_to_px(X_m, Y_m)

        if player.team_id == 0: color = C_TEAM_A
        elif player.team_id == 1: color = C_TEAM_B
        else: color = C_NO_TEAM

        cv2.circle(sidebar, (px, py), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(sidebar, (px, py), dot_r, (255, 255, 255), halo_th, cv2.LINE_AA) 

        # HALO VERT TOURNANT (Si ouvert)
        if getattr(player, 'is_open', False):
            angle_start = (state.frame_idx * 10) % 360
            angle_end = angle_start + 180
            C_OPEN = (100, 255, 100)
            cv2.ellipse(sidebar, (px, py), (dot_r + int(5*scale_f), dot_r + int(5*scale_f)), 
                        0, angle_start, angle_end, C_OPEN, halo_th, cv2.LINE_AA)

    # --- SÉPARATEUR GAUCHE ET PLACEHOLDER DASHBOARD ---
    cv2.line(sidebar, (0, 0), (0, sidebar_h), (60, 60, 70), max(1, int(2 * scale_f)))

    # ==========================================
    # ZONE DASHBOARD TACTIQUE (PREMIUM UI)
    # ==========================================
    # On démarre le dashboard exactement à la moitié (half_h)
    dash_start_y = half_h
    cv2.line(sidebar, (0, dash_start_y), (sidebar_w, dash_start_y), (80, 80, 90), max(1, int(2 * scale_f)))
    
    # Redistribution des colonnes en pourcentages
    COL_LBL = mg_x
    COL_A   = int(sidebar_w * 0.35)
    COL_B   = int(sidebar_w * 0.55)
    COL_ALL = int(sidebar_w * 0.72)
    COL_DIF = int(sidebar_w * 0.88)
    
    y_curr = dash_start_y + int(30 * scale_f)
    
    # En-têtes (Typographie épurée et grise)
    header_color = (130, 130, 140)
    cv2.putText(sidebar, "METRIQUE", (COL_LBL, y_curr), FONT_MONO, f_sub, header_color, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "EQUIPE A", (COL_A, y_curr), FONT_MONO, f_sub, C_TEAM_A, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "EQUIPE B", (COL_B, y_curr), FONT_MONO, f_sub, C_TEAM_B, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "MATCH",    (COL_ALL, y_curr), FONT_MONO, f_sub, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(sidebar, "DIFF.",    (COL_DIF, y_curr), FONT_MONO, f_sub, C_WARN, 1, cv2.LINE_AA)
    
    cv2.line(sidebar, (mg_x, y_curr + int(15 * scale_f)), (sidebar_w - mg_x, y_curr + int(15 * scale_f)), (50, 50, 60), 1)
    
    y_curr += int(40 * scale_f)

    def draw_section(title: str, y: int) -> int:
        """Dessine un titre de section sobre avec un accent de couleur discret."""
        cv2.putText(sidebar, title.upper(), (mg_x, y), FONT_MONO, f_main, (180, 180, 190), 1, cv2.LINE_AA)
        cv2.line(sidebar, (mg_x, y + int(10 * scale_f)), (mg_x + int(100 * scale_f), y + int(10 * scale_f)), C_WARN, max(1, int(2 * scale_f)))
        return y + int(35 * scale_f)
    
    def draw_row(label: str, val_a: float, val_b: float, val_all: float, y_pos: int, unit: str = "", is_sub: bool = False) -> int:
        """Dessine une ligne avec gestion de l'indentation et du style hiérarchique."""
        font_scale = 0.35 * scale_f if is_sub else f_sub
        label_color = (160, 160, 170) if is_sub else (230, 230, 240)
        val_color = (130, 130, 140) if is_sub else (255, 255, 255)
        indent = int(25 * scale_f) if is_sub else 0
        
        fmt = "{:.1f}" if "m/s2" in unit else "{:.0f}" 
        
        str_a = f"{fmt.format(val_a)}{unit}"
        str_b = f"{fmt.format(val_b)}{unit}"
        str_all = f"{fmt.format(val_all)}{unit}"
        
        diff = val_a - val_b
        diff_val = round(diff)
        diff_txt = f"{'+' if diff_val > 0 else ''}{diff_val}"
        diff_color = (80, 80, 90) if abs(diff_val) == 0 else C_WARN

        cv2.putText(sidebar, label, (COL_LBL + indent, y_pos), FONT_MONO, font_scale, label_color, 1, cv2.LINE_AA)
        cv2.putText(sidebar, str_a, (COL_A, y_pos), FONT_MONO, font_scale, val_color, 1, cv2.LINE_AA)
        cv2.putText(sidebar, str_b, (COL_B, y_pos), FONT_MONO, font_scale, val_color, 1, cv2.LINE_AA)
        cv2.putText(sidebar, str_all, (COL_ALL, y_pos), FONT_MONO, font_scale, val_color, 1, cv2.LINE_AA)
        
        if not is_sub:
            cv2.putText(sidebar, diff_txt, (COL_DIF, y_pos), FONT_MONO, font_scale, diff_color, 1, cv2.LINE_AA)

        return y_pos + int((25 if is_sub else 40) * scale_f)

    m_a = state.team_metrics[0]
    m_b = state.team_metrics[1]
    
    # --- SECTION 1 : CINÉMATIQUE ---
    y_curr = draw_section("Performance Athletique", y_curr)
    
    y_curr = draw_row("Vitesse Moy.", m_a["avg_speed"], m_b["avg_speed"], state.avg_speed_kmh, y_curr, " km/h")
    y_curr = draw_row("Ecart-type", m_a["std_speed"], m_b["std_speed"], state.std_speed_kmh, y_curr, " km/h", is_sub=True)
    y_curr = draw_row("Vitesse Min", m_a["min_speed"], m_b["min_speed"], state.min_speed_kmh, y_curr, " km/h", is_sub=True)
    y_curr = draw_row("Vitesse Max", m_a["max_speed"], m_b["max_speed"], state.max_speed_kmh, y_curr, " km/h", is_sub=True)
    y_curr += int(10 * scale_f)

    y_curr = draw_row("Accel. Moy.", m_a["avg_accel"], m_b["avg_accel"], state.avg_accel_ms2, y_curr, " m/s2")
    y_curr = draw_row("Ecart-type", m_a["std_accel"], m_b["std_accel"], state.std_accel_ms2, y_curr, " m/s2", is_sub=True)
    y_curr = draw_row("Accel. Min", m_a["min_accel"], m_b["min_accel"], state.min_accel_ms2, y_curr, " m/s2", is_sub=True)
    y_curr = draw_row("Accel. Max", m_a["max_accel"], m_b["max_accel"], state.max_accel_ms2, y_curr, " m/s2", is_sub=True)
    
    y_curr += int(20 * scale_f)
    
    # --- SECTION 2 : TACTIQUE ---
    y_curr = draw_section("Placement & Spatialisation", y_curr)
    y_curr = draw_row("Spacing (Aire)", m_a["spacing"], m_b["spacing"], 0, y_curr, " m2")
    y_curr = draw_row("Dans Raquette", m_a["paint_count"], m_b["paint_count"], m_a["paint_count"] + m_b["paint_count"], y_curr, " jou.")

    y_curr += int(25 * scale_f)

    # --- BLOC D'ALERTE (Style épuré) ---
    total_paint = m_a["paint_count"] + m_b["paint_count"]
    alert_active = total_paint > 4
    
    alert_bg = (20, 20, 25)
    alert_border = C_WARN if alert_active else (60, 60, 70)
    
    cv2.rectangle(sidebar, (mg_x, y_curr), (sidebar_w - mg_x, y_curr + int(45 * scale_f)), alert_bg, -1)
    cv2.rectangle(sidebar, (mg_x, y_curr), (sidebar_w - mg_x, y_curr + int(45 * scale_f)), alert_border, 1)

    alert_txt = f"CHARGES RAQUETTES : {int(total_paint)} JOUEURS"
    if alert_active:
        alert_txt += " | ALERTE REBOND"
        
    text_color = C_OK if not alert_active else C_WARN
    cv2.putText(sidebar, alert_txt, (mg_x + int(20 * scale_f), y_curr + int(28 * scale_f)), FONT_MONO, f_sub, text_color, 1, cv2.LINE_AA)

    return sidebar

# ===========================================================================
# 4. ASSEMBLAGE FINAL
# ===========================================================================

def render_debug_frame(frame: np.ndarray, state: MatchState, sidebar_w: int, hud_h: int, margin_ratio: float = 1.50) -> np.ndarray:
    """
    Assemble la vidéo, le HUD et la Sidebar avec des dimensions dynamiques.
    Conforme au schéma : Gauche [ HUD + Vidéo ] | Droite [ Sidebar complète ]
    """
    h, w = frame.shape[:2]

    # 1. Préparation de la vidéo annotée
    main = draw_zones_and_masks(frame, state, margin_ratio, mask_palyer=False, mask_net=True)
    main = draw_detections(main, state, txt_hoop=False, txt_player=True, show_team_crop=True)

    # 2. Création du HUD adaptatif
    hud = build_top_hud(w, hud_h, state)

    # 3. Assemblage Bloc Gauche (HUD + Vidéo)
    left_pane = np.vstack([hud, main])

    # 4. Création Bloc Droit (Sidebar). Sa hauteur doit matcher le left_pane.
    sidebar_h = h + hud_h
    sidebar = build_sidebar(sidebar_h=sidebar_h, sidebar_w=sidebar_w, state=state)

    # 5. Assemblage final horizontal
    return np.hstack([left_pane, sidebar])