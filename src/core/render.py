"""
render.py
---------
Moteur de rendu graphique (Debug V1).
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

def draw_detections(frame: np.ndarray, state: MatchState, txt_hoop: bool = False, txt_player: bool = False, show_team_crop: bool = True) -> np.ndarray:
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

def build_top_hud(total_width: int, state: MatchState) -> np.ndarray:
    """
    Génère le bandeau de debug supérieur.
    Contient : état des triggers et sifflet (gauche) | scores de tir continus (droite).
    """
    HUD_H = 55
    hud = np.full((HUD_H, total_width, 3), C_BG_DARK, dtype=np.uint8)
    cv2.line(hud, (0, HUD_H - 1), (total_width, HUD_H - 1), (60, 60, 70), 1)

    trig = state.active_triggers
    y = 34  # Ligne de base du texte

    def put(img, text, x, color):
        """Écrit du texte et retourne la nouvelle position X."""
        cv2.putText(img, text, (int(x), int(y)), FONT_MONO, 0.45, color, 1, cv2.LINE_AA)
        return x + cv2.getTextSize(text, FONT_MONO, 0.45, 1)[0][0] + 12

    # ==========================================
    # 1. INDICATEURS D'ÉTAT (Alignés à gauche)
    # ==========================================
    cursor = 15

    # Stabilité caméra (utilise la variable stricte si on a fait la modif précédente)
    cam_ok = getattr(state.camera, 'is_stable', False) 
    if hasattr(state.camera, 'is_stable_strict'):
        cam_ok = state.camera.is_stable_strict

    cursor = put(hud, "CAM:STABLE" if cam_ok else "CAM:MOVING",
                 cursor, C_OK if cam_ok else C_WARN)
    put(hud, "|", cursor - 6, C_OFF)
    cursor += 8

    # SAM Filet
    sam_net = trig.get("sam_net_active", False)
    cursor = put(hud, "SAM-NET:ON" if sam_net else "SAM-NET:OFF",
                 cursor, C_OK if sam_net else C_OFF)
    put(hud, "|", cursor - 6, C_OFF)
    cursor += 8

    # SAM Joueurs
    sam_ply = trig.get("sam_players_active", False)
    cursor = put(hud, "SAM-PLY:ON" if sam_ply else "SAM-PLY:OFF",
                 cursor, C_OK if sam_ply else C_OFF)
    put(hud, "|", cursor - 6, C_OFF)
    cursor += 8

    # Logo AR
    ar_ok = trig.get("ar_active", False)
    cursor = put(hud, "AR:ON" if ar_ok else "AR:OFF",
                 cursor, C_OK if ar_ok else C_OFF)
    put(hud, "|", cursor - 6, C_OFF)
    cursor += 8

    # Sifflet (Nouveau format sobre)
    whistle_on = state.is_whistle_active
    c_whistle = (0, 215, 255) if whistle_on else C_OFF  # Jaune/Orange si actif
    cursor = put(hud, "WHISTLE:ON" if whistle_on else "WHISTLE:OFF",
                 cursor, c_whistle)

    # ==========================================
    # 2. SCORES DE TIR (Alignés à droite, toujours visibles)
    # ==========================================
    
    # Mapping exact des clés générées dans detect_shots.py
    metrics = [
        ("geometry", "GEOM"),
        ("net_area", "NET_"),
        ("deform", "DEFO"),
        ("optical", "OPTI"),
        ("occlusion", "OCCL"),
        ("velocity", "VELO")
    ]

    scores = state.shot_scores if state.shot_scores else {}
    
    # On calcule l'affichage de droite à gauche pour bien le coller au bord droit
    cursor_right = (2*total_width/3)
    elements_to_draw = []

    for key, short_name in reversed(metrics):
        val = scores.get(key, 0.0)
        text = f"{short_name}:{val:.2f}"
        
        # Logique de couleur selon la valeur du score
        if val == 0.0:
            color = C_OFF              # Gris foncé (Inactif)
        elif val >= 0.40:
            color = C_OK               # Vert (Score fort / Seuil franchi)
        else:
            color = (200, 200, 200)    # Blanc cassé (Score faible/moyen)

        tw = cv2.getTextSize(text, FONT_MONO, 0.42, 1)[0][0]
        cursor_right -= tw
        elements_to_draw.append((text, cursor_right, color))
        cursor_right -= 12 # Espacement entre les colonnes

    # Ajout du label "SHOT" tout à gauche du bloc
    shot_label = "SHOT  "
    tw = cv2.getTextSize(shot_label, FONT_MONO, 0.42, 1)[0][0]
    cursor_right -= tw
    elements_to_draw.append((shot_label, cursor_right, (240, 240, 240)))

    # On dessine tout le bloc des scores
    for text, x_pos, color in elements_to_draw:
        cv2.putText(hud, text, (int(x_pos), int(y)), FONT_MONO, 0.42, color, 1, cv2.LINE_AA)

    return hud

# ===========================================================================
# 3. SIDEBAR DROITE (Minimap Horizontale + Futur Dashboard)
# ===========================================================================

def build_sidebar(sidebar_h: int, sidebar_w: int, state: MatchState) -> np.ndarray:
    """
    Génère la barre latérale droite.
    Haut : Minimap 2D horizontale.
    Bas : Espace réservé pour le futur Dashboard.
    """
    sidebar = np.full((sidebar_h, sidebar_w, 3), C_BG_DARK, dtype=np.uint8)

    # Marges
    mg_x = 20
    mg_y = 60  # Espace pour le titre en haut

    # --- Calcul de l'échelle (Horizontal) ---
    # On fitte la longueur du terrain (28m) dans la largeur de la sidebar
    sc = (sidebar_w - (mg_x * 2)) / COURT_L
    court_px_w = int(COURT_L * sc)
    court_px_h = int(COURT_W * sc)

    # Fond du terrain
    cv2.rectangle(sidebar,
                  (mg_x, mg_y),
                  (mg_x + court_px_w, mg_y + court_px_h),
                  C_COURT, -1)

    # --- En-tête : Titre + Statistiques ---
    title_y = 35
    title_txt = "MINIMAP & DASHBOARD"
    cv2.putText(sidebar, title_txt, (mg_x, title_y),
                FONT_MONO, 0.5, (240, 240, 245), 1, cv2.LINE_AA)
    
    stats_txt = (
        f"| PLYRS: {len(state.players)} "
        f"| AVG_SPD: {state.avg_speed_kmh:.1f} KM/H "
        f"| STD_DEV: {state.std_speed_kmh:.1f}"
    )
    tw_title = cv2.getTextSize(title_txt, FONT_MONO, 0.5, 1)[0][0]
    cv2.putText(sidebar, stats_txt, (mg_x + tw_title + 15, title_y),
                FONT_MONO, 0.42, C_WARN, 1, cv2.LINE_AA)

    lw = max(1, round(sc * 0.035))

    # --- PROJECTION HORIZONTALE ---
    def court_to_px(x_m, y_m):
        """
        L'axe X_m (28m) s'étend de gauche à droite.
        L'axe Y_m (15m) s'étend de haut en bas.
        """
        # On peut inverser Y si le terrain paraît "à l'envers" par rapport à la vidéo
        # flipped_y = COURT_W - y_m  <- Décommente si les paniers haut/bas sont inversés
        px = int(x_m * sc) + mg_x
        py = int(y_m * sc) + mg_y
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
    # AFFICHAGE DES JOUEURS
    # ==========================================
    dot_r = max(6, int(sc * 0.45))
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
        cv2.circle(sidebar, (px, py), dot_r, (255, 255, 255), 2, cv2.LINE_AA) 

    # --- SÉPARATEUR GAUCHE ET PLACEHOLDER DASHBOARD ---
    cv2.line(sidebar, (0, 0), (0, sidebar_h), (60, 60, 70), 2)

    # ==========================================
    # ZONE DASHBOARD TACTIQUE
    # ==========================================
    # Ligne pour délimiter visuellement la zone du dashboard
    dash_start_y = mg_y + court_px_h + 30
    cv2.line(sidebar, (0, dash_start_y), (sidebar_w, dash_start_y), (60, 60, 70), 2)
    
    # Coordonnées des colonnes (Élargies pour respirer)
    COL_LBL = mg_x
    COL_A   = mg_x + 220
    COL_B   = mg_x + 380
    COL_DIF = mg_x + 540
    
    y_curr = dash_start_y + 35
    
    # En-têtes de colonnes
    cv2.putText(sidebar, "METRIQUE", (COL_LBL, y_curr), FONT_MONO, 0.45, C_OFF, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "EQUIPE A", (COL_A, y_curr), FONT_MONO, 0.45, C_TEAM_A, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "EQUIPE B", (COL_B, y_curr), FONT_MONO, 0.45, C_TEAM_B, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "DIFF.",    (COL_DIF, y_curr), FONT_MONO, 0.45, C_WARN, 1, cv2.LINE_AA)
    
    cv2.line(sidebar, (mg_x, y_curr + 10), (sidebar_w - mg_x, y_curr + 10), (60, 60, 70), 1)
    
    y_curr += 35

    def draw_section(title: str, y: int) -> int:
        """Dessine un séparateur de section avec un fond gris."""
        cv2.rectangle(sidebar, (mg_x, y - 18), (sidebar_w - mg_x, y + 6), (25, 30, 35), -1)
        cv2.putText(sidebar, title, (mg_x + 10, y - 2), FONT_MONO, 0.45, (150, 150, 160), 1, cv2.LINE_AA)
        return y + 30

    def draw_row(label: str, val_a: float, val_b: float, y_pos: int, unit: str = "", is_int: bool = False) -> int:
        """Dessine une ligne de métrique et incrémente la position Y automatiquement."""
        # Formatage intelligent (sans décimale pour les compteurs de joueurs)
        fmt = "{:.0f}" if is_int else "{:.1f}"
        str_a = f"{fmt.format(val_a)}{unit}"
        str_b = f"{fmt.format(val_b)}{unit}"
        
        diff = val_a - val_b
        diff_txt = f"{'+' if diff > 0 else ''}{fmt.format(diff)}"
        
        cv2.putText(sidebar, label, (COL_LBL + 10, y_pos), FONT_MONO, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(sidebar, str_a, (COL_A, y_pos), FONT_MONO, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(sidebar, str_b, (COL_B, y_pos), FONT_MONO, 0.45, (255,255,255), 1, cv2.LINE_AA)
        
        # Le zéro parfait n'existe pas en float, on met en gris si c'est presque zéro
        diff_color = C_OFF if abs(diff) < 0.05 else C_WARN
        cv2.putText(sidebar, diff_txt, (COL_DIF, y_pos), FONT_MONO, 0.45, diff_color, 1, cv2.LINE_AA)

        return y_pos + 30 # Espace pour la prochaine ligne

    metrics_a = state.team_metrics[0]
    metrics_b = state.team_metrics[1]
    
    # --- SECTION 1 : CINÉMATIQUE ---
    y_curr = draw_section("CINEMATIQUE (Moyennes lissées)", y_curr)
    y_curr = draw_row("Vitesse Moy.", metrics_a["avg_speed"], metrics_b["avg_speed"], y_curr, " km/h")
    y_curr = draw_row("Ecart-Type V.", metrics_a["std_speed"], metrics_b["std_speed"], y_curr, " km/h")
    y_curr = draw_row("Accel. Moy.",  metrics_a["avg_accel"], metrics_b["avg_accel"], y_curr, " m/s2")
    y_curr = draw_row("Ecart-Type A.", metrics_a["std_accel"], metrics_b["std_accel"], y_curr, " m/s2")
    
    y_curr += 10 # Espace supplémentaire
    
    # --- SECTION 2 : TACTIQUE ---
    y_curr = draw_section("PLACEMENT & SPATIALISATION", y_curr)
    y_curr = draw_row("Spacing (Aire)", metrics_a["spacing"], metrics_b["spacing"], y_curr, " m2")
    y_curr = draw_row("Joueurs en Raquette", metrics_a["paint_count"], metrics_b["paint_count"], y_curr, "", is_int=True)

    y_curr += 15

    # --- 3. BLOC D'ALERTE ENCOMBREMENT ---
    total_paint = metrics_a["paint_count"] + metrics_b["paint_count"]
    
    # La boîte devient rouge vif si la raquette est surpeuplée
    alert_color = (60, 60, 200) if total_paint > 4 else (35, 40, 45)
    text_color = (255, 255, 255) if total_paint > 4 else (150, 150, 150)
    
    cv2.rectangle(sidebar, (mg_x, y_curr), (sidebar_w - mg_x, y_curr + 35), alert_color, -1)
    cv2.rectangle(sidebar, (mg_x, y_curr), (sidebar_w - mg_x, y_curr + 35), (60, 60, 70), 1) # Bordure

    alert_txt = f"ENCOMBREMENT TOTAL RAQUETTES : {int(total_paint)} JOUEURS"
    if total_paint > 4:
        alert_txt += "  [ ALERTE REBOND / POST-UP ]"
        
    cv2.putText(sidebar, alert_txt, (mg_x + 15, y_curr + 23), FONT_MONO, 0.45, text_color, 1, cv2.LINE_AA)

    return sidebar

# ===========================================================================
# 4. ASSEMBLAGE FINAL
# ===========================================================================

def render_debug_frame(frame: np.ndarray, state: MatchState, sidebar_w: int, margin_ratio: float = 1.50) -> np.ndarray:
    """
    Conforme à ton schéma : 
    Gauche [ HUD + Vidéo ] | Droite [ Sidebar complète ]
    """
    h, w = frame.shape[:2]
    HUD_H = 55

    # 1. Préparation de la vidéo annotée
    main = draw_zones_and_masks(frame, state, margin_ratio, mask_palyer=False, mask_net=True)
    main = draw_detections(main, state, txt_hoop=False, txt_player=False, show_team_crop=True)

    # 2. Création du HUD
    hud = build_top_hud(w, state)

    # 3. Assemblage Bloc Gauche (HUD + Vidéo)
    left_pane = np.vstack([hud, main])

    # 4. Création Bloc Droit (Sidebar). Sa hauteur doit matcher le left_pane.
    sidebar_h = h + HUD_H
    sidebar = build_sidebar(sidebar_h=sidebar_h, sidebar_w=sidebar_w, state=state)

    # 5. Assemblage final horizontal
    return np.hstack([left_pane, sidebar])