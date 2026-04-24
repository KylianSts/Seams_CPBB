"""
render.py
---------
Moteur de rendu graphique (Debug V0).
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
# 3. MINIMAP (Barre latérale droite)
# ===========================================================================

def build_sidebar(sidebar_h: int, sidebar_w: int, state: MatchState) -> np.ndarray:
    """
    Génère la barre latérale avec la minimap 2D complète du terrain FIBA.
    Prend toute la hauteur de l'écran, avec le compteur de joueurs en haut.
    """
    sidebar = np.full((sidebar_h, sidebar_w, 3), C_BG_DARK, dtype=np.uint8)

    # Marges et échelle
    mg_x, mg_y = 20, 80  # Marge supérieure agrandie pour laisser respirer l'en-tête
    sc = (sidebar_h - mg_y * 1.5) / COURT_L  # Ajustement de l'échelle à la nouvelle hauteur totale
    court_px_w = int(COURT_W * sc)
    court_px_h = int(COURT_L * sc)

    # Fond du terrain
    cv2.rectangle(sidebar,
                  (mg_x, mg_y),
                  (mg_x + court_px_w, mg_y + court_px_h),
                  C_COURT, -1)

    # --- En-tête : Titre + Statistiques Globales ---
    title_y = 40
    title_txt = "DASHBOARD TACTIQUE"
    cv2.putText(sidebar, title_txt, (mg_x, title_y),
                FONT_MONO, 0.5, (240, 240, 245), 1, cv2.LINE_AA)
    
    # Formatage des statistiques avec 1 chiffre après la virgule
    stats_txt = (
        f"| PLYRS: {len(state.players)} "
        f"| AVG_SPD: {state.avg_speed_kmh:.1f} KM/H "
        f"| STD_DEV: {state.std_speed_kmh:.1f}"
    )
    
    tw_title = cv2.getTextSize(title_txt, FONT_MONO, 0.5, 1)[0][0]
    
    # Affichage du bloc de statistiques à côté du titre
    cv2.putText(sidebar, stats_txt, (mg_x + tw_title + 15, title_y),
                FONT_MONO, 0.42, C_WARN, 1, cv2.LINE_AA)

    # Épaisseur des lignes
    lw = max(1, round(sc * 0.035))

    # --- SYMÉTRIE HORIZONTALE ---
    def court_to_px(x_m, y_m):
        """Convertit (X_mètres, Y_mètres) en pixels avec Flip Horizontal."""
        # On inverse l'axe Y (15m) pour corriger l'effet miroir de la caméra
        flipped_y = COURT_W - y_m 
        
        px = int(flipped_y * sc) + mg_x
        py = int(x_m * sc) + mg_y
        return px, py

    def pline(pts, closed=False):
        """Dessine une ligne brisée à partir de coordonnées en mètres."""
        pts_px = np.array([court_to_px(p[0], p[1]) for p in pts], dtype=np.int32)
        cv2.polylines(sidebar, [pts_px.reshape(-1, 1, 2)], closed, C_LINE, lw, cv2.LINE_AA)

    # ==========================================
    # DESSIN DU TERRAIN FIBA (Mesures officielles)
    # ==========================================
    
    # 1. Lignes de touche (Contour)
    pline([[0, 0], [COURT_L, 0], [COURT_L, COURT_W], [0, COURT_W]], closed=True)
    
    # 2. Ligne médiane
    pline([[14.0, 0.0], [14.0, COURT_W]])

    # 3. Cercle central (Rayon 1.8m)
    cx, cy = court_to_px(14.0, 7.5)
    cv2.circle(sidebar, (cx, cy), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)

    # 4. Raquettes / Key (Rectangle 5.8m x 4.9m)
    # Zone gauche
    pline([[0, 5.05], [5.8, 5.05], [5.8, 9.95], [0, 9.95]], closed=True)
    # Zone droite
    pline([[28.0, 5.05], [22.2, 5.05], [22.2, 9.95], [28.0, 9.95]], closed=True)

    # 5. Cercles des lancers francs (Rayon 1.8m)
    cx_lf_l, cy_lf_l = court_to_px(5.8, 7.5)
    cv2.circle(sidebar, (cx_lf_l, cy_lf_l), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)
    cx_lf_r, cy_lf_r = court_to_px(22.2, 7.5)
    cv2.circle(sidebar, (cx_lf_r, cy_lf_r), int(1.8 * sc), C_LINE, lw, cv2.LINE_AA)

    # 6. Ligne à 3 points (Arc de cercle + Lignes droites)
    arc_angles = np.linspace(-1.36, 1.36, 25)
    arc_left = [[1.575 + 6.75 * np.cos(a), 7.5 + 6.75 * np.sin(a)] for a in arc_angles]
    line_3pt_left = [[0.0, 0.9], [2.99, 0.9]] + arc_left + [[2.99, 14.1], [0.0, 14.1]]
    pline(line_3pt_left, closed=False)

    # Symétrie mathématique pour la ligne à 3 points droite
    line_3pt_right = [[28.0 - p[0], p[1]] for p in line_3pt_left]
    pline(line_3pt_right, closed=False)

    # 7. Paniers et Planches
    pline([[1.2, 6.9], [1.2, 8.1]]) # Planche gauche
    pline([[26.8, 6.9], [26.8, 8.1]]) # Planche droite
    cx_b_l, cy_b_l = court_to_px(1.575, 7.5)
    cv2.circle(sidebar, (cx_b_l, cy_b_l), max(2, int(0.225 * sc)), C_LINE, lw, cv2.LINE_AA)
    cx_b_r, cy_b_r = court_to_px(26.425, 7.5)
    cv2.circle(sidebar, (cx_b_r, cy_b_r), max(2, int(0.225 * sc)), C_LINE, lw, cv2.LINE_AA)

    # ==========================================
    # AFFICHAGE DES JOUEURS
    # ==========================================
    dot_r = max(10, int(sc * 0.50))
    for track_id, player in state.players.items():
        if player.court_pos_m is None:
            continue
        X_m, Y_m = player.court_pos_m
        
        # Sécurité pour garder les points à l'intérieur de l'affichage
        X_m = max(0.0, min(COURT_L, X_m))
        Y_m = max(0.0, min(COURT_W, Y_m))
        
        px, py = court_to_px(X_m, Y_m)

        if player.team_id == 0:
            color = C_TEAM_A   # Rouge
        elif player.team_id == 1:
            color = C_TEAM_B # Cyan
        else:
            color = C_NO_TEAM # Gris (Calibration)

        # Dessin du joueur
        cv2.circle(sidebar, (px, py), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(sidebar, (px, py), dot_r, (255, 255, 255), 2, cv2.LINE_AA) # Bordure blanche

    # Séparateur gauche
    cv2.line(sidebar, (0, 0), (0, sidebar_h), (60, 60, 70), 1)

    return sidebar


# ===========================================================================
# ASSEMBLAGE FINAL
# ===========================================================================

def render_debug_frame(frame: np.ndarray, state: MatchState, sidebar_w: int = 550, margin_ratio: float = 1.50) -> np.ndarray:
    """
    - Sidebar prend toute la hauteur.
    - HUD prend la largeur de la vidéo uniquement.
    """
    h, w = frame.shape[:2]
    HUD_H = 55

    # 1. Préparation de la vidéo annotée
    main = draw_zones_and_masks(frame, state, margin_ratio, mask_palyer=False, mask_net=True)
    # On utilise votre nouvelle version de draw_detections
    main = draw_detections(main, state, txt_hoop=False, txt_player=False, show_team_crop=True)

    # 2. Création du HUD (Largeur = w, celle de la vidéo)
    hud = build_top_hud(w, state)

    # 3. Assemblage de la colonne de gauche (HUD au dessus de la vidéo)
    left_pane = np.vstack([hud, main])

    # 4. Création de la Sidebar (Hauteur = h + HUD_H, pour correspondre au left_pane)
    sidebar = build_sidebar(sidebar_h=h + HUD_H, sidebar_w=sidebar_w, state=state)

    # 5. Assemblage final horizontal
    return np.hstack([left_pane, sidebar])