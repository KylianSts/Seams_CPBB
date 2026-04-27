"""
metrics.py
----------
Module d'analytique et de cinématique.
Calcule les vitesses, accélérations et statistiques globales des joueurs.
"""

import numpy as np
from core.state import MatchState
from scipy.spatial import ConvexHull

def compute_kinematics(state: MatchState, fps: float) -> None:
    """
    Calcule la vitesse (km/h) et l'accélération (m/s²) par joueur et par équipe.
    Intègre un lissage long (TV) et une 'Physic Gate' pour filtrer les bugs de projection.
    """
    dt_frame = 1.0 / fps
    team_data = {0: {"speeds": [], "accels": []}, 1: {"speeds": [], "accels": []}}

    # Paramètres de lissage et de sécurité
    WINDOW_SIZE = 25       # 1 seconde complète à 25fps (lissage fort pour la TV)
    MAX_SPEED_KMH = 35.0   # Physic Gate : limite humaine (Usain Bolt)

    for player in state.players.values():
        if player.court_pos_m is None:
            continue

        # 1. Historique de position
        player.pos_history_m.append(player.court_pos_m)

        # 2. Calcul Vitesse (Fenêtre longue pour lisser)
        if len(player.pos_history_m) >= WINDOW_SIZE:
            p_old = player.pos_history_m[-WINDOW_SIZE]
            p_new = player.pos_history_m[-1]
            dist_m = np.sqrt((p_new[0] - p_old[0])**2 + (p_new[1] - p_old[1])**2)
            
            # dt pour (WINDOW_SIZE - 1) intervalles
            dt_window = (WINDOW_SIZE - 1) * dt_frame
            new_speed_ms = dist_m / dt_window
            new_speed_kmh = new_speed_ms * 3.6

            # --- 3. LA PHYSIC GATE ---
            if new_speed_kmh > MAX_SPEED_KMH:
                # Calcul physiquement impossible (probablement un saut d'homographie)
                # On annule l'accélération et on GARDE l'ancienne vitesse
                player.accel_ms2 = 0.0
                # player.speed_kmh reste intact
            else:
                # Comportement normal
                old_speed_ms = (player.speed_kmh / 3.6)
                # On calcule l'accélération sur la même fenêtre pour une transition douce
                player.accel_ms2 = (new_speed_ms - old_speed_ms) / dt_window
                player.speed_kmh = new_speed_kmh

        # 4. Groupement par équipe (seulement si le joueur bouge > 1km/h pour éviter le bruit statique)
        if player.team_id in [0, 1] and player.speed_kmh > 1.0:
            team_data[player.team_id]["speeds"].append(player.speed_kmh)
            team_data[player.team_id]["accels"].append(player.accel_ms2)

    # 5. Calcul des statistiques par équipe
    all_speeds = []
    for tid in [0, 1]:
        s_list = team_data[tid]["speeds"]
        a_list = team_data[tid]["accels"]
        
        if s_list:
            state.team_metrics[tid]["avg_speed"] = float(np.mean(s_list))
            state.team_metrics[tid]["std_speed"] = float(np.std(s_list))
            all_speeds.extend(s_list)
        if a_list:
            state.team_metrics[tid]["avg_accel"] = float(np.mean(a_list))
            state.team_metrics[tid]["std_accel"] = float(np.std(a_list))

    # 6. Global & Différences
    if all_speeds:
        state.avg_speed_kmh = float(np.mean(all_speeds))
        state.std_speed_kmh = float(np.std(all_speeds))

def is_in_paint(pos_m: tuple) -> bool:
    """Vérifie si une position en mètres est dans l'une des deux raquettes FIBA."""
    x, y = pos_m
    # Raquette Gauche : 0 à 5.8m en X | 5.05 à 9.95m en Y (Largeur 4.9m centrée sur 7.5m)
    in_left = (0.0 <= x <= 5.8) and (5.05 <= y <= 9.95)
    # Raquette Droite : 22.2 à 28m en X
    in_right = (22.2 <= x <= 28.0) and (5.05 <= y <= 9.95)
    return in_left or in_right

def calculate_spacing(points: list) -> float:
    """Calcule l'aire du polygone convexe formé par les joueurs (m²)."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume) # Dans ConvexHull 2D, 'volume' est l'aire
    except Exception:
        return 0.0