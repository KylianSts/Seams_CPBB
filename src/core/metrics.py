"""
metrics.py
----------
Module d'analytique et de cinématique.
Calcule les vitesses, accélérations et statistiques globales des joueurs.
"""

import numpy as np
from core.state import MatchState

import numpy as np
from core.state import MatchState

def compute_kinematics(state: MatchState, fps: float) -> None:
    """
    Calcule la vitesse (km/h) et l'accélération (m/s²) par joueur et par équipe.
    """
    dt_frame = 1.0 / fps
    team_data = {0: {"speeds": [], "accels": []}, 1: {"speeds": [], "accels": []}}

    for player in state.players.values():
        if player.court_pos_m is None:
            continue

        # 1. Historique de position
        player.pos_history_m.append(player.court_pos_m)

        # 2. Calcul Vitesse (Fenêtre de 5 frames pour lisser)
        if len(player.pos_history_m) >= 5:
            p_old = player.pos_history_m[-5]
            p_new = player.pos_history_m[-1]
            dist_m = np.sqrt((p_new[0] - p_old[0])**2 + (p_new[1] - p_old[1])**2)
            
            # dt pour 4 intervalles entre 5 points
            dt_window = 4 * dt_frame
            new_speed_ms = dist_m / dt_window
            new_speed_kmh = new_speed_ms * 3.6

            # 3. Calcul Accélération (Variation entre l'ancienne vitesse et la nouvelle)
            # On compare la vitesse actuelle à celle de la frame précédente
            old_speed_ms = (player.speed_kmh / 3.6)
            player.accel_ms2 = (new_speed_ms - old_speed_ms) / dt_frame
            player.speed_kmh = new_speed_kmh

        # 4. Groupement par équipe (seulement si le joueur bouge > 1km/h pour éviter le bruit)
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