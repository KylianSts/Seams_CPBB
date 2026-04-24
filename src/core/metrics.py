"""
metrics.py
----------
Module d'analytique et de cinématique.
Calcule les vitesses, accélérations et statistiques globales des joueurs.
"""

import numpy as np
from core.state import MatchState

def compute_kinematics(state: MatchState, fps: float) -> None:
    """
    Calcule la vitesse de chaque joueur en km/h et met à jour 
    la moyenne et l'écart-type global du match.
    """
    speeds = []

    for player in state.players.values():
        if player.court_pos_m is None:
            continue

        # 1. Ajout de la position spatiale actuelle à l'historique
        player.pos_history_m.append(player.court_pos_m)

        # 2. Calcul de la vitesse lissée (besoin d'au moins 5 frames d'historique)
        if len(player.pos_history_m) >= 5:
            p_old = player.pos_history_m[0]  # Plus vieille position en mémoire
            p_new = player.pos_history_m[-1] # Position actuelle

            # Calcul de la distance euclidienne (en mètres)
            dist_m = np.sqrt((p_new[0] - p_old[0])**2 + (p_new[1] - p_old[1])**2)

            # Temps écoulé (en secondes) entre p_old et p_new
            dt = (len(player.pos_history_m) - 1) / fps

            # Vitesse : (m/s) convertie en km/h (* 3.6)
            if dt > 0:
                speed_ms = dist_m / dt
                player.speed_kmh = speed_ms * 3.6

        # On n'inclut dans les stats globales que les joueurs qui bougent un minimum
        # (Pour éviter que les remplaçants sur le banc ne fassent chuter la moyenne)
        if player.speed_kmh > 1.0:
            speeds.append(player.speed_kmh)

    # 3. Mise à jour des mathématiques globales (Moyenne & Écart-type)
    if speeds:
        state.avg_speed_kmh = float(np.mean(speeds))
        state.std_speed_kmh = float(np.std(speeds)) # Calcul de l'écart-type
    else:
        state.avg_speed_kmh = 0.0
        state.std_speed_kmh = 0.0