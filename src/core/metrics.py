"""
metrics.py
----------
Module d'analytique et de cinématique.
Calcule les vitesses, accélérations et statistiques globales des joueurs.
"""

import math
import numpy as np
from scipy.spatial import ConvexHull
from core.state import MatchState

def is_in_paint(pos_m: tuple) -> bool:
    """Vérifie si une position en mètres est dans l'une des deux raquettes FIBA."""
    x, y = pos_m
    in_left = (0.0 <= x <= 5.8) and (5.05 <= y <= 9.95)
    in_right = (22.2 <= x <= 28.0) and (5.05 <= y <= 9.95)
    return in_left or in_right

def calculate_spacing(points: list) -> float:
    """Calcule l'aire du polygone convexe formé par les joueurs (m²)."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def compute_kinematics(state: MatchState, fps: float) -> None:
    """
    Calcule la vitesse, l'accélération, les données spatiales (Spacing, Raquette)
    et la proximité du défenseur le plus proche pour chaque joueur.
    """
    dt_frame = 1.0 / fps
    team_data = {
        0: {"speeds": [], "accels": [], "positions": [], "in_paint": 0},
        1: {"speeds": [], "accels": [], "positions": [], "in_paint": 0}
    }

    WINDOW_SIZE = 25       
    MAX_SPEED_KMH = 35.0   

    # ==========================================
    # 1. MISE À JOUR CINÉMATIQUE (Par Joueur)
    # ==========================================
    for player in state.players.values():
        if player.court_pos_m is None:
            continue

        player.pos_history_m.append(player.court_pos_m)

        if len(player.pos_history_m) >= WINDOW_SIZE:
            p_old = player.pos_history_m[-WINDOW_SIZE]
            p_new = player.pos_history_m[-1]
            
            # math.hypot est plus rapide et propre que sqrt(a^2 + b^2)
            dist_m = math.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
            
            dt_window = (WINDOW_SIZE - 1) * dt_frame
            new_speed_ms = dist_m / dt_window
            new_speed_kmh = new_speed_ms * 3.6

            # Physic Gate
            if new_speed_kmh > MAX_SPEED_KMH:
                player.accel_ms2 = 0.0
            else:
                old_speed_ms = (player.speed_kmh / 3.6)
                player.accel_ms2 = (new_speed_ms - old_speed_ms) / dt_window
                player.speed_kmh = new_speed_kmh

        if player.team_id in [0, 1]:
            team_data[player.team_id]["positions"].append(player.court_pos_m)
            if is_in_paint(player.court_pos_m):
                team_data[player.team_id]["in_paint"] += 1

            # On ne compte que les joueurs en mouvement pour les stats
            if player.speed_kmh > 1.0:
                team_data[player.team_id]["speeds"].append(player.speed_kmh)
                team_data[player.team_id]["accels"].append(player.accel_ms2)


    # ==========================================
    # 2. PROXIMITÉ DU DÉFENSEUR (Pairwise Distance)
    # ==========================================
    # On groupe les joueurs actifs par équipe
    team_players = {0: [], 1: []}
    for player in state.players.values():
        if player.court_pos_m is not None and player.team_id in [0, 1]:
            team_players[player.team_id].append(player)

    # Calcul O(N*M) - très rapide car maximum 5v5
    for tid in [0, 1]:
        adv_tid = 1 - tid # Équipe adverse
        for p1 in team_players[tid]:
            min_dist = float('inf')
            for p2 in team_players[adv_tid]:
                dist = math.hypot(p1.court_pos_m[0] - p2.court_pos_m[0], p1.court_pos_m[1] - p2.court_pos_m[1])
                if dist < min_dist:
                    min_dist = dist
            
            # Enregistrement dans l'état du joueur
            p1.closest_defender_dist_m = float(min_dist) if min_dist != float('inf') else None


    # ==========================================
    # 3. AGRÉGATION PAR ÉQUIPE
    # ==========================================
    all_speeds = []
    all_accels = []

    for tid in [0, 1]:
        d = team_data[tid]
        state.team_metrics[tid]["spacing"] = calculate_spacing(d["positions"])
        state.team_metrics[tid]["paint_count"] = float(d["in_paint"])
        
        # Vitesses
        if d["speeds"]:
            state.team_metrics[tid]["avg_speed"] = float(np.mean(d["speeds"]))
            state.team_metrics[tid]["std_speed"] = float(np.std(d["speeds"]))
            state.team_metrics[tid]["min_speed"] = float(np.min(d["speeds"]))
            state.team_metrics[tid]["max_speed"] = float(np.max(d["speeds"]))
            all_speeds.extend(d["speeds"])
        else:
            state.team_metrics[tid]["avg_speed"] = state.team_metrics[tid]["std_speed"] = 0.0
            state.team_metrics[tid]["min_speed"] = state.team_metrics[tid]["max_speed"] = 0.0

        # Accélérations
        if d["accels"]:
            state.team_metrics[tid]["avg_accel"] = float(np.mean(d["accels"]))
            state.team_metrics[tid]["std_accel"] = float(np.std(d["accels"]))
            state.team_metrics[tid]["min_accel"] = float(np.min(d["accels"]))
            state.team_metrics[tid]["max_accel"] = float(np.max(d["accels"]))
            all_accels.extend(d["accels"])
        else:
            state.team_metrics[tid]["avg_accel"] = state.team_metrics[tid]["std_accel"] = 0.0
            state.team_metrics[tid]["min_accel"] = state.team_metrics[tid]["max_accel"] = 0.0


    # ==========================================
    # 4. AGRÉGATION GLOBALE (Tous les joueurs)
    # ==========================================
    if all_speeds:
        state.avg_speed_kmh = float(np.mean(all_speeds))
        state.std_speed_kmh = float(np.std(all_speeds))
        state.min_speed_kmh = float(np.min(all_speeds))
        state.max_speed_kmh = float(np.max(all_speeds))
    else:
        state.avg_speed_kmh = state.std_speed_kmh = 0.0
        state.min_speed_kmh = state.max_speed_kmh = 0.0

    if all_accels:
        state.avg_accel_ms2 = float(np.mean(all_accels))
        state.std_accel_ms2 = float(np.std(all_accels))
        state.min_accel_ms2 = float(np.min(all_accels))
        state.max_accel_ms2 = float(np.max(all_accels))
    else:
        state.avg_accel_ms2 = state.std_accel_ms2 = 0.0
        state.min_accel_ms2 = state.max_accel_ms2 = 0.0
    
    # ==========================================
    # 5. TACTIQUE : ATTAQUE ET JOUEURS OUVERTS
    # ==========================================
    attacking_team, target_hoop = detect_attacking_team(state)
    state.attacking_team_id = attacking_team
    
    if attacking_team is not None and target_hoop is not None:
        evaluate_open_players(state, attacking_team, target_hoop)


import math
import numpy as np

# Constantes des paniers FIBA (X, Y)
HOOP_LEFT = (1.575, 7.5)
HOOP_RIGHT = (26.425, 7.5)

def detect_attacking_team(state) -> tuple[int, tuple]:
    """
    Déduit l'équipe en attaque et le panier ciblé via la 'Pression Relative'.
    Retourne (team_id_attaque, coordonnees_panier_attaque) ou (None, None).
    """
    team_players = {0: [], 1: []}
    for p in state.players.values():
        if p.court_pos_m is not None and p.team_id in [0, 1]:
            team_players[p.team_id].append(p)

    if len(team_players[0]) == 0 or len(team_players[1]) == 0:
        return None, None

    # 1. Calcul du centre de gravité (Barycentre X) de tous les joueurs pour savoir où se joue l'action
    all_x = [p.court_pos_m[0] for p in team_players[0] + team_players[1]]
    global_center_x = sum(all_x) / len(all_x)
    
    # 2. Déduction du panier actif (demi-terrain où se trouvent les joueurs)
    active_hoop = HOOP_LEFT if global_center_x < 14.0 else HOOP_RIGHT
    
    # 3. Calcul de la distance moyenne de chaque équipe vers le panier ACTIF
    avg_dist = {}
    for tid in [0, 1]:
        dists = [math.hypot(p.court_pos_m[0] - active_hoop[0], p.court_pos_m[1] - active_hoop[1]) 
                 for p in team_players[tid]]
        avg_dist[tid] = sum(dists) / len(dists)
        
    # 4. L'équipe en DÉFENSE est celle qui est en moyenne la plus proche de l'arceau (elle fait barrage).
    # L'équipe en ATTAQUE est donc l'autre.
    defending_team = 0 if avg_dist[0] < avg_dist[1] else 1
    attacking_team = 1 - defending_team
    
    return attacking_team, active_hoop


def evaluate_open_players(state, attacking_team: int, target_hoop: tuple, 
                          iso_threshold_m: float = 1.8, threat_range_m: float = 8.5) -> None:
    """
    Marque les joueurs comme 'ouverts' s'ils sont isolés ET dans la zone de menace.
    """
    for player in state.players.values():
        player.is_open = False # Reset par défaut
        
        # On n'évalue que les joueurs de l'équipe en attaque
        if player.team_id != attacking_team or player.court_pos_m is None or player.closest_defender_dist_m is None:
            continue
            
        # 1. Distance au panier attaqué
        dist_to_hoop = math.hypot(player.court_pos_m[0] - target_hoop[0], player.court_pos_m[1] - target_hoop[1])
        
        # 2. Validation : Isolé + Menace
        is_isolated = player.closest_defender_dist_m >= iso_threshold_m
        is_a_threat = dist_to_hoop <= threat_range_m
        
        if is_isolated and is_a_threat:
            player.is_open = True