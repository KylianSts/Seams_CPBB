"""
metrics.py
----------
Module d'analytique spatiale et de cinématique.
Calcule les vitesses, accélérations, métriques collectives (spacing)
et déduit les phases tactiques (équipe en attaque, joueurs ouverts).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from core.state import MatchState, PlayerState


@dataclass
class MetricsConfig:
    """
    Configuration centralisée des règles physiques et tactiques du basketball.
    Remplace les variables codées en dur pour faciliter le fine-tuning.
    """
    # --- Cinématique ---
    window_size: int = 30          # Taille de la fenêtre glissante pour lisser la vitesse
    max_speed_kmh: float = 35.0    # Plafond physique (Physic Gate) pour ignorer le bruit de tracking
    min_speed_threshold: float = 0.2 # Vitesse minimale pour être inclus dans la moyenne (évite le biais des joueurs statiques)

    # --- Tactique ---
    iso_threshold_m: float = 2.5   # Distance minimum du défenseur pour qu'un joueur soit considéré "isolé"
    threat_range_m: float = 8.5    # Rayon d'action autour du panier pour être considéré comme une "menace"

    # --- Géométrie FIBA (en mètres) ---
    hoop_left: Tuple[float, float] = (1.575, 7.5)
    hoop_right: Tuple[float, float] = (26.425, 7.5)

    # Zones de la raquette [x_min, y_min, x_max, y_max]
    paint_left: Tuple[float, float, float, float] = (0.0, 5.05, 5.8, 9.95)
    paint_right: Tuple[float, float, float, float] = (22.2, 5.05, 28.0, 9.95)


# ===========================================================================
# UTILITAIRES GÉOMÉTRIQUES
# ===========================================================================

def is_in_paint(pos_m: Tuple[float, float], cfg: MetricsConfig) -> bool:
    """Vérifie si une coordonnée (X, Y) se trouve dans l'une des deux raquettes."""
    x, y = pos_m
    
    in_left = (cfg.paint_left[0] <= x <= cfg.paint_left[2]) and \
              (cfg.paint_left[1] <= y <= cfg.paint_left[3])
              
    in_right = (cfg.paint_right[0] <= x <= cfg.paint_right[2]) and \
               (cfg.paint_right[1] <= y <= cfg.paint_right[3])
               
    return in_left or in_right


def calculate_spacing(points: List[Tuple[float, float]]) -> float:
    """
    Calcule l'empreinte spatiale d'une équipe (en m²).
    Utilise l'enveloppe convexe (Convex Hull) formée par les positions des joueurs.
    """
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume) # En 2D, l'attribut 'volume' de SciPy renvoie l'aire polygonale
    except Exception:
        # Contournement de sécurité si les points sont colinéaires (ex: alignement parfait)
        return 0.0


# ===========================================================================
# ANALYSE TACTIQUE
# ===========================================================================

def detect_attacking_team(
    team_players: Dict[int, List[PlayerState]], 
    cfg: MetricsConfig
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """
    Déduit l'équipe en attaque par analyse du centre de gravité et de la "Pression Relative".
    L'équipe en défense est statistiquement celle qui fait barrage (plus proche de son propre panier).
    """
    if not team_players[0] or not team_players[1]:
        return None, None

    # 1. Barycentre global pour définir le demi-terrain actif
    all_x = [p.court_pos_m[0] for p in team_players[0] + team_players[1]]
    global_center_x = sum(all_x) / len(all_x)
    
    active_hoop = cfg.hoop_left if global_center_x < 14.0 else cfg.hoop_right
    
    # 2. Distance moyenne de chaque équipe au panier actif
    avg_dist = {}
    for tid in [0, 1]:
        dists = [math.hypot(p.court_pos_m[0] - active_hoop[0], p.court_pos_m[1] - active_hoop[1]) 
                 for p in team_players[tid]]
        avg_dist[tid] = sum(dists) / len(dists)
        
    # L'équipe avec la distance moyenne la plus faible protège l'arceau (Défense).
    defending_team = 0 if avg_dist[0] < avg_dist[1] else 1
    attacking_team = 1 - defending_team
    
    return attacking_team, active_hoop


def evaluate_open_players(
    team_players: Dict[int, List[PlayerState]], 
    attacking_team: int, 
    target_hoop: Tuple[float, float], 
    cfg: MetricsConfig
) -> None:
    """
    Marque dynamiquement les attaquants comme "ouverts" (démarqués) 
    s'ils représentent une menace directe avec suffisamment d'espace.
    """
    for player in team_players[attacking_team]:
        player.is_open = False # État par défaut
        
        if player.court_pos_m is None or player.closest_defender_dist_m is None:
            continue
            
        dist_to_hoop = math.hypot(player.court_pos_m[0] - target_hoop[0], 
                                  player.court_pos_m[1] - target_hoop[1])
        
        is_isolated = player.closest_defender_dist_m >= cfg.iso_threshold_m
        is_a_threat = dist_to_hoop <= cfg.threat_range_m
        
        if is_isolated and is_a_threat:
            player.is_open = True


# ===========================================================================
# MOTEUR PRINCIPAL (CINÉMATIQUE & AGRÉGATION)
# ===========================================================================

def compute_kinematics(state: MatchState, fps: float, cfg: MetricsConfig = MetricsConfig()) -> None:
    """
    Moteur principal d'analytique.
    Traite la cinématique (Vitesse/Accélération), la spatialisation (Spacing/Raquette),
    et dérive la tactique globale en une seule passe optimisée sur l'état des joueurs.
    """
    dt_frame = 1.0 / fps
    
    # Structures intermédiaires pour éviter les boucles multiples sur state.players
    team_players = {0: [], 1: []}
    team_data = {
        0: {"speeds": [], "accels": [], "positions": [], "in_paint": 0},
        1: {"speeds": [], "accels": [], "positions": [], "in_paint": 0}
    }

    # ==========================================
    # PASS 1 : Cinématique individuelle & Tri
    # ==========================================
    for player in state.players.values():
        if player.court_pos_m is None:
            continue

        player.pos_history_m.append(player.court_pos_m)

        # Calcul des dérivées (Vitesse & Accélération) sur fenêtre glissante
        if len(player.pos_history_m) >= cfg.window_size:
            p_old = player.pos_history_m[-cfg.window_size]
            p_new = player.pos_history_m[-1]
            
            dist_m = math.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
            dt_window = (cfg.window_size - 1) * dt_frame
            
            new_speed_ms = dist_m / dt_window
            new_speed_kmh = new_speed_ms * 3.6

            # Filtrage des aberrations de tracking (Téléportation)
            if new_speed_kmh > cfg.max_speed_kmh:
                player.accel_ms2 = 0.0
            else:
                old_speed_ms = (player.speed_kmh / 3.6)
                player.accel_ms2 = (new_speed_ms - old_speed_ms) / dt_frame
                player.speed_kmh = new_speed_kmh

        # Ventilation par équipe pour les passes suivantes
        if player.team_id in [0, 1]:
            tid = player.team_id
            team_players[tid].append(player)
            team_data[tid]["positions"].append(player.court_pos_m)
            
            if is_in_paint(player.court_pos_m, cfg):
                team_data[tid]["in_paint"] += 1

            if player.speed_kmh > cfg.min_speed_threshold:
                team_data[tid]["speeds"].append(player.speed_kmh)
                team_data[tid]["accels"].append(player.accel_ms2)

    # ==========================================
    # PASS 2 : Évaluation des distances inter-joueurs (Défenseur le plus proche)
    # ==========================================
    for tid in [0, 1]:
        adv_tid = 1 - tid
        for p1 in team_players[tid]:
            min_dist = float('inf')
            for p2 in team_players[adv_tid]:
                dist = math.hypot(p1.court_pos_m[0] - p2.court_pos_m[0], 
                                  p1.court_pos_m[1] - p2.court_pos_m[1])
                if dist < min_dist:
                    min_dist = dist
            
            p1.closest_defender_dist_m = float(min_dist) if min_dist != float('inf') else None

    # ==========================================
    # PASS 3 : Agrégation des métriques (Équipes & Global)
    # ==========================================
    all_speeds = []
    all_accels = []

    for tid in [0, 1]:
        d = team_data[tid]
        metrics = state.team_metrics[tid]
        
        metrics["spacing"] = calculate_spacing(d["positions"])
        metrics["paint_count"] = float(d["in_paint"])
        
        if d["speeds"]:
            metrics["avg_speed"] = float(np.mean(d["speeds"]))
            metrics["std_speed"] = float(np.std(d["speeds"]))
            metrics["min_speed"] = float(np.min(d["speeds"]))
            metrics["max_speed"] = float(np.max(d["speeds"]))
            all_speeds.extend(d["speeds"])
        else:
            metrics["avg_speed"] = metrics["std_speed"] = 0.0
            metrics["min_speed"] = metrics["max_speed"] = 0.0

        if d["accels"]:
            metrics["avg_accel"] = float(np.mean(d["accels"]))
            metrics["std_accel"] = float(np.std(d["accels"]))
            metrics["min_accel"] = float(np.min(d["accels"]))
            metrics["max_accel"] = float(np.max(d["accels"]))
            all_accels.extend(d["accels"])
        else:
            metrics["avg_accel"] = metrics["std_accel"] = 0.0
            metrics["min_accel"] = metrics["max_accel"] = 0.0

    # Consolidation à l'échelle du match
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
    # PASS 4 : Analyse Tactique de Haut Niveau
    # ==========================================
    attacking_team, target_hoop = detect_attacking_team(team_players, cfg)
    
    state.attacking_team_id = attacking_team
    state.target_hoop = target_hoop
    
    if attacking_team is not None and target_hoop is not None:
        evaluate_open_players(team_players, attacking_team, target_hoop, cfg)