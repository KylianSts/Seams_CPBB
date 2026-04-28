"""
detect_team.py
--------------
Module de clustering et de classification des équipes.

Utilise GMM (Gaussian Mixture Models) pour gérer les variances d'éclairage.
Intègre un `calibration_stride` pour échantillonner les frames et capturer plus de diversité.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from core.state import MatchState

logger = logging.getLogger(__name__)


class TeamDetector:
    def __init__(self, calibration_frames: int = 100, history_size: int = 30):
        """
        Args:
            calibration_frames: Nombre d'échantillons cibles pour le GMM.
            history_size: Nombre de frames mémorisées par joueur pour le vote.
        """
        self.calibration_frames = calibration_frames
        self.history_size = history_size
        
        self.is_calibrated = False
        self.frames_collected = 0
        self.histograms_buffer = [] 
        
        self.gmm: Optional[GaussianMixture] = None
        self.player_votes: Dict[int, deque] = {}


    def _get_torso_histogram(self, frame: np.ndarray, bbox_px: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Extrait l'histogramme HSV du centre du torse."""
        x1, y1, x2, y2 = map(int, bbox_px)
        h_img, w_img = frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 20:
            return None

        # CORE CROP : On ignore tête, jambes, et les bords latéraux (bras/occlusions)
        crop_y1 = y1 + int(h * 0.20)
        crop_y2 = y2 - int(h * 0.40)
        crop_x1 = x1 + int(w * 0.25)
        crop_x2 = x2 - int(w * 0.25)
        
        torso_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if torso_bgr.size == 0:
            return None

        torso_hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(torso_hsv, (0, 20, 20), (180, 255, 240))

        hist = cv2.calcHist([torso_hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist.flatten()


    def collect_from_raw_boxes(self, frame: np.ndarray, isolated_boxes: List[Tuple]) -> None:
        """Extrait les histogrammes à partir d'une liste de joueurs préalablement filtrés (isolés)."""
        if self.is_calibrated: return
            
        for box in isolated_boxes:
            hist = self._get_torso_histogram(frame, box[:4])
            if hist is not None:
                self.histograms_buffer.append(hist)
                
        self.frames_collected += 1


    def _run_calibration(self):
        """Lance l'algorithme GMM sur les données accumulées."""
        if len(self.histograms_buffer) < 10:
            logger.warning("Pas assez de données pour calibrer les équipes.")
            return

        data = np.array(self.histograms_buffer, dtype=np.float32)
        
        # GMM permet des clusters de tailles et de formes différentes
        self.gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        self.gmm.fit(data)
        
        self.is_calibrated = True
        logger.info(f"Calibration GMM terminée ({len(data)} échantillons sur {self.frames_collected} frames).")


    def update(self, state: MatchState, frame: np.ndarray) -> None:
        """Inférence avec Soft Voting Quadratique, Seuil de Rejet et Rééquilibrage."""
        if not self.is_calibrated:
            return 

        active_ids = set(state.players.keys())
        self.player_votes = {k: v for k, v in self.player_votes.items() if k in active_ids}

        # 1. Calcul des votes pour chaque joueur
        for track_id, player in state.players.items():
            hist = self._get_torso_histogram(frame, player.bbox_px)
            
            if hist is not None:
                # Probabilités brutes du GMM (ex: [0.80, 0.20])
                probs = self.gmm.predict_proba([hist])[0] 
                
                # SEUIL DE REJET : Si le GMM est trop incertain, on ignore l'image
                if np.max(probs) < 0.65:
                    pass # On ne vote pas, on laisse l'historique faire son travail
                else:
                    # AMPLIFICATION (Score Quadratique)
                    squared_probs = probs ** 2
                    
                    # Normalisation pour que la somme fasse toujours 1.0 (100%)
                    weighted_probs = squared_probs / np.sum(squared_probs)
                    
                    if track_id not in self.player_votes:
                        self.player_votes[track_id] = deque(maxlen=self.history_size)
                    self.player_votes[track_id].append(weighted_probs)

            # DÉCISION INITIALE PAR MOYENNE LISSÉE
            if track_id in self.player_votes and len(self.player_votes[track_id]) > 0:
                all_probs = np.array(list(self.player_votes[track_id]))
                mean_probs = np.mean(all_probs, axis=0)
                
                player.team_id = int(np.argmax(mean_probs))


        # ==========================================
        # 2. RÈGLE MÉTIER : MAX 5 JOUEURS PAR ÉQUIPE
        # ==========================================
        team_0_players = [p for p in state.players.values() if p.team_id == 0]
        team_1_players = [p for p in state.players.values() if p.team_id == 1]

        # Fonction locale pour récupérer la certitude d'un joueur pour son équipe actuelle
        def get_confidence(track_id, target_team):
            if track_id in self.player_votes and self.player_votes[track_id]:
                mean_probs = np.mean(list(self.player_votes[track_id]), axis=0)
                return mean_probs[target_team]
            return 0.0

        # Rééquilibrage Équipe 0 -> Dépassement basculé vers Équipe 1
        if len(team_0_players) > 5:
            # On trie par confiance (du moins sûr au plus sûr)
            team_0_players.sort(key=lambda p: get_confidence(p.track_id, 0))
            # Les X premiers (les plus faibles) sont forcés dans l'équipe adverse
            for i in range(len(team_0_players) - 5):
                team_0_players[i].team_id = 1

        # Rééquilibrage Équipe 1 -> Dépassement basculé vers Équipe 0
        elif len(team_1_players) > 5:
            # On trie par confiance (du moins sûr au plus sûr)
            team_1_players.sort(key=lambda p: get_confidence(p.track_id, 1))
            # Les X premiers sont forcés dans l'équipe adverse
            for i in range(len(team_1_players) - 5):
                team_1_players[i].team_id = 0