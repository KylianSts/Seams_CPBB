"""
detect_team.py
--------------
Module d'Intelligence Artificielle pour le clustering des équipes (GMM).
Analyse les couleurs des joueurs, exclut le parquet, et utilise une logique
de "Soft Voting" temporel pour attribuer les identités d'équipes avec robustesse.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from core.filters import FiltersConfig, get_torso_box

if TYPE_CHECKING:
    from core.state import PlayerState

logger = logging.getLogger(__name__)

# --- Alias de type ---
BBox = Tuple[float, float, float, float]


@dataclass
class TeamConfig:
    """Configuration centralisée pour la classification des équipes."""
    n_components: int = 2              # Nombre d'équipes sur le terrain
    min_calibration_samples: int = 10  # Minimum syndical pour entraîner le GMM
    
    # --- Juge Temporel (Soft Voting) ---
    voting_window: int = 30            # Nombre de frames (passé/futur) pour lisser la décision
    max_players_per_team: int = 5      # Règle métier FIBA
    
    # --- Tolérances Couleur ---
    court_exclusion_margin_h: int = 15 # Marge de Teinte (Hue) pour ignorer le parquet
    court_exclusion_margin_sv: int = 50 # Marge Saturation/Value pour ignorer le parquet


class TeamDetector:
    def __init__(self, cfg: TeamConfig = TeamConfig()):
        self.cfg = cfg
        self.is_calibrated = False
        self.frames_collected = 0
        self.histograms_buffer: List[np.ndarray] = [] 
        
        self.gmm: Optional[GaussianMixture] = None

    # ===========================================================================
    # 1. EXTRACTION VISUELLE ET CALIBRATION
    # ===========================================================================

    def _get_dynamic_court_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Échantillonne la couleur du parquet au centre-bas de l'écran."""
        h, w = frame.shape[:2]
        patch = frame[int(h * 0.85):int(h * 0.95), int(w * 0.45):int(w * 0.55)]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return np.median(hsv_patch, axis=(0, 1))

    def _get_torso_histogram(self, frame: np.ndarray, bbox_px: BBox, court_hsv: np.ndarray, filter_cfg: FiltersConfig) -> Optional[np.ndarray]:
        """Extrait l'histogramme HSV du torse en soustrayant dynamiquement le parquet."""
        h_img, w_img = frame.shape[:2]
        
        # 1. Utilisation du standard DRY : on récupère le torse officiel
        t_box = get_torso_box(bbox_px, filter_cfg)
        x1, y1, x2, y2 = map(int, t_box)
        
        # Sécurité des bords
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        if (x2 - x1) < 10 or (y2 - y1) < 20:
            return None

        torso_bgr = frame[y1:y2, x1:x2]
        if torso_bgr.size == 0:
            return None

        torso_hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        
        # Masque de base : on ignore le noir pur et le blanc brûlé
        base_mask = cv2.inRange(torso_hsv, (0, 20, 20), (180, 255, 240))

        # Masque du parquet (Exclusion)
        dh, dsv = self.cfg.court_exclusion_margin_h, self.cfg.court_exclusion_margin_sv
        lower_court = np.array([max(0, court_hsv[0] - dh), max(0, court_hsv[1] - dsv), max(0, court_hsv[2] - dsv)], dtype=np.uint8)
        upper_court = np.array([min(180, court_hsv[0] + dh), min(255, court_hsv[1] + dsv), min(255, court_hsv[2] + dsv)], dtype=np.uint8)
        
        court_mask = cv2.inRange(torso_hsv, lower_court, upper_court)

        # Fusion des masques
        final_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(court_mask))

        if cv2.countNonZero(final_mask) == 0:
            return None

        # Histogramme 2D (Teinte et Saturation)
        hist = cv2.calcHist([torso_hsv], [0, 1], final_mask, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist.flatten()

    def collect_from_raw_boxes(self, frame: np.ndarray, isolated_boxes: List[Tuple], filter_cfg: FiltersConfig) -> None:
        """Accumule les données pour l'entraînement initial du GMM."""
        if self.is_calibrated: 
            return
            
        court_hsv = self._get_dynamic_court_hsv(frame)
            
        for box in isolated_boxes:
            # Assure la compatibilité si raw_boxes est un tuple long (x1, y1, x2, y2, conf, ...)
            hist = self._get_torso_histogram(frame, box[:4], court_hsv, filter_cfg)
            if hist is not None:
                self.histograms_buffer.append(hist)
                
        self.frames_collected += 1

    def calibrate(self) -> None:
        """Lance l'algorithme d'apprentissage non-supervisé (GMM)."""
        if len(self.histograms_buffer) < self.cfg.min_calibration_samples:
            logger.warning("Pas assez de données pour calibrer les équipes.")
            return

        data = np.array(self.histograms_buffer, dtype=np.float32)
        
        self.gmm = GaussianMixture(
            n_components=self.cfg.n_components, 
            covariance_type='full', 
            random_state=42
        )
        self.gmm.fit(data)
        
        self.is_calibrated = True
        logger.info(f"Calibration GMM terminée ({len(data)} échantillons sur {self.frames_collected} frames).")

    # ===========================================================================
    # 2. INFÉRENCE CONTINUE (Ne modifie PAS l'état)
    # ===========================================================================

    def extract_evidence(
        self, 
        frame: np.ndarray, 
        players_dict: Dict[int, 'PlayerState'], 
        filter_cfg: FiltersConfig
    ) -> Dict[int, Tuple[float, float]]:
        """
        Extrait les probabilités brutes (Proba_A, Proba_B) pour la frame T.
        Retourne un dictionnaire {track_id: (pA, pB)}. 
        C'est le pipeline qui l'insèrera dans le gmm_history des joueurs.
        """
        evidence = {}
        if not self.is_calibrated or self.gmm is None:
            return evidence

        court_hsv = self._get_dynamic_court_hsv(frame)

        for track_id, player in players_dict.items():
            hist = self._get_torso_histogram(frame, player.bbox_px, court_hsv, filter_cfg)
            if hist is not None:
                probs = self.gmm.predict_proba([hist])[0] 
                evidence[track_id] = (float(probs[0]), float(probs[1]))
                
        return evidence

    # ===========================================================================
    # 3. LE JUGE TEMPOREL (Soft Voting & Règles Métier)
    # ===========================================================================

    def resolve_teams(
        self, 
        players_dict: Dict[int, 'PlayerState'], 
        target_frame_idx: int
    ) -> Dict[int, int]:
        """
        Applique l'algorithme de Soft Voting Bidirectionnel.
        Pondère l'historique de chaque joueur par son niveau d'occlusion,
        puis applique la règle métier "Max 5 joueurs par équipe".
        Retourne les assignations finales {track_id: team_id}.
        """
        raw_teams = {}
        confidences = {}

        # 1. SOFT VOTING (Analyse du Casier)
        for tid, player in players_dict.items():
            score_A, score_B = 0.0, 0.0

            for frame_idx, prob_A, prob_B, occlusion in player.gmm_history:
                if abs(frame_idx - target_frame_idx) <= self.cfg.voting_window:
                    
                    # Le Bourreau d'Occlusion : (1 - O)^3 (Écrase les preuves occlusées)
                    occ_factor = (1.0 - occlusion) ** 3
                    
                    # L'Amplificateur de Certitude : P^2
                    score_A += (prob_A ** 2) * occ_factor
                    score_B += (prob_B ** 2) * occ_factor

            if score_A == 0.0 and score_B == 0.0:
                # Aucune preuve récente valide, on garde l'équipe par défaut
                raw_teams[tid] = player.team_id if player.team_id is not None else 0
                confidences[tid] = 0.0
            else:
                raw_teams[tid] = 0 if score_A > score_B else 1
                # Calcul de la certitude pour le rééquilibrage
                confidences[tid] = abs(score_A - score_B) / (score_A + score_B)

        # 2. RÈGLE MÉTIER (Max 5 Joueurs)
        team_0_ids = [tid for tid, t_id in raw_teams.items() if t_id == 0]
        team_1_ids = [tid for tid, t_id in raw_teams.items() if t_id == 1]

        # Fonction de rééquilibrage interne
        def balance_team(overflow_ids, target_team_id):
            # Tri des IDs par ordre croissant de certitude (du plus douteux au plus sûr)
            overflow_ids.sort(key=lambda x: confidences[x])
            
            # Les X joueurs les plus douteux sont basculés dans l'équipe adverse
            excess = len(overflow_ids) - self.cfg.max_players_per_team
            for i in range(excess):
                raw_teams[overflow_ids[i]] = target_team_id

        if len(team_0_ids) > self.cfg.max_players_per_team:
            balance_team(team_0_ids, 1)
        elif len(team_1_ids) > self.cfg.max_players_per_team:
            balance_team(team_1_ids, 0)

        return raw_teams