"""
detect_audio.py
---------------
Module d'analyse sémantique de l'audio par Deep Learning.
Utilise YAMNet (MobileNetV1) pour classifier les sifflets et la joie de la foule.
"""

import logging
import subprocess
import csv
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from scipy.io import wavfile

# Configuration pour masquer les warnings inutiles de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)

@dataclass
class AudioEvent:
    event_type: str
    timestamp: float
    duration: float
    confidence: float

@dataclass
class YamnetConfig:
    target_sr: int = 16000           # YAMNet EXIGE du 16 kHz
    
    # Seuils de confiance (de 0.0 à 1.0)
    whistle_threshold: float = 0.25  # Un sifflet est très perçant, 25% de certitude YAMNet suffit
    crowd_threshold: float = 0.15    # La foule est un bruit de fond, un seuil bas est recommandé
    
    # Temps minimum entre deux événements pour ne pas les fusionner (en secondes)
    merge_gap_s: float = 1.0         


# ===========================================================================
# 1. UTILITAIRES & CHARGEMENT
# ===========================================================================

def _extract_wav(video_path: Path, out_path: Path, sr: int) -> bool:
    """Extrait la piste audio de la vidéo au format WAV mono, 16kHz."""
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", 
           "-ar", str(sr), "-ac", "1", "-loglevel", "error", str(out_path)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _load_yamnet():
    """Charge le modèle YAMNet depuis TF Hub et récupère la liste des 521 classes."""
    logger.info("Chargement du modèle YAMNet (peut prendre quelques secondes au 1er lancement)...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Récupération du fichier CSV contenant les noms des classes intégré au modèle
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
            
    return model, class_names


# ===========================================================================
# 2. MOTEUR D'ANALYSE ET FUSION
# ===========================================================================

def _merge_predictions(scores: np.ndarray, threshold: float, event_name: str, gap_s: float, frame_duration_s: float = 0.48) -> List[AudioEvent]:
    """
    Fusionne les frames où la probabilité dépasse le seuil en événements continus.
    YAMNet génère une prédiction toutes les 0.48 secondes.
    """
    active_frames = scores > threshold
    events = []
    start_idx = None
    
    max_gap_frames = max(1, int(gap_s / frame_duration_s))
    
    for i in range(len(active_frames)):
        if active_frames[i] and start_idx is None:
            start_idx = i
        elif not active_frames[i] and start_idx is not None:
            # Regarde en avant pour voir si l'événement reprend vite (micro-coupure)
            if i + 1 < len(active_frames) and np.any(active_frames[i:min(i + max_gap_frames, len(active_frames))]):
                continue 
            
            # Fin réelle de l'événement
            duration = (i - start_idx) * frame_duration_s
            max_conf = float(np.max(scores[start_idx:i]))
            timestamp = start_idx * frame_duration_s
            events.append(AudioEvent(event_name, timestamp, duration, round(max_conf, 3)))
            start_idx = None
            
    # Gérer le cas où l'événement touche la fin du fichier
    if start_idx is not None:
        duration = (len(active_frames) - start_idx) * frame_duration_s
        max_conf = float(np.max(scores[start_idx:]))
        timestamp = start_idx * frame_duration_s
        events.append(AudioEvent(event_name, timestamp, duration, round(max_conf, 3)))
            
    return events


def detect_events_from_audio(wav_path: Path, cfg: YamnetConfig) -> List[AudioEvent]:
    # 1. Chargement de l'audio
    rate, data = wavfile.read(wav_path)
    # YAMNet attend un tenseur de float32 entre -1.0 et +1.0
    waveform = data.astype(np.float32) / 32768.0  
    
    # 2. Chargement du Modèle
    model, class_names = _load_yamnet()
    
    # Index des classes qui nous intéressent dans AudioSet
    whistle_idx = class_names.index('Whistle')
    
    # La foule est souvent une combinaison de ces trois classes
    crowd_classes = ['Cheering', 'Applause', 'Crowd']
    crowd_indices = [class_names.index(c) for c in crowd_classes if c in class_names]

    # 3. Inférence (La magie du Deep Learning s'opère ici)
    # Le modèle renvoie un tableau de scores de taille (N_frames, 521 classes)
    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy() # Conversion du tenseur TF en array Numpy

    # 4. Extraction des signaux spécifiques
    whistle_scores = scores[:, whistle_idx]
    
    # Pour la foule, on additionne les probabilités des classes d'acclamation
    crowd_scores = np.sum(scores[:, crowd_indices], axis=1)

    # 5. Fusion temporelle
    whistle_events = _merge_predictions(whistle_scores, cfg.whistle_threshold, "sifflet", cfg.merge_gap_s)
    crowd_events = _merge_predictions(crowd_scores, cfg.crowd_threshold, "foule", cfg.merge_gap_s)

    # Combinaison et tri chronologique
    all_events = whistle_events + crowd_events
    return sorted(all_events, key=lambda e: e.timestamp)


def get_match_audio_events(video_path: Path, cfg: YamnetConfig = YamnetConfig()) -> List[AudioEvent]:
    tmp = video_path.parent / f"_tmp_{video_path.stem}.wav"
    events = []
    if _extract_wav(video_path, tmp, cfg.target_sr):
        events = detect_events_from_audio(tmp, cfg)
        if tmp.exists(): tmp.unlink()
    return events

# ===========================================================================
# TEST RAPIDE
# ===========================================================================

if __name__ == "__main__":
    import sys
    
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/videos/nantes.mp4")
    
    if video.exists():
        print(f"Extraction et analyse YAMNet en cours pour : {video.name}...")
        results = get_match_audio_events(video)
        print(f"\n--- ANALYSE DEEP LEARNING TERMINÉE : {video.name} ---")
        
        for e in results:
            mins, secs = divmod(int(e.timestamp), 60)
            icon = "📣" if e.event_type == "foule" else "🛑"
            print(f"{icon} [{e.event_type.upper()}] {mins:02d}:{secs:02d} | Conf MAX: {e.confidence:.3f} | Durée: {e.duration:.2f}s")
            
        if not results:
            print("Aucun événement détecté.")
    else:
        print(f"Fichier vidéo introuvable : {video.resolve()}")