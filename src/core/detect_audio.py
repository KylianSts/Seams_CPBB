import logging
import subprocess
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from scipy.io import wavfile
from scipy.signal import spectrogram

logger = logging.getLogger(__name__)

@dataclass
class AudioEvent:
    event_type: str
    timestamp: float
    duration: float
    confidence: float

@dataclass
class AudioConfig:
    # ── Paramètres Cible : Sifflet (Ultra-Permissif) ──────────
    # Bande élargie pour capter les sifflets distordus ou de différentes marques
    whistle_freq_min: float = 3000.0
    whistle_freq_max: float = 5000.0
    
    # On descend à 50 millisecondes pour les coups de sifflets ultra-brefs
    whistle_min_duration: float = 0.05  
    whistle_max_duration: float = 3.0 
    
    # ── Paramètres d'Analyse (DSP) ───────────
    target_sr: int = 22050
    nperseg: int = 2048
    noverlap: int = 1536  
    
    # Tolérance à l'impureté (écho, foule, souffle). 
    # Passé de 0.15 à 0.60 : on accepte des sons très peu "musicaux"
    whistle_flatness_tolerance: float = 0.60

# ===========================================================================
# NOYAU D'ANALYSE MATHÉMATIQUE ASSOUPLI
# ===========================================================================

def _extract_wav(video_path: Path, out_path: Path, sr: int) -> bool:
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", 
           "-ar", str(sr), "-ac", "1", "-loglevel", "error", str(out_path)]
    return subprocess.run(cmd, capture_output=True).returncode == 0

def _compute_spectral_flatness(Sxx_band: np.ndarray, tolerance: float, eps: float = 1e-10) -> np.ndarray:
    arith_mean = np.mean(Sxx_band + eps, axis=0)
    geom_mean = np.exp(np.mean(np.log(Sxx_band + eps), axis=0))
    flatness = geom_mean / arith_mean
    return np.clip(1.0 - (flatness / tolerance), 0.0, 1.0)

def _compute_whistle_scores(Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    dt = times[1] - times[0]
    mask = (freqs >= cfg.whistle_freq_min) & (freqs <= cfg.whistle_freq_max)
    if not np.any(mask): return np.zeros(len(times))

    # Suppression du filtre morphologique (on garde le signal brut pour ne rien rater)
    band_data = Sxx[mask, :]
        
    band_energy = np.mean(band_data, axis=0) + 1e-10
    total_energy = np.mean(Sxx, axis=0) + 1e-10
    
    # 1. Prominence très assouplie
    # Si la bande représente au moins 12.5% de l'énergie totale, le score est de 1.0 (Maximum)
    prominence = band_energy / total_energy
    prom_score = np.clip(prominence * 8.0, 0.0, 1.0)
    
    # 2. Tonicité extrêmement tolérante
    tonality_score = _compute_spectral_flatness(band_data, cfg.whistle_flatness_tolerance)
    
    # 3. Détection de pic d'énergie local (Spike dynamique)
    # Au lieu d'un seuil binaire, on utilise une courbe continue
    window_size = max(1, int(3.0 / dt))  # Contexte de 3 secondes
    padded_energy = np.pad(band_energy, (window_size//2, window_size//2), mode='reflect')
    
    energy_score = np.zeros(len(times))
    for i in range(len(times)):
        local_mean = np.mean(padded_energy[i:i+window_size])
        spike_ratio = band_energy[i] / (local_mean + 1e-10)
        # S'il est juste 10% plus fort que la moyenne locale, le score commence à monter.
        # S'il est 60% plus fort, le score est à 1.0.
        energy_score[i] = np.clip((spike_ratio - 1.1) / 0.5, 0.0, 1.0)
    
    # Score final : multiplication douce des probabilités
    final_score = energy_score * tonality_score * prom_score
        
    return final_score

# ===========================================================================
# MOTEUR DE LIAISON TEMPORELLE
# ===========================================================================

def _merge_events(scores: np.ndarray, times: np.ndarray, min_d: float, max_d: float, gap: float = 0.4) -> List[AudioEvent]:
    # Ultra-sensibilité : Le moindre score > 2% déclenche un événement
    active = scores > 0.02 
    events = []
    start_idx = None
    
    dt = times[1] - times[0]
    max_gap_frames = int(gap / dt)
    
    for i in range(len(active)):
        if active[i] and start_idx is None:
            start_idx = i
        elif not active[i] and start_idx is not None:
            # On ignore les micro-coupures
            if i + 1 < len(active) and np.any(active[i:min(i + max_gap_frames, len(active))]):
                continue 
            
            dur = times[i-1] - times[start_idx]
            if min_d <= dur <= max_d:
                # On prend la confiance MAXIMALE du pic, pas la moyenne, 
                # pour ne pas pénaliser les sifflets qui "fade out" doucement.
                conf = float(np.max(scores[start_idx:i]))
                events.append(AudioEvent("sifflet", float(times[start_idx]), float(dur), round(conf, 3)))
            start_idx = None
            
    if start_idx is not None:
        dur = times[-1] - times[start_idx]
        if min_d <= dur <= max_d:
            conf = float(np.max(scores[start_idx:]))
            events.append(AudioEvent("sifflet", float(times[start_idx]), float(dur), round(conf, 3)))
            
    return events

def detect_events_from_audio(wav_path: Path, cfg: AudioConfig) -> List[AudioEvent]:
    rate, data = wavfile.read(wav_path)
    data = data.astype(np.float32) / np.max(np.abs(data))
    
    f, t, Sxx = spectrogram(data, fs=rate, window='hann', nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    Sxx = np.abs(Sxx) ** 2  

    w_scores = _compute_whistle_scores(Sxx, f, t, cfg)
    # Gap de 0.4s pour bien fusionner les sifflets trillés (ceux où l'arbitre souffle par à-coups)
    w_ev = _merge_events(w_scores, t, cfg.whistle_min_duration, cfg.whistle_max_duration, gap=0.4)

    return sorted(w_ev, key=lambda e: e.timestamp)

def get_match_audio_events(video_path: Path, cfg: AudioConfig = AudioConfig()) -> List[AudioEvent]:
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
        print(f"Extraction et analyse [MODE RADAR OUVERT] en cours pour : {video.name}...")
        results = get_match_audio_events(video)
        print(f"\n--- ANALYSE AUDIO TERMINÉE : {video.name} ---")
        for e in results:
            mins, secs = divmod(int(e.timestamp), 60)
            print(f"[{e.event_type.upper()}] {mins:02d}:{secs:02d} | Conf MAX: {e.confidence:.3f} | Durée: {e.duration:.2f}s")
        if not results:
            print("Aucun sifflet détecté (le fichier audio est-il muet ?).")
    else:
        print(f"Fichier vidéo introuvable : {video.resolve()}")