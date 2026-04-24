"""
collect_from_videos.py
---------------
Extrait des frames individuelles centrées sur des timestamps définis.

Format des timestamps : HMMSS (entier)
  - 1h20m30s → 12030
  - 0h05m30s → 530

Comportements spéciaux :
  - CLIP_DURATION = -1   → toute la vidéo (ignore les timestamps)
  - timestamps = []      → toute la vidéo (équivalent à CLIP_DURATION=-1)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Dataclasses
# ---------------------------------------------------------------------------

TASK_NAME       = "task_8"
INPUT_VIDEO_DIR = Path("data/raw/videos")
OUTPUT_DIR      = Path(f"data/raw/images/object_detection/{TASK_NAME}")

# Durée de la fenêtre d'extraction en secondes. (-1 = toute la vidéo)
CLIP_DURATION = -1

# Échantillonnage : 1 frame sauvegardée toutes les N frames lues.
FRAME_STRIDE = 60 * 5


@dataclass
class VideoInfo:
    """Conteneur pour les métadonnées d'une vidéo."""
    path: Path
    duration_sec: float
    fps: float
    total_frames: int


# ---------------------------------------------------------------------------
# Tags (Données d'entrée)
# ---------------------------------------------------------------------------

TAGS: Dict[str, List[int]] = {
    # Exemple :
    # "match_auch_montaigu.mp4": [10623, 4510],
    # "match_finale.mp4": [],
}


# ===========================================================================
# ÉTAPE 1 — Parsing Temporel
# ===========================================================================

def parse_hmmss(value: int) -> float:
    """Convertit un entier HMMSS en secondes."""
    seconds = value % 100
    minutes = (value // 100) % 100
    hours   = value // 10000

    if not (0 <= seconds <= 59):
        raise ValueError(f"Secondes invalides dans {value} : {seconds} (attendu 0-59)")
    if not (0 <= minutes <= 59):
        raise ValueError(f"Minutes invalides dans {value} : {minutes} (attendu 0-59)")

    return float(hours * 3600 + minutes * 60 + seconds)


def hmmss_to_label(value: int) -> str:
    """Retourne un label lisible pour le nom de fichier : 12030 → '1h20m30s'."""
    seconds = value % 100
    minutes = (value // 100) % 100
    hours   = value // 10000
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


# ===========================================================================
# ÉTAPE 2 — Moteur d'Extraction Vidéo
# ===========================================================================

def get_video_info(video_path: Path) -> Optional[VideoInfo]:
    """Récupère les métadonnées de la vidéo de manière sécurisée."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Impossible d'ouvrir la vidéo : {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    cap.release()
    return VideoInfo(path=video_path, duration_sec=duration, fps=fps, total_frames=total_frames)


def extract_clip(
    video: VideoInfo,
    output_dir: Path,
    timestamp_sec: Optional[float],
    clip_duration: float,
    stride: int,
    label: str,
) -> int:
    """
    Extrait et sauvegarde les frames d'une fenêtre temporelle donnée.
    Utilise cap.grab() pour une lecture accélérée des frames ignorées.
    """
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        return 0

    # Calcul de la plage de frames
    if clip_duration == -1 or timestamp_sec is None:
        start_f = 0
        end_f   = video.total_frames
    else:
        half_f  = int((clip_duration / 2) * video.fps)
        ts_f    = int(timestamp_sec * video.fps)
        start_f = max(0, ts_f - half_f)
        end_f   = min(video.total_frames, ts_f + half_f)

    if end_f <= start_f:
        cap.release()
        return 0

    # Placement du curseur de lecture
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    video_stem  = video.path.stem
    saved_count = 0
    total_to_read = end_f - start_f

    # Barre de progression pour ce clip spécifique
    desc = f"  ↳ {label}"
    with tqdm(total=total_to_read, desc=desc, unit="f", leave=False) as pbar:
        for frame_index in range(total_to_read):
            # Si on doit garder cette frame (modulo stride)
            if frame_index % stride == 0:
                ret, frame = cap.read() # Lit ET décode les pixels (lent)
                if not ret:
                    break
                
                abs_frame = start_f + frame_index
                filename  = output_dir / f"{video_stem}_{label}_frame_{abs_frame:05d}.png"
                cv2.imwrite(str(filename), frame)
                saved_count += 1
            else:
                # Si on ignore la frame, on se contente d'avancer le curseur 
                ret = cap.grab()
                if not ret:
                    break
            
            pbar.update(1)

    cap.release()
    return saved_count


# ===========================================================================
# Point d'entrée & Orchestration
# ===========================================================================

def main() -> None:
    logger.info("=== video_tagger — Extraction optimisée de frames ===")
    mode_desc = "(Toute la vidéo)" if CLIP_DURATION == -1 else f"({CLIP_DURATION}s par tag)"
    logger.info(f"Config : CLIP_DURATION={CLIP_DURATION}s {mode_desc} | FRAME_STRIDE=1/{FRAME_STRIDE}")

    if not TAGS:
        logger.warning("Le dictionnaire TAGS est vide. Fin du programme.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for filename, timestamps in TAGS.items():
        video_path = INPUT_VIDEO_DIR / filename
        video_info = get_video_info(video_path)

        if not video_info:
            continue

        logger.info(f"\nTraitement : {filename} ({video_info.duration_sec:.0f}s @ {video_info.fps:.0f}fps)")

        # Cas 1 : Extraction de toute la vidéo
        if not timestamps or CLIP_DURATION == -1:
            reason = "Liste vide" if not timestamps else "CLIP_DURATION = -1"
            logger.info(f"→ Mode Extraction Complète ({reason})")
            n = extract_clip(
                video=video_info,
                output_dir=OUTPUT_DIR,
                timestamp_sec=None,
                clip_duration=-1,
                stride=FRAME_STRIDE,
                label="full"
            )
            total_saved += n
            logger.info(f"✓ [Full] {n} frames sauvegardées.")

        # Cas 2 : Extraction par tags
        else:
            logger.info(f"→ {len(timestamps)} tag(s) à traiter.")
            for raw_ts in sorted(timestamps):
                try:
                    ts_sec = parse_hmmss(raw_ts)
                except ValueError as e:
                    logger.error(f"  Tag ignoré ({raw_ts}) : {e}")
                    continue

                if ts_sec > video_info.duration_sec:
                    logger.warning(f"  Tag ignoré ({raw_ts}) : Dépasse la durée de la vidéo.")
                    continue

                label = hmmss_to_label(raw_ts)
                n = extract_clip(
                    video=video_info,
                    output_dir=OUTPUT_DIR,
                    timestamp_sec=ts_sec,
                    clip_duration=CLIP_DURATION,
                    stride=FRAME_STRIDE,
                    label=label
                )
                total_saved += n
                logger.info(f"  ✓ [{label}] {n} frames sauvegardées.")

    logger.info(f"\n=== Bilan : {total_saved} frames générées au total dans {OUTPUT_DIR.name}/ ===")


if __name__ == "__main__":
    main()