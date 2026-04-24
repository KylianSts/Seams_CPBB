"""
collect_from_youtube.py
--------------------
Télécharge des vidéos YouTube et en extrait directement des séquences 
de frames nettes pour constituer un dataset d'entraînement.

Optimisations apportées :
  - Utilisation de `cap.grab()` lors des sauts (stride) pour accélérer la lecture.
  - Architecture modulaire basée sur des Dataclasses pour la configuration.
  - Barre de progression globale (`tqdm`) pour le suivi multithreadé.

Dépendances : yt-dlp, opencv-python, ffmpeg (installé sur le système), tqdm
"""

import gc
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import yt_dlp
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

TASK_NAME = "task_8"
OUTPUT_DIR = Path(f"data/raw/images/object_detection/{TASK_NAME}")

@dataclass
class SequenceConfig:
    """Configuration pour l'extraction des séquences temporelles."""
    num_sequences: int = 3
    seq_len: int = 5
    stride: int = 10
    sharpness_threshold: float = 100.0


# ===========================================================================
# ÉTAPE 1 — Téléchargement (yt-dlp)
# ===========================================================================

def build_ydl_opts(output_dir: Path) -> dict:
    """Construit les options yt-dlp pour le téléchargement optimisé de la vidéo."""
    return {
        "format": (
            "bestvideo[height<=1080][vcodec^=av01]"
            "/bestvideo[height<=1080][vcodec^=vp9]"
            "/bestvideo[height<=1080]"
            "/best"
        ),
        "outtmpl": str(output_dir / "%(id)s_temp.%(ext)s"),
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr", "web_embedded", "default"],
            }
        },
        "ignoreerrors": True,
        "retries": 2,
        "fragment_retries": 10,
        "cookies-from-browser": "firefox"
    }


def download_video(url: str, output_dir: Path) -> Tuple[Optional[Path], str]:
    """
    Télécharge la vidéo depuis YouTube.
    Retourne (Chemin_du_fichier_temporaire, Video_ID).
    """
    ydl_opts = build_ydl_opts(output_dir)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not info:
        return None, "unknown"

    video_id = info.get("id", "unknown")
    video_path = Path(ydl.prepare_filename(info))

    if not video_path.exists() or video_path.stat().st_size == 0:
        return None, video_id

    return video_path, video_id


# ===========================================================================
# ÉTAPE 2 — Moteur d'Extraction et de Netteté
# ===========================================================================

def is_frame_sharp(frame, threshold: float) -> bool:
    """Évalue la netteté d'une frame via la variance du filtre Laplacien."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold


def extract_sequences(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    config: SequenceConfig,
) -> int:
    """
    Recherche et extrait des séquences complètes de frames nettes.
    Utilise cap.grab() pour les sauts de stride afin d'optimiser le CPU.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV ne peut pas ouvrir la vidéo : {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        required_span = config.seq_len * config.stride

        if total_frames <= required_span:
            raise RuntimeError(f"Vidéo trop courte ({total_frames} frames) pour la configuration demandée.")

        saved_seqs = 0
        max_attempts = config.num_sequences * 30

        for attempt in range(max_attempts):
            if saved_seqs >= config.num_sequences:
                break

            # Tirage d'un point de départ aléatoire sécurisé
            start_idx = random.randint(0, total_frames - 1 - required_span)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            
            sequence_frames = []
            is_valid_sequence = True

            for i in range(config.seq_len):
                success, frame = cap.read()
                
                # Vérification de la lecture et de la netteté
                if not success or not is_frame_sharp(frame, config.sharpness_threshold):
                    is_valid_sequence = False
                    break
                
                real_frame_idx = start_idx + (i * config.stride)
                sequence_frames.append((real_frame_idx, frame))
                
                # Si on n'est pas à la dernière frame de la séquence, on saute le 'stride'
                if i < config.seq_len - 1:
                    for _ in range(config.stride - 1):
                        cap.grab() # Beaucoup plus rapide que cap.read()

            # Sauvegarde si toute la séquence est valide
            if is_valid_sequence:
                seq_num = saved_seqs + 1
                for frame_idx, frm in sequence_frames:
                    dest = output_dir / f"{video_id}_seq{seq_num}_frame{frame_idx:05d}.png"
                    cv2.imwrite(str(dest), frm)
                saved_seqs += 1

        return saved_seqs

    finally:
        cap.release()


# ===========================================================================
# ÉTAPE 3 — Nettoyage
# ===========================================================================

def cleanup_temp_file(video_path: Optional[Path], retries: int = 3, delay: float = 1.0) -> None:
    """Supprime le fichier vidéo temporaire de manière sécurisée."""
    if video_path is None or not video_path.exists():
        return

    for attempt in range(1, retries + 1):
        try:
            video_path.unlink()
            return
        except OSError as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.warning(f"Impossible de supprimer le fichier {video_path} : {e}")


# ===========================================================================
# Orchestration Multithread
# ===========================================================================

def process_single_video(url: str, output_dir: Path, config: SequenceConfig) -> str:
    """Traite une seule URL : Téléchargement -> Extraction -> Nettoyage."""
    video_path: Optional[Path] = None
    video_id: str = "unknown"
    status_msg = ""

    try:
        video_path, video_id = download_video(url, output_dir)

        if not video_path:
            status_msg = f"Échec DL ou fichier vide pour {url}"
            return status_msg

        saved = extract_sequences(video_path, video_id, output_dir, config)
        status_msg = f"[OK] {video_id} : {saved}/{config.num_sequences} séquences."

    except RuntimeError as e:
        status_msg = f"Erreur vidéo ({video_id}) : {e}"
    except Exception as e:
        status_msg = f"Erreur inattendue ({url}) : {e}"

    finally:
        cleanup_temp_file(video_path)
        gc.collect()

    return status_msg


def run_extraction_pipeline(
    urls: List[str],
    output_dir: Path,
    config: SequenceConfig,
    max_workers: int = 2,
    rate_limit_range: Tuple[float, float] = (7.5, 14.2),
) -> None:
    """Gère le pool de threads et l'affichage global de la progression."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        f"Démarrage Pipeline : {len(urls)} vidéos "
        f"({config.num_sequences} séquences de {config.seq_len} frames, stride={config.stride})"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, url, output_dir, config): url
            for url in urls
        }

        # Utilisation de tqdm pour suivre l'avancement global
        with tqdm(total=len(urls), desc="Progression globale", unit="vid") as pbar:
            for future in as_completed(futures):
                url = futures[future]
                try:
                    result_msg = future.result()
                    # On affiche le message de résultat propre dans la console sans casser tqdm
                    tqdm.write(result_msg) 
                except Exception as e:
                    tqdm.write(f"[ERREUR CRITIQUE] {url} : {e}")

                pbar.update(1)
                
                # Pause anti-ban (rate limit)
                time.sleep(random.uniform(*rate_limit_range))

    logger.info("=== Pipeline terminé ===")

# ===========================================================================
# Point d'entrée
# ===========================================================================

def main():
        
    TRAIN_URLS = [
        # "https://www.youtube.com/watch?v=EXEMPLE1",
        # "https://www.youtube.com/watch?v=EXEMPLE2",
    ]

    config = SequenceConfig(
        num_sequences=3,
        seq_len=3,
        stride=10,
        sharpness_threshold=100.0
    )

    run_extraction_pipeline(
        urls=TRAIN_URLS,
        output_dir=OUTPUT_DIR,
        config=config,
        max_workers=12,
    )

if __name__ == "__main__":
    main()