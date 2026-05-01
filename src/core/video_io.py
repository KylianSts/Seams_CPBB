"""
video_io.py
-----------
Module utilitaire pour les opérations d'Entrée/Sortie vidéo.
Gère spécifiquement le muxing (fusion) audio/vidéo via FFmpeg de manière
cross-platform et sécurisée.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

def add_audio_from_source(source_path: Path, silent_video_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Extrait la piste audio de la vidéo source originale et l'ajoute à la vidéo annotée (muette).
    Ré-encode la vidéo au standard H.264 (yuv420p) pour garantir la lecture sur tous les lecteurs 
    (notamment QuickTime et les navigateurs Web).
    
    Args:
        source_path: Chemin de la vidéo originale (contenant le son).
        silent_video_path: Chemin de la vidéo traitée par l'IA (muette).
        output_path: Chemin de destination. Si None, écrase la vidéo muette.
        
    Returns:
        bool: True si l'opération a réussi, False sinon.
    """
    if not source_path.exists():
        logger.error(f"Impossible d'ajouter l'audio : Fichier source introuvable ({source_path})")
        return False
        
    if not silent_video_path.exists():
        logger.error(f"Impossible d'ajouter l'audio : Vidéo muette introuvable ({silent_video_path})")
        return False

    # Si aucun chemin de sortie n'est fourni, on va écraser la vidéo muette d'origine
    if output_path is None:
        output_path = silent_video_path

    # On crée un nom de fichier temporaire sécurisé pour la vidéo muette
    temp_silent = silent_video_path.with_name(f"_temp_muet_{silent_video_path.name}")
    
    try:
        # Renommage cross-platform (replace écrase silencieusement si la cible existe sur Windows)
        silent_video_path.replace(temp_silent)
    except OSError as e:
        logger.error(f"Erreur système lors de la préparation du fichier temporaire : {e}")
        return False

    logger.info("Fusion de la piste audio d'origine avec la vidéo annotée...")

    # Commande FFmpeg ultra-robuste avec conversion H.264 universelle
    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_silent),        # Input 0 : Vidéo annotée muette
        "-i", str(source_path),        # Input 1 : Vidéo source originale
        
        # --- Paramètres Vidéo ---
        "-c:v", "libx264",             # Codec vidéo universel H.264
        "-preset", "fast",             # Vitesse d'encodage (compromis poids/vitesse)
        "-pix_fmt", "yuv420p",         # Espace colorimétrique obligatoire pour compatibilité Web/Apple
        
        # --- Mapping et Audio ---
        "-map", "0:v:0",               # On prend le flux vidéo de l'Input 0
        "-map", "1:a:0?",              # On prend l'audio de l'Input 1 (le '?' évite de crasher s'il n'y a pas de son)
        "-c:a", "aac",                 # Codec audio universel
        
        "-shortest",                   # Coupe proprement à la fin du flux le plus court
        "-loglevel", "error",          # N'affiche que les erreurs dans la console Python
        str(output_path)               # Fichier final
    ]
    
    try:
        # Exécution de la commande
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Muxing audio réussi avec succès !")
            # Nettoyage propre du fichier temporaire
            if temp_silent.exists():
                temp_silent.unlink()
            return True
        else:
            logger.error(f"Échec de FFmpeg. Erreur : {result.stderr}")
            # Restauration du fichier en cas d'échec de FFmpeg
            if temp_silent.exists():
                temp_silent.replace(output_path)
            return False
            
    except FileNotFoundError:
        # Cette erreur est levée si Python ne trouve pas l'exécutable "ffmpeg" sur le système
        logger.error("FFmpeg n'est pas installé ou n'est pas dans le PATH du système.")
        logger.warning("La vidéo a été sauvegardée sans le son original.")
        # Restauration du fichier muet
        if temp_silent.exists():
            temp_silent.replace(output_path)
        return False