# Autora: Giulia Katherine Romas Tomazeli

from pathlib import Path
import logging
from configs.config import LOGS_DIR

def get_logger(model_name, log_name):
    """
    Cria um logger para um modelo.
    """

    log_dir = LOGS_DIR / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    logger = logging.getLogger( f"{model_name}_{log_name}")
    logger.setLevel(logging.INFO)

    # evita adicionar handlers duplicados
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8" )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger