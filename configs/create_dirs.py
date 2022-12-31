# Create the Directories Required

from pathlib import Path
from config import DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR


def create_dirs():
    # Create Dirs
    for dir in [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)
