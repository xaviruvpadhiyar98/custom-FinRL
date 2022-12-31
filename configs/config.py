from pathlib import Path
from time import time
from yaml import safe_load

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
MODEL_NAME = "ppo"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
RANDOM_DIR = int(time())

# # Add your Stock here with time period from current to last n days
TICKERS = ["TCS.NS", "BAJFINANCE.NS", "WIPRO.NS", "CANBK.NS", "IGL.NS", "TATACONSUM.NS"]
TICKERS.sort()

# Dont Modify the below for now
TECHNICAL_INDICATORS = [
    "EMA_8",
    "EMA_21",
    "RSI_14",
    "BBAND_UP",
    "BBAND_MID",
    "BBAND_DOWN",
]
INTERVAL = "1h"
PERIOD = f"{360+7}d"

TIMESTEPS = 10000
MODEL_SAVE_AT_N_STEPS = 5000
STEPS = 1
WANDB_PROJECT_NAME = "StockTradingRL"
WANDB_SWEEP_CONFIG_FILE_PATH = Path(r"configs/wandb_sweep_config.yaml")
WANDB_SWEEP_CONFIG = safe_load(WANDB_SWEEP_CONFIG_FILE_PATH.read_text())
