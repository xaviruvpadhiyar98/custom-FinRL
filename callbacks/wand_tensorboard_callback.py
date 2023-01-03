from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from pathlib import Path


class TensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        holding_plus_shares = self.locals["infos"][-1].popitem()
        if isinstance(holding_plus_shares[0], int):
            holdings = holding_plus_shares[1]["current_holdings"]
            shares = np.sum(holding_plus_shares[1]["shares"])
            self.logger.record(key="train/holdings", value=holdings)
            self.logger.record(key="train/shares", value=shares)
        return True
