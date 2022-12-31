from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from pathlib import Path


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose, model_save_path, model_save_freq):
        super().__init__(verbose)
        self.model_save_path = model_save_path
        self.model_save_freq = model_save_freq

    def _on_step(self) -> bool:
        self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        holding_plus_shares = self.locals["infos"][-1].popitem()

        if holding_plus_shares[0][:4].isdigit():
            holdings = holding_plus_shares[1]["current_holdings"]
            shares = np.sum(holding_plus_shares[1]["shares"])
            self.logger.record(key="train/holdings", value=holdings)
            self.logger.record(key="train/shares", value=shares)

        if self.n_calls % self.model_save_freq == 0:
            self.save_model()
        return True

    def save_model(self) -> None:
        model_path_save = f"{self.model_save_path}/{self.n_calls}"
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        self.model.save(model_path_save)
        wandb.save(model_path_save)
