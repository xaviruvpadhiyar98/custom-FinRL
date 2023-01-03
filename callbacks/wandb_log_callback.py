from stable_baselines3.common.callbacks import BaseCallback
from feature_engineering.technical_indicators import (
    calculate_sharpe_and_get_ending_amount,
)
import wandb


class WandDBLogCallback(BaseCallback):
    def __init__(self, verbose):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals["infos"]
        sharpe, ending_amount = calculate_sharpe_and_get_ending_amount(info)
        # wandb.log({"Sharpe": sharpe, "Ending Amount": ending_amount})
        print({"Sharpe": sharpe, "Ending Amount": ending_amount})
        return True
