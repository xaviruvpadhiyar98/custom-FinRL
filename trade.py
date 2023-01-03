from typing import List
from environments.stock_trading_env import StockTradingEnv
from configs.create_dirs import create_dirs
from configs.config import *
import pandas as pd

from stable_baselines3.common.monitor import Monitor

import quantstats as qs
import numpy as np

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

set_random_seed(42, using_cuda=True)

# wandb.init(sync_tensorboard=True, save_code=False)


create_dirs()
# df = yahoo_downloader(tickers=TICKERS, period=PERIOD, interval=INTERVAL)
# df = add_technical_indicators(df=df, tickers=TICKERS)
# df = clean_data(df=df, tickers=TICKERS)
# df = add_buy_sell_hold_single_column(df=df)
# df.to_csv(f"datasets/{'-'.join(TICKERS)}")


df = pd.read_csv(f"datasets/{'-'.join(TICKERS)}")
df = df.set_index("Unnamed: 0")
train_size = df.index.values[-1] - int(df.index.values[-1] * 0.10)
train_df = df.loc[:train_size]
trade_df = df.loc[train_size + 1 :]
trade_arrays = (
    trade_df[["Close"] + TECHNICAL_INDICATORS + ["Buy/Sold/Hold"]]
    .groupby(trade_df.index)
    .apply(np.array)
    .values
)


def calculate_sharpe_and_get_ending_amount(infos: dict):
    last_key = next(reversed(infos))
    if isinstance(last_key, str):
        infos.pop(last_key)

    current_holdings = np.array([v["current_holdings"] for k, v in infos.items()])
    returns = (current_holdings[1:] - current_holdings[:-1]) / current_holdings[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns)
    return sharpe_ratio, current_holdings[-1]


def inference(model_path, env):
    env.reset()
    model_ppo = PPO.load(
        path=model_path,
        env=env,
        custom_objects={"tensorboard_log": None},
    )
    obs = env.reset()
    while True:
        action, _states = model_ppo.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            return calculate_sharpe_and_get_ending_amount(info)


def main():
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, TECHNICAL_INDICATORS))
    model_paths = Path(r"trained_models\BEST_MODEL\ppo").rglob("*.zip")
    ea = 0
    for model_path in model_paths:
        sharpe, ending_amount = inference(
            model_path=model_path.as_posix(), env=trade_env
        )
        if ending_amount > ea:
            print(f"{model_path}\t{sharpe}\t{ending_amount}")
            ea = ending_amount


if __name__ == "__main__":
    main()
