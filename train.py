import sys
import traceback
from downloaders.yahoo_downloader import yahoo_downloader
from feature_engineering.technical_indicators import (
    add_technical_indicators,
    clean_data,
    add_buy_sell_hold_single_column,
    calculate_sharpe_and_get_ending_amount,
)
from callbacks.wand_tensorboard_callback import TensorboardCallback
from callbacks.wandb_log_callback import WandDBLogCallback
from environments.stock_trading_env import StockTradingEnv
from configs.create_dirs import create_dirs
from configs.config import *
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
import numpy as np

from stable_baselines3 import PPO
import wandb

wandb.tensorboard.patch(
    root_logdir=Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}").as_posix()
)

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
train_arrays = (
    train_df[["Close"] + TECHNICAL_INDICATORS + ["Buy/Sold/Hold"]]
    .groupby(train_df.index)
    .apply(np.array)
    .values
)
trade_arrays = (
    trade_df[["Close"] + TECHNICAL_INDICATORS + ["Buy/Sold/Hold"]]
    .groupby(trade_df.index)
    .apply(np.array)
    .values
)

train_env = Monitor(StockTradingEnv(train_arrays, TICKERS, TECHNICAL_INDICATORS))
trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, TECHNICAL_INDICATORS))
model_ppo = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=0,
    tensorboard_log=Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}"),
    clip_range=0.171,
    ent_coef=0.005118,
    gae_lambda=0.9221,
    learning_rate=0.0000199,
    n_steps=2048,
)
callback_after_eval = WandDBLogCallback(verbose=0)

checkpoint_callback = CheckpointCallback(
    save_freq=MODEL_SAVE_AT_N_STEPS,
    save_path=f"{TRAINED_MODEL_DIR}/BEST_MODEL/{MODEL_NAME}/",
    name_prefix="best_model_best_hyperparams",
    save_replay_buffer=True,
    save_vecnormalize=True,
    verbose=0,
)
eval_callback = EvalCallback(
    trade_env,
    best_model_save_path=f"{TRAINED_MODEL_DIR}/BEST_MODEL/{MODEL_NAME}/",
    log_path=f"{LOGS_DIR}",
    eval_freq=MODEL_SAVE_AT_N_STEPS,
    deterministic=False,
    render=False,
    n_eval_episodes=10,
    verbose=0,
    callback_after_eval=callback_after_eval,
)

tensorboard_callback = TensorboardCallback()

callback = CallbackList(
    [
        checkpoint_callback,
        eval_callback,
        tensorboard_callback,
    ]
)

model_ppo.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="best_model_best_hyper_params",
    callback=callback,
    progress_bar=True,
)
