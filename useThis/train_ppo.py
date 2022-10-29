from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import config
from callbacks.SaveOnBestTrainingRewardCallback import (
    TensorboardCallback,
)
from envs.MinimalStockTradingEnv import StockTradingEnv
from FeatureEngineer import FeatureEngineer
from YahooDownloader import YahooDownloader
from pandas import read_csv
from time import time

# Create Dirs
for dir in config.DIRS:
    Path(dir).mkdir(parents=True, exist_ok=True)

# Download this tickers
tickers = [
    "TCS.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "EICHERMOT.NS",
    "HDFCBANK.NS",
    "HINDUNILVR.NS",
]
# df = YahooDownloader(tickers=tickers, interval="1h", period="720d").process()

# Add Feature Points like RSI, MACD, etc
# fe_df = FeatureEngineer(df=df, tickers=tickers).process()
# fe_df.to_csv(f"datasets/{'-'.join(tickers)}-720d-1h.csv")

df = read_csv(f"datasets/{'-'.join(tickers)}-720d-1h.csv")
train, trade = FeatureEngineer(df=df, tickers=tickers, test_size=0.15).process()

# set_random_seed(0)
total_timesteps = 10_000
learning_rate = 0.0001
verbose = 0


# PPO
model_name = "ppo"
models_dir = "trained_models"
random_dir = int(time())
e_train_gym = StockTradingEnv(
    df=train, tickers=tickers, train_or_test="train", model_name=model_name
)
e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)
model_ppo = PPO(
    policy="MlpPolicy",
    env=e_train_gym,
    verbose=verbose,
    tensorboard_log=f"./tensorboard_log/{model_name}/",
    ent_coef=0.01,
    learning_rate=learning_rate,
)

for i in range(30):
    model_ppo.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback(),
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}-{learning_rate}-{total_timesteps}-steps",
        # eval_env=e_trade_gym,
        # eval_log_path=f"./tensorboard_log/{model_name}/",
    )
    model_ppo.save(f"{models_dir}/{model_name}/{random_dir}/{total_timesteps*i}")

# for i in range(1):
#     done = False
#     obs = e_trade_gym.reset()
#     while not done:
#         action, _ = model_ppo.predict(obs)
#         obs, rewards, done, info = e_trade_gym.step(action)
#         if done:
#             print(f"{model_name=} {info=}")
