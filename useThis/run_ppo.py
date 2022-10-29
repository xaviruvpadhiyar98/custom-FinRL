from pathlib import Path

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.utils import set_random_seed

import config
from callbacks.SaveOnBestTrainingRewardCallback import (
    TensorboardCallback,
)
from envs.MinimalStockTradingEnv import StockTradingEnv
from FeatureEngineer import FeatureEngineer
from YahooDownloader import YahooDownloader
from pandas import read_csv

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
train, trade = FeatureEngineer(df=df, tickers=tickers).process()

# set_random_seed(0)
total_timesteps = 30_000
learning_rate = 0.00001
verbose = 0


# PPO
model_name = "ppo"
model_path = f"trained_models/ppo/1667035003/170000.zip"

e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)
e_trade_gym.reset()

model_ppo = PPO.load(model_path, e_trade_gym)

for i in range(1):
    obs = e_trade_gym.reset()
    done = False
    while not done:
        action, _states = model_ppo.predict(obs)
        obs, rewards, done, info = e_trade_gym.step(action)
        if done:
            print(f"{info=}")
