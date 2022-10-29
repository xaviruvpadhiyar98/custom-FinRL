from pathlib import Path

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.utils import set_random_seed

import config
from callbacks.SaveOnBestTrainingRewardCallback import (
    TensorboardCallback,
)
from envs.MinimalStockTradingEnv2 import StockTradingEnv
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

set_random_seed(0)
total_timesteps = 30_000
learning_rate = 0.00001
verbose = 0


# PPO
model_name = "ppo"
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
model_ppo.learn(
    total_timesteps=total_timesteps,
    callback=TensorboardCallback(),
    tb_log_name=f"{model_name}-witheval-{learning_rate}-{total_timesteps}-steps",
    eval_env=e_trade_gym,
    eval_log_path=f"./tensorboard_log/{model_name}/",
)

done = False
obs = e_trade_gym.reset()

while not done:
    action, _ = model_ppo.predict(obs, deterministic=True)
    obs, rewards, done, info = e_trade_gym.step(action)
    if done:
        print(f"{model_name=} {info=}")


# A2C model
model_name = "a2c"
e_train_gym = StockTradingEnv(
    df=train, tickers=tickers, train_or_test="train", model_name=model_name
)
e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)


model_a2c = A2C(
    policy="MlpPolicy",
    env=e_train_gym,
    verbose=verbose,
    tensorboard_log=f"./tensorboard_log/{model_name}/",
    ent_coef=0.01,
    learning_rate=learning_rate,
)
model_a2c.learn(
    total_timesteps=total_timesteps,
    callback=TensorboardCallback(),
    tb_log_name=f"{model_name}-witheval-{learning_rate}-{total_timesteps}-steps",
    eval_env=e_trade_gym,
    eval_log_path=f"./tensorboard_log/{model_name}/",
)

done = False
obs = e_trade_gym.reset()

while not done:
    action, _ = model_a2c.predict(obs, deterministic=True)
    obs, rewards, done, info = e_trade_gym.step(action)
    if done:
        print(f"{model_name=} {info=}")

# DDPG Model
model_name = "ddpg"
e_train_gym = StockTradingEnv(
    df=train, tickers=tickers, train_or_test="train", model_name=model_name
)
e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)


model_ddpg = DDPG(
    policy="MlpPolicy",
    env=e_train_gym,
    verbose=verbose,
    tensorboard_log=f"./tensorboard_log/{model_name}/",
    learning_rate=learning_rate,
)
model_ddpg.learn(
    total_timesteps=total_timesteps,
    callback=TensorboardCallback(),
    tb_log_name=f"{model_name}-witheval-{learning_rate}-{total_timesteps}-steps",
    eval_env=e_trade_gym,
    eval_log_path=f"./tensorboard_log/{model_name}/",
)

done = False
obs = e_trade_gym.reset()

while not done:
    action, _ = model_ddpg.predict(obs, deterministic=True)
    obs, rewards, done, info = e_trade_gym.step(action)
    if done:
        print(f"{model_name=} {info=}")

# TD3 Model
model_name = "td3"
e_train_gym = StockTradingEnv(
    df=train, tickers=tickers, train_or_test="train", model_name=model_name
)
e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)


model_td3 = TD3(
    policy="MlpPolicy",
    env=e_train_gym,
    verbose=verbose,
    tensorboard_log=f"./tensorboard_log/{model_name}/",
    learning_rate=learning_rate,
)
model_td3.learn(
    total_timesteps=total_timesteps,
    callback=TensorboardCallback(),
    tb_log_name=f"{model_name}-witheval-{learning_rate}-{total_timesteps}-steps",
    eval_env=e_trade_gym,
    eval_log_path=f"./tensorboard_log/{model_name}/",
)

done = False
obs = e_trade_gym.reset()

while not done:
    action, _ = model_td3.predict(obs, deterministic=True)
    obs, rewards, done, info = e_trade_gym.step(action)
    if done:
        print(f"{model_name=} {info=}")

# SAC Model
model_name = "sac"
e_train_gym = StockTradingEnv(
    df=train, tickers=tickers, train_or_test="train", model_name=model_name
)
e_trade_gym = StockTradingEnv(
    df=trade, tickers=tickers, train_or_test="test", model_name=model_name
)

model_sac = SAC(
    policy="MlpPolicy",
    env=e_train_gym,
    verbose=verbose,
    tensorboard_log=f"./tensorboard_log/{model_name}/",
    learning_rate=learning_rate,
    ent_coef="auto_0.1",
)
model_sac.learn(
    total_timesteps=total_timesteps,
    callback=TensorboardCallback(),
    tb_log_name=f"{model_name}-witheval-{learning_rate}-{total_timesteps}-steps",
    eval_env=e_trade_gym,
    eval_log_path=f"./tensorboard_log/{model_name}/",
)

done = False
obs = e_trade_gym.reset()
while not done:
    action, _ = model_sac.predict(obs, deterministic=True)
    obs, rewards, done, info = e_trade_gym.step(action)
    if done:
        print(f"{model_name=} {info=}")

# rewards=-8.77685546875e-06 for 5000 epoch
# rewards=-3.86962890625e-05 for 10000 epoch
# rewards=0.0 for 12000 epoch
# rewards=0.0 for 15000 epoch
# rewards=0.0 for 20000 epoch
