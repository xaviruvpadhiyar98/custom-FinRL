from downloaders.yahoo_downloader import yahoo_downloader
from feature_engineering.technical_indicators import (
    add_technical_indicators,
    clean_data,
    add_buy_sell_hold_single_column,
    calculate_sharpe_and_get_ending_amount,
)
from callbacks.wand_tensorboard_callback import TensorboardCallback
from environments.stock_trading_env import StockTradingEnv
from configs.create_dirs import create_dirs
from configs.config import *
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
import wandb

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


def train(config=None):
    with wandb.init(config=config) as run:
        train_env = DummyVecEnv(
            [lambda: StockTradingEnv(train_df, TICKERS, TECHNICAL_INDICATORS)]
        )
        train_env = VecFrameStack(train_env, 1)
        model_params = wandb.config
        model_ppo = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=0,
            tensorboard_log=Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}"),
            **model_params,
            # device="cuda",
        )

        for i in range(STEPS):
            model_ppo.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=run.id,
                callback=TensorboardCallback(
                    verbose=0,
                    model_save_path=f"{TRAINED_MODEL_DIR}/{MODEL_NAME}/wandb/{run.id}/",
                    model_save_freq=MODEL_SAVE_AT_N_STEPS,
                ),
            )

        model_path = sorted(
            list(Path(f"{TRAINED_MODEL_DIR}/{MODEL_NAME}/wandb/{run.id}/").rglob("*")),
            key=lambda x: int(x.stem),
            reverse=True,
        )[-1]
        # wandb.log()
        # print(model_path)

        trade_env = DummyVecEnv(
            [lambda: StockTradingEnv(trade_df, TICKERS, TECHNICAL_INDICATORS)]
        )
        trade_env = VecFrameStack(trade_env, 1)
        model_ppo = PPO.load(model_path)
        obs = trade_env.reset()
        while True:
            action, _states = model_ppo.predict(obs)
            obs, rewards, dones, info = trade_env.step(action)
            if dones:
                sharpe, ending_amount = calculate_sharpe_and_get_ending_amount(info)
                wandb.log({"Sharpe": sharpe, "Ending Amount": ending_amount})
                break


def main():
    sweep_id = wandb.sweep(WANDB_SWEEP_CONFIG, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, train, count=5)


if __name__ == "__main__":
    from traceback import print_exc
    from sys import stderr

    try:
        main()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(print_exc(), file=stderr)
        exit(1)


# train_env = DummyVecEnv(
#     [lambda: StockTradingEnv(train_df, TICKERS, TECHNICAL_INDICATORS)]
# )
# train_env = VecFrameStack(train_env, 1)


# model_params = {
#     # "batch_size": 2048 * 2,
#     "n_steps": 4096,
#     "gamma": 0.8115992973499895,
#     "learning_rate": 8.795585311974417e-05,
#     "clip_range": 0.22022300234864176,
#     "gae_lambda": 0.9637265320016442,
# }
# train(model_params)
# model_ppo = PPO(
#     policy="MlpPolicy",
#     env=train_env,
#     verbose=0,
#     tensorboard_log=f"./tensorboard_log/ppo/",
#     **model_params,
#     device="cuda",
# )


# for i in range(STEPS):
#     model_ppo.learn(
#         total_timesteps=TIMESTEPS,
#         reset_num_timesteps=False,
#         tb_log_name="profiling-logging",
#         callback=TensorboardCallback(
#             verbose=0,
#             model_save_path=f"{TRAINED_MODEL_DIR}/{MODEL_NAME}/wandb/{RANDOM_DIR}/",
#             model_save_freq=MODEL_SAVE_AT_N_STEPS,
#         ),
#     )
