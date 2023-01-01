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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)

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


def train(config=None):
    with wandb.init(config=config, sync_tensorboard=True) as run:
        try:

            train_env = Monitor(
                StockTradingEnv(train_df, TICKERS, TECHNICAL_INDICATORS)
            )
            trade_env = Monitor(
                StockTradingEnv(trade_df, TICKERS, TECHNICAL_INDICATORS)
            )
            # train_env = SubprocVecEnv(train_env)
            #     SubprocVecEnv(
            #         [
            #             StockTradingEnv(train_df, TICKERS, TECHNICAL_INDICATORS)
            #             for _ in range(1)
            #         ]
            #     )
            # )
            # trade_env = Monitor(
            #     SubprocVecEnv(
            #         [
            #             StockTradingEnv(train_df, TICKERS, TECHNICAL_INDICATORS)
            #             for _ in range(1)
            #         ]
            #     )
            # )

            model_params = wandb.config
            model_ppo = PPO(
                policy="MlpPolicy",
                env=train_env,
                verbose=0,
                tensorboard_log=Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}"),
                **model_params,
            )

            callback_after_eval = WandDBLogCallback(verbose=0)

            checkpoint_callback = CheckpointCallback(
                save_freq=MODEL_SAVE_AT_N_STEPS,
                save_path=f"{TRAINED_MODEL_DIR}/{MODEL_NAME}/wandb/{run.id}/",
                name_prefix="rl_model_ppo",
                save_replay_buffer=True,
                save_vecnormalize=True,
                verbose=0,
            )
            eval_callback = EvalCallback(
                trade_env,
                best_model_save_path=f"{TRAINED_MODEL_DIR}/BEST_MODEL/{MODEL_NAME}/wandb/{run.id}/",
                log_path=f"{LOGS_DIR}",
                eval_freq=MODEL_SAVE_AT_N_STEPS,
                deterministic=True,
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
                tb_log_name=run.id,
                callback=callback,
                progress_bar=True,
            )

        except Exception:
            print(traceback.print_exc(), file=sys.stderr)


def main():
    sweep_id = wandb.sweep(WANDB_SWEEP_CONFIG, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, train, count=1)


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
