from environments.stock_trading_env import StockTradingEnv
from configs.create_dirs import create_dirs
from configs.config import *
import pandas as pd

from stable_baselines3.common.monitor import Monitor

import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

from stable_baselines3 import PPO


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


def calculate_sharpe_and_get_ending_amount(info):
    last_key = next(reversed(info))
    if not last_key[:4].isdigit():
        info.pop(last_key)
    df = pd.DataFrame.from_dict(info, orient="index")
    df["Daily Return"] = df["current_holdings"].pct_change(1)
    stock = df["Daily Return"]
    sharpe = qs.stats.sharpe(stock)
    print(df.tail(50).to_markdown())
    ending_amount = df["current_holdings"].values[-1]
    return (sharpe, ending_amount)


def inference():
    trade_env = Monitor(StockTradingEnv(trade_df, TICKERS, TECHNICAL_INDICATORS))
    trade_env.reset()
    model_path = Path(f"{TRAINED_MODEL_DIR}/BEST_MODEL/best_model002.zip")
    model_ppo = PPO.load(
        path=model_path.as_posix(),
        env=trade_env,
        custom_objects={"tensorboard_log": None},
    )
    obs = trade_env.reset()
    while True:
        action, _states = model_ppo.predict(obs)
        obs, rewards, dones, info = trade_env.step(action)
        if dones:
            print(f"{calculate_sharpe_and_get_ending_amount(info)=}")
            break


def main():
    inference()


if __name__ == "__main__":
    main()
