from copy import copy
from gym import Env, spaces
import numpy as np
from collections import deque
import quantstats as qs

qs.extend_pandas()


class StockTradingEnv(Env):
    HMAX = 10
    INITIAL_AMOUNT = 100_000
    BUY_COST = SELL_COST = 0.001

    def __init__(self, arrays, tickers, features):
        self.arrays = arrays
        self.tickers = tickers
        self.features = features
        self.num_of_features = len(features)
        self.num_of_tickers = len(tickers)
        self.action_space = spaces.Box(
            -1, 1, shape=(self.num_of_tickers,), dtype=np.int32
        )
        self.obs_formula = 1 + (self.num_of_tickers * (1 + self.num_of_features + 1))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_formula,),
            dtype=np.float32,
        )
        self.last_2_holdings = deque(maxlen=2)

    def reset(self):
        self.reward = 0.0
        self.info = {}
        self.state = self.generate_state(reset=True)
        return self.state

    def step(self, actions):
        actions = np.rint(actions)
        done = bool(self.index == len(self.arrays) - 1)
        if done:
            return (self.state, self.reward, done, self.info)

        for ticker_index, action in enumerate(actions):
            if action == 0:
                continue

            close_price_index = 1 + ticker_index * (1 + self.num_of_features + 1)
            price_per_share = self.state[close_price_index]
            number_of_shares_index = (
                1
                + ticker_index * (1 + self.num_of_features + 1)
                + (1 + self.num_of_features)
            )
            if action == 1.0:
                self.buy(price_per_share, number_of_shares_index)
            if action == -1.0:
                self.sell(price_per_share, number_of_shares_index)

        current_holdings, shares_available = self.get_holdings()
        self.last_2_holdings.append(current_holdings)
        previous_holdings = self.last_2_holdings[0]

        self.info[self.index] = {
            "current_holdings": current_holdings,
            "shares": shares_available,
        }

        self.reward = self.calculate_reward(previous_holdings, current_holdings)
        self.state = self.generate_state()
        return (self.state, self.reward, done, self.info)

    def buy(self, price_per_share, number_of_shares_index):
        shares = min(self.state[0] // price_per_share, self.HMAX)
        buy_prices_with_commission = (price_per_share * (1 + self.BUY_COST)) * shares
        self.state[0] -= buy_prices_with_commission
        self.state[number_of_shares_index] += shares

    def sell(self, price_per_share, number_of_shares_index):
        shares = self.state[number_of_shares_index]
        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (
                price_per_share * (1 + self.SELL_COST)
            ) * shares
            self.state[0] += sell_prices_with_commission
            self.state[number_of_shares_index] -= shares

    def get_holdings(self):
        close_prices = self.state[1 :: 1 + self.num_of_features + 1]
        shares_available = self.state[
            1 + 1 + self.num_of_features :: 1 + self.num_of_features + 1
        ]

        holdings = np.sum(np.multiply(close_prices, shares_available)) + self.state[0]
        return holdings, shares_available

    def generate_state(self, reset=False):
        if not reset:
            self.index += 1
        else:
            self.index = 0

        vals = self.arrays[self.index].reshape(-1)
        state = np.array([self.INITIAL_AMOUNT])
        state = np.append(state, vals)

        if reset:
            return state

        return self.update_new_states_with_old_values(self.state, state)

    def update_new_states_with_old_values(self, old_state, new_state):
        old_state = self.state
        new_state[0] = old_state[0]

        start = 1 + 1 + self.num_of_features
        end = 1 + self.num_of_features + 1

        new_state[start::end] = copy(old_state[start::end])

        return new_state

    def calculate_reward(self, previous_holdings, current_holdings):
        change_in_holdings = current_holdings - previous_holdings
        if change_in_holdings > 0:
            return change_in_holdings * 0.1
        return change_in_holdings * -0.1


def calculate_sharpe_and_get_ending_amount(info):
    last_key = next(reversed(info))
    if not isinstance(last_key, int):
        info.pop(last_key)

    current_holdings = np.array([value["current_holdings"] for value in info.values()])
    returns = (current_holdings[1:] - current_holdings[:-1]) / current_holdings[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns)
    return sharpe_ratio, current_holdings[-1]


if __name__ == "__main__":
    import pandas as pd
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from pathlib import Path
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from pytorch_lightning import seed_everything

    seed_everything(20)
    # # Add your Stock here with time period from current to last n days
    TICKERS = [
        "TCS.NS",
        "BAJFINANCE.NS",
        "WIPRO.NS",
        "CANBK.NS",
        "IGL.NS",
        "TATACONSUM.NS",
    ]
    TICKERS.sort()

    # Dont Modify the below for now
    TECHNICAL_INDICATORS = [
        "EMA_8",
        "EMA_21",
        "RSI_14",
        "BBAND_UP",
        "BBAND_MID",
        "BBAND_DOWN",
    ]

    TENSORBOARD_LOG_DIR = "tensorboard_log"
    MODEL_NAME = "PPO"
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

    num_cpu = 2
    n_steps = 8000
    env_id = StockTradingEnv
    env_kwargs = {
        "arrays": train_arrays,
        "tickers": TICKERS,
        "features": TECHNICAL_INDICATORS,
    }

    train_env = make_vec_env(env_id=env_id, n_envs=num_cpu, env_kwargs=env_kwargs)
    model_ppo = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        tensorboard_log=Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}"),
        batch_size=(num_cpu * n_steps) // 2,
        n_steps=n_steps,
    )
    # model_ppo.learn(10_000 * num_cpu, progress_bar=True)
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, TECHNICAL_INDICATORS))
    obs = trade_env.reset()
    while True:
        action, _states = model_ppo.predict(obs)
        (state, reward, done, info) = trade_env.step(action)
        if done:
            (sharpe, ending_amount) = calculate_sharpe_and_get_ending_amount(info)
            print(sharpe, ending_amount)
            break
