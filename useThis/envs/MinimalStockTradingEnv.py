from typing import List

import numpy as np
import pandas as pd
from gym import Env, spaces


class StockTradingEnv(Env):
    # defaults
    # hmax = how much to buy in a lot
    HMAX = 10
    INITIAL_AMOUNT = 100_000
    BUY_COST = 0.001
    SELL_COST = 0.001
    REWARD_SCALING = 1e-4

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        model_name: str,
        train_or_test: str = "train",
    ):
        self.df = df
        self.tickers = tickers
        self.model_name = model_name
        self.train_or_test = train_or_test

        self.number_of_stocks = len(tickers)
        self.technical_indicators = self.df.columns.to_list()[7:]
        self.number_of_technical_indicator_used = len(self.technical_indicators)

        # 1 -> total amount column
        # 2 * numberOfStocks -> number_of_shares_bought_and_sell and its price column for each ticker
        # totalNumberOfTechnicalIndicatorUsed * numberOfStocks -> RSI, MACD, etc for each ticker
        self.total_number_of_feature_points = (
            1
            + (2 * self.number_of_stocks)
            + self.number_of_technical_indicator_used * self.number_of_stocks
        )

        # action space Box = [-1, -1, -1] * per each stock as an action
        # -1 to sell, 0 hold, 1 buy
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.number_of_stocks,),
            dtype=np.float32,
        )

        # we do not know the observational space yet so putting it to inf per each stock
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_number_of_feature_points,),
            dtype=np.float32,
        )

    def step(self, actions):
        # check whether we are end of the dataframe if so done it
        done = self.index == self.df.index[-1]
        if done:
            df_results = pd.DataFrame(self.total_assets)
            df_results["Daily Return"] = df_results["Ending Amount"].pct_change(1)

            df_results.to_csv(
                f"results/{self.model_name}-{self.train_or_test}-assets{'-'.join(self.tickers)}-1h-150d.csv",
                index=False,
            )
            sharpe = (
                (252**0.5)
                * df_results["Daily Return"].mean()
                / df_results["Daily Return"].std()
            )
            return (
                self.state,
                self.reward,
                True,
                {
                    "Sharpe": sharpe,
                    "Max Drawdown": df_results["Daily Return"].min(),
                    "Max Upwards": df_results["Daily Return"].max(),
                    "Ending Amount": df_results["Ending Amount"].iloc[-1],
                },
            )

        begin_total_asset = self._get_asset_memory()
        for i, action in enumerate(actions):
            if action > 0:
                self.buy(i)
            else:
                self.sell(i)

        end_total_asset = self._get_asset_memory()
        self.reward = int((end_total_asset - begin_total_asset) * 0.2)
        if self.reward > 0:
            self.reward += 10
        else:
            self.reward -= 20

        result = {
            "Day": self.data.date.iloc[0],
            "Ending Amount": end_total_asset,
            "Ending Reward": self.reward,
        }
        for i in range(self.number_of_stocks):
            result[f"{self.tickers[i]} Price"] = self.state[1 + i]
            result[f"{self.tickers[i]} Buy/Sell"] = self.state[
                1 + self.number_of_stocks + i
            ]

        self.total_assets.append(result)
        self.state = self.generate_state()
        return self.state, self.reward, False, {}

    def reset(self):
        self.buy_cost_pct: List[float] = [self.BUY_COST] * self.number_of_stocks
        self.sell_cost_pct: List[float] = [self.SELL_COST] * self.number_of_stocks
        self.index = 0
        self.reward = 0
        self.total_assets = []
        self.state = self.generate_state(True)
        return self.state

    def generate_state(self, generate: bool = False):
        if generate:
            initial_amount = [self.INITIAL_AMOUNT]
            num_stock_shares = [0] * self.number_of_stocks
            self.index = self.df.index.values[0]
        else:
            initial_amount = [self.state[0]]
            num_stock_shares = list(
                self.state[
                    (1 + self.number_of_stocks) : (1 + self.number_of_stocks * 2)
                ]
            )
            self.index += 1
        self.data = self.df.loc[self.index]
        prices = self.data.close.to_list()

        tech_list = sum(
            (self.data[tech].to_list() for tech in self.technical_indicators),
            [],
        )
        return np.array(
            initial_amount + prices + num_stock_shares + tech_list, dtype=np.float32
        )

    def buy(self, index):

        buy_amount = self.state[1 + index] * self.HMAX * (1 + self.buy_cost_pct[index])
        if self.state[0] > buy_amount:
            self.state[0] -= buy_amount
            self.state[1 + self.number_of_stocks + index] += self.HMAX

    def sell(self, index):

        sell_amount = (
            self.state[1 + index] * self.HMAX * (1 + self.sell_cost_pct[index])
        )

        if self.state[1 + self.number_of_stocks + index] >= self.HMAX:
            self.state[0] += sell_amount
            self.state[1 + self.number_of_stocks + index] -= self.HMAX

    def _get_asset_memory(self):
        initial_amount = self.state[0]
        close_prices = np.array(self.state[1 : (self.number_of_stocks + 1)])
        num_stock_shares = np.array(
            self.state[(self.number_of_stocks + 1) : (self.number_of_stocks * 2 + 1)]
        )
        return initial_amount + sum(close_prices * num_stock_shares)


if __name__ == "__main__":
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from stable_baselines3.common.env_checker import check_env

    df = read_csv("datasets/fe-df-tcs-bajfinance-intellect-1h-150d.csv")
    tickers: List[str] = df.tic.unique().tolist()
    df = df.reset_index()
    df.index = df.date.factorize()[0]

    # Check the your env is valid, Warning is fine, ment to be ignored
    check_env(StockTradingEnv(df=df, tickers=tickers))
