from typing import List

import pandas as pd
from talib import BBANDS, CCI, DX, EMA, MACD, RSI, SMA, MA_Type


class FeatureEngineer:
    def __init__(
        self, df: pd.DataFrame, tickers: List[str], test_size: float = 0.2
    ) -> None:
        self.df = df
        self.tickers = tickers
        self.use_user_defined_feature: bool = True
        self.new_df: pd.DataFrame = pd.DataFrame()
        self.test_size = test_size

    def add_user_defined_feature(self):
        if self.use_user_defined_feature:
            for tic in self.tickers:
                one_tic_df = self.df[self.df.tic == tic].copy()
                one_tic_df["sma_30"] = SMA(one_tic_df.close, timeperiod=30)
                one_tic_df["sma_60"] = SMA(one_tic_df.close, timeperiod=60)
                one_tic_df["ema_8"] = EMA(one_tic_df.close, timeperiod=8)
                one_tic_df["ema_21"] = EMA(one_tic_df.close, timeperiod=21)
                macd, macdsignal, macdhist = MACD(
                    one_tic_df.close, fastperiod=12, slowperiod=26, signalperiod=9
                )
                one_tic_df["macd"] = macd

                one_tic_df["rsi_14"] = RSI(one_tic_df.close, timeperiod=14)

                one_tic_df["dx_30"] = DX(
                    one_tic_df.high, one_tic_df.low, one_tic_df.close, timeperiod=30
                )
                one_tic_df["cci_30"] = CCI(
                    one_tic_df.high, one_tic_df.low, one_tic_df.close, timeperiod=30
                )

                upper, middle, lower = BBANDS(one_tic_df.close, matype=MA_Type.T3)
                one_tic_df["boll_ub"] = upper
                one_tic_df["boll_lb"] = lower
                one_tic_df["boll_mid"] = middle

                one_tic_df = one_tic_df.dropna()
                one_tic_df = one_tic_df.reset_index(drop=True)
                self.new_df = pd.concat([self.new_df, one_tic_df])
        return self

    def format_data(self):
        self.new_df = self.new_df.sort_values(["date", "tic"])
        self.new_df = self.new_df.reset_index(drop=True)
        self.new_df = self.new_df.set_index("date")
        filter = (
            self.new_df.groupby([self.new_df.index]).nunique().eq(len(self.tickers))
        )
        self.new_df = self.new_df.where(filter).dropna()
        self.new_df = self.new_df.reset_index()
        self.new_df.index = self.new_df.date.factorize()[0]
        return self

    def split_train_test_data(self):
        last_index = self.new_df.index.values[-1]
        test_size = int(self.test_size * last_index)
        train_size = last_index - test_size
        assert train_size + test_size == last_index
        return self.new_df.loc[:train_size], self.new_df.loc[train_size + 1 :]

    def process(self):
        """
        add_user_defined_feature
        """
        self.add_user_defined_feature()
        self.format_data()
        return self.split_train_test_data()
