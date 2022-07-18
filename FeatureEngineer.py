from typing import List

import pandas as pd
from talib import BBANDS, CCI, DX, EMA, MACD, RSI, SMA, MA_Type

from YahooDownloader import YahooDownloader


class FeatureEngineer:

    INDICATORS = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list: List[str] = INDICATORS,
        use_technical_indicator: bool = True,
        use_vix: bool = False,
        use_turbulence: bool = False,
        use_user_defined_feature: bool = False,
    ):
        self.df = df
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.use_user_defined_feature = use_user_defined_feature

    def add_technical_indicator(self):
        if self.use_technical_indicator:
            self.df["sma_30"] = SMA(self.df.close, timeperiod=30)
            self.df["sma_60"] = SMA(self.df.close, timeperiod=60)
            self.df["ema_8"] = EMA(self.df.close, timeperiod=8)
            self.df["ema_21"] = EMA(self.df.close, timeperiod=21)
            macd, macdsignal, macdhist = MACD(
                self.df.close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            self.df["macd"] = macd

            self.df["rsi_30"] = RSI(self.df.close, timeperiod=30)
            self.df["rsi_70"] = RSI(self.df.close, timeperiod=70)

            self.df["dx_30"] = DX(
                self.df.high, self.df.low, self.df.close, timeperiod=30
            )
            self.df["cci_30"] = CCI(
                self.df.high, self.df.low, self.df.close, timeperiod=30
            )

            upper, middle, lower = BBANDS(self.df.close, matype=MA_Type.T3)
            self.df["boll_ub"] = upper
            self.df["boll_lb"] = lower
            self.df = self.df.dropna()
            self.df = self.df.reset_index(drop=True)

        return self

    def add_vix(self):
        if self.use_vix:
            df_vix = (
                YahooDownloader(
                    tickers=["^VIX"],
                    start_date=self.df.date.iloc[0],
                    end_date=self.df.date.iloc[-1],
                )
                .download_data()
                .post_process()
            )
            df_vix = df_vix[["date", "close"]]
            self.df_vix = self.df_vix.rename(
                columns={
                    "date": "date",
                    "close": "vix",
                }
            )
            self.df = self.df.merge(self.df_vix, on="date")
        return self

    def add_turbulence(self):
        """Not Implemented"""
        return self

    def add_user_defined_feature(self):
        if self.use_user_defined_feature:
            self.df["daily_return"] = self.df.close.pct_change(1)
        return self

    def post_process(self):
        """
        add_technical_indicator
        add_vix
        add_turbulence
        add_user_defined_feature
        """
        self.add_technical_indicator()
        self.add_vix()
        self.add_user_defined_feature()
        self.add_turbulence()
        self.df.index = self.df.date.factorize()[0]
        return self.df

    def clean_data(self):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        self.df.index = self.df.date.factorize()[0]
        merged_closes = self.df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        return self.df
