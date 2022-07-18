from typing import List, Optional, Union

import pandas as pd
import yfinance as yf


class YahooDownloader:
    def __init__(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "5y",
        interval: Optional[str] = "1d",
    ):

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.df = pd.DataFrame()

    def download_data(self):
        # documentation -> https://pypi.org/project/yfinance/

        if self.start_date or self.end_date:
            self.df = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by="Ticker",
                auto_adjust=True,
                prepost=True,
            )
        else:
            self.df = yf.download(
                tickers=self.tickers,
                period=self.period,
                interval=self.interval,
                group_by="Ticker",
                auto_adjust=True,
                prepost=True,
            )

        return self

    def format_multi_level_columns(self):
        # dealing-with-multi-level-column-names
        # https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801

        if len(self.tickers) > 1:
            self.df = (
                self.df.stack(level=0)
                .rename_axis(["Date", "Ticker"])
                .reset_index(level=1)
            )
        else:
            self.df["Ticker"] = self.tickers[0]
        return self

    def rename_columns(self):
        self.df = self.df.reset_index()
        self.df = self.df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Ticker": "tic",
            }
        )
        return self

    def add_day_column(self):
        self.df["day"] = self.df["date"].dt.dayofweek
        return self

    def format_date_to_str(self):
        self.df["date"] = self.df.date.dt.strftime("%Y-%m-%d")
        return self

    def drop_missing_data(self):
        old_count = self.df.shape[0]
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        new_count = self.df.shape[0]
        if new_count != old_count:
            print(f"Total {old_count - new_count} rows dropped!")

        print(f"Shape of DataFrame: {self.df.shape}")
        self.df = self.df.sort_values(
            by=["date", "tic"], ignore_index=True
        ).reset_index(drop=True)
        return self

    def post_process(self) -> pd.DataFrame:
        """
        Rename columns to lower case
        Adds day column
        Converts date to str
        Drop missing value
        """
        self.format_multi_level_columns()
        self.rename_columns()
        self.add_day_column()
        self.format_date_to_str()
        self.drop_missing_data()
        return self.df

    def save_to_csv(self, filename: str):
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        return self


def main():
    tickers = ["BAJFINANCE.NS", "WIPRO.NS"]
    df = YahooDownloader(tickers=tickers).download_data().post_process()

    print(df.head(10))


if __name__ == "__main__":
    main()
