import yfinance as yf


def yahoo_downloader(tickers, period, interval):
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="Ticker",
        auto_adjust=True,
        prepost=True,
    )
    df = df.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index(level=1)
    df = df.reset_index()
    return df
