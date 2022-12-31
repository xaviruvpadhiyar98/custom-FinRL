from talib import EMA, RSI, BBANDS, MA_Type
import quantstats as qs
import pandas as pd

qs.extend_pandas()


def add_technical_indicators(df, tickers):
    for ticker in tickers:
        close_value = df[df["Ticker"] == ticker]["Close"].values
        df.loc[df["Ticker"] == ticker, "EMA_8"] = EMA(close_value, timeperiod=8)
        df.loc[df["Ticker"] == ticker, "EMA_21"] = EMA(close_value, timeperiod=21)
        df.loc[df["Ticker"] == ticker, "RSI_14"] = RSI(close_value, timeperiod=14)

        upper, middle, lower = BBANDS(close_value, matype=MA_Type.T3)
        df.loc[df["Ticker"] == ticker, "BBAND_UP"] = upper
        df.loc[df["Ticker"] == ticker, "BBAND_MID"] = middle
        df.loc[df["Ticker"] == ticker, "BBAND_DOWN"] = lower

    return df


def clean_data(df, tickers):
    df = df.dropna(axis=0).reset_index(drop=True)
    df.index = df["Date"].factorize()[0]
    df = df.loc[df.groupby(level=0).count()["Date"] == len(tickers)]
    df = df.dropna(axis=0).reset_index(drop=True)
    df.index = df["Date"].factorize()[0]
    return df


def add_buy_sell_hold_single_column(df):
    df["Buy/Sold/Hold"] = 0.0
    return df


def calculate_sharpe_and_get_ending_amount(info):
    try:
        last_key = next(reversed(info[0]))
        if not last_key[:4].isdigit():
            info[0].pop(last_key)
            # print(info)

        df = pd.DataFrame.from_dict(info[0], orient="index")
        df["Daily Return"] = df["current_holdings"].pct_change(1)
        stock = df["Daily Return"]
        sharpe = qs.stats.sharpe(stock)
        ending_amount = df["current_holdings"].values[-1]
        return (sharpe, ending_amount)
    except Exception as e:
        from traceback import print_exc
        import sys

        # print(info)

        print(print_exc())
        sys.exit()


# import numpy as np

# a = [
#     {
#         "2022-11-10 15:15:00+05:30": {
#             "current_holdings": 99927.02350006104,
#             "shares": np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
#         },
#         "terminal_observation": np.array(
#             [
#                 5.58891397e04,
#                 6.57520020e03,
#                 6.59306835e03,
#                 6.54991305e03,
#                 5.48848436e01,
#                 6.73526546e03,
#                 6.64362284e03,
#                 6.55198021e03,
#                 0.00000000e00,
#                 3.33049988e02,
#                 3.27783699e02,
#                 3.21151948e02,
#                 7.77638288e01,
#                 3.35523756e02,
#                 3.29893261e02,
#                 3.24262766e02,
#                 5.80000000e01,
#                 4.14899994e02,
#                 4.12738769e02,
#                 4.11329829e02,
#                 5.66022274e01,
#                 4.17266954e02,
#                 4.13782136e02,
#                 4.10297317e02,
#                 2.80000000e01,
#                 7.71000000e02,
#                 7.71132660e02,
#                 7.73159025e02,
#                 4.46647858e01,
#                 7.76326068e02,
#                 7.71460107e02,
#                 7.66594146e02,
#                 1.00000000e01,
#                 3.25460010e03,
#                 3.26853554e03,
#                 3.26451416e03,
#                 4.61639014e01,
#                 3.31284028e03,
#                 3.27810879e03,
#                 3.24337729e03,
#                 0.00000000e00,
#                 3.92750000e02,
#                 3.92775998e02,
#                 3.90361930e02,
#                 6.32009096e01,
#                 3.96175087e02,
#                 3.94324246e02,
#                 3.92473405e02,
#                 0.00000000e00,
#             ]
#         ),
#     }
# ]
# # last_key = next(reversed(a[0]))
# # if not last_key[:4].isdigit():
# #     a[0].pop(last_key)

# print(calculate_sharpe_and_get_ending_amount(a))
