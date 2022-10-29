from pandas import read_csv

df = read_csv("ind_nifty50list.csv")
df["Symbol"] = df["Symbol"] + ".NS"
df["Symbol"].to_csv("nifty50symbols.csv", index=False)
