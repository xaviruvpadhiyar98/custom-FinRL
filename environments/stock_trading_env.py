from copy import copy
from gym import Env, spaces
import numpy as np
from collections import deque


class StockTradingEnv(Env):
    HMAX = 10
    INITIAL_AMOUNT = 100_000
    BUY_COST = SELL_COST = 0.001

    def __init__(self, df, tickers, features):
        self.df = df
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
        done = bool(self.index == self.df.index[-1])
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
        #         if not self.info:
        #             previous_holdings = current_holdings
        #         else:
        #             previous_holdings = next(reversed(self.info.values()))["current_holdings"]

        self.info[str(self.data["Date"].values[0])] = {
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
            self.index = self.df.index.values[0]

        self.data = self.df.loc[self.index]
        state = np.array([self.INITIAL_AMOUNT])

        vals = self.data[["Close"] + self.features + ["Buy/Sold/Hold"]].values.reshape(
            -1
        )
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
