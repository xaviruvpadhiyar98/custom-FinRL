from __future__ import annotations

from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

from stable_baselines3.common.logger import configure


class StockTradingEnv(Env):
    """OpenAI gym ENV fir stock trading env"""

    metadata: dict[str, List[str]] = {"render.modes": ["human"]}

    """OpenAI StockTrading Env
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hmax: int = 100,
        initial_amount: int = 100_000,
        buy_cost: float = 0.001,
        sell_cost: float = 0.001,
        reward_scaling: float = 1e-4,
        terminal: bool = False,
        make_plots: bool = False,
        print_verbosity: int = 10,
        initial: bool = True,
        previous_state: List = [],
        model_name: str = "anything",
        mode: str = "human",
        iteration: int = 0,
    ) -> None:
        """init

        Args:
            df (pd.DataFrame): train or trade pandas dataframe
            hmax (int, optional): total amount of stocks to buy and sell. Defaults to 100.
            initial_amount (int, optional): starting value. Defaults to 100_000.
            buy_cost (float, optional): buying cost of a stock. Defaults to 0.001.
            sell_cost (float, optional): selling cost of a stock. Defaults to 0.001.
            reward_scaling (float, optional): reward for each buy and sell. Defaults to 1e-4.
            terminal (bool, optional): True or False whether one iteration from start to end
                                       in train or trade df is complete or not. Defaults to False.
            make_plots (bool, optional): Enable Plotting for one iteration . Defaults to False.
            print_verbosity (int, optional): verbose level uses MOD operator. Defaults to 10.
            initial (bool, optional): whether we are starting or resuming. Defaults to True.
            previous_state (List, optional): resume from previous state if `intial` is false. Defaults to [].
            model_name (str, optional): name of model can be anything use to save graph and all. Defaults to "".
            mode (str, optional): render mode human or bot or anything. Defaults to "".
            iteration (int, optional): counter for iteration. Defaults to "".
        """

        # init
        self.df = df
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.reward_scaling = reward_scaling
        self.terminal = terminal
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Post init -> values we require later

        # total unique rows in a dataframe, unique coz we use tickers.
        self.total_unique_rows: int = len(self.df.index.unique()) + self.df.index[0]

        # all the tickers
        self.tics: List[str] = self.df.tic.unique().tolist()

        # process stock_dimension, buy_sell_cost, num_stock_shares
        self.stock_dimension: int = len(self.tics)
        self.buy_cost_pct: List[float] = [self.buy_cost] * self.stock_dimension
        self.sell_cost_pct: List[float] = [self.sell_cost] * self.stock_dimension
        self.num_stock_shares: List[int] = [0] * self.stock_dimension

        # process all technical indicator and find total len of all
        # here 8 because -> first 8 columns are ->
        # ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
        self.tech_indicator_list: List[str] = self.df.columns.to_list()[8:]
        self.total_unique_tech_indicators = len(self.tech_indicator_list)

        # take first row for tickers
        self.index: int = self.df.index[0]
        self.data: pd.DataFrame = self.df.loc[self.index, :]

        # How state_space is calculated - >
        # [HMAX]
        # + ["DATE", "CLOSE"] * len(TICKERS)
        # + len(['sma_30', 'sma_60', 'ema_8', 'ema_21', 'macd', 'rsi_30', 'rsi_70','dx_30', 'cci_30', 'boll_ub', 'boll_lb']) * len(TICKERS)
        self.state_space: List[float] = (
            1
            + (2 * self.stock_dimension)
            + (len(self.tech_indicator_list) * self.stock_dimension)
        )

        # Action space ->
        # -1, 0, 1 represent selling, holding, and buying one share
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.stock_dimension,)
        )

        # Observation space will be the similar to state_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        # initialize the starting state for single or multiple tickers
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # memorize all the total balance changes
        # the initial total asset is calculated by
        # cash + sum (num_share_stock_i * price_stock_i)
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dimension], dtype=object)
            )
        ]

        self.rewards_memory = []
        self.actions_memory = []
        # we need sometimes to preserve the state
        # in the middle of trading process
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.logger = configure("results", ["stdout", "csv", "tensorboard"])
        self.np_random, _ = seeding.np_random(1)
        # print(f"{self.state=}")
        # print(f"{len(self.state)=}")
        # print(f"{self.state_space=}")

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dimension + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dimension + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dimension + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dimension + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        sell_num_shares = _do_sell_normal()
        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dimension + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dimension + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        buy_num_shares = _do_buy()
        return buy_num_shares

    def step(self, actions):

        # print(f"{self.index=} {self.total_unique_rows=}")

        # check whether one iteration is complete or not
        # if complete then calculate rewards, sharpe, make plots, etc
        self.terminal = self.index >= self.total_unique_rows - 1
        if self.terminal:
            # print(f"{self.episode=}")

            if self.make_plots:
                self._make_plot()

            # end_total_assets = initial price + (close prices) * (num_of_stock_shares)
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (1 + self.stock_dimension)])
                * np.array(
                    self.state[
                        (1 + self.stock_dimension) : (1 + self.stock_dimension * 2)
                    ]
                )
            )
            # initial_amount is only cash part of our initial asset
            tot_reward = end_total_asset - self.asset_memory[0]

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            daily_return_std = df_total_value["daily_return"].std()
            daily_return_mean = df_total_value["daily_return"].mean()
            sharpe = 0
            if daily_return_std != 0:
                sharpe = (252**0.5) * daily_return_mean / daily_return_std
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if (
                (self.episode % self.print_verbosity == 0)
                and sharpe > 0
                and (end_total_asset > self.asset_memory[0])
                and (self.trades > 1)
            ):
                print("=================================")
                print(f"day: {self.index}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if self.model_name and self.mode and self.make_plots:
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_rewards.to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png",
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            self.logger.record("environment/portfolio_value", end_total_asset)
            self.logger.record("environment/total_reward", tot_reward)
            self.logger.record(
                "environment/total_reward_pct",
                (tot_reward / (end_total_asset - tot_reward)) * 100,
            )
            self.logger.record("environment/total_cost", self.cost)
            self.logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            # actions initially is scaled between 0 to 1
            actions = (actions * self.hmax).astype(int)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dimension + 1)])
                * np.array(
                    self.state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                )
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dimension+1]}")
                # print(f"take sell action before : {actions[index]}")
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f"take sell action after : {actions[index]}")
                # print(f"Num shares after: {self.state[index+self.stock_dimension+1]}")

            for index in buy_index:
                # print(f"take buy action: {actions[index]}")
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.index += 1
            self.iteration = self.iteration + 1
            self.data = self.df.loc[self.index, :]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dimension + 1)])
                * np.array(
                    self.state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                )
            )

            # add current state in state_recorder for each step
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dimension], dtype=object)
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dimension + 1)])
                * np.array(
                    self.previous_state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.index: int = self.df.index[0]
        self.data = self.df.loc[self.index, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state

    def _make_plot(self):
        if self.make_plots:
            plt.plot(self.asset_memory, "r")
            plt.savefig(f"results/account_value_trade_{self.episode}.png")
            plt.close()

    def _initiate_state(self):
        """
        initial state -
        [initial_amount, close_values, num_stock_shares, sma_30, sma_60, ...]
        """
        if self.initial:
            if self.stock_dimension > 1:
                state = (
                    [self.initial_amount]
                    + self.data.close.to_list()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].to_list()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + self.num_stock_shares
                    + sum(
                        ([self.data[tech]] for tech in self.tech_indicator_list),
                        [],
                    )
                )  # for single stock
        else:
            if self.stock_dimension > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.to_list()
                    + self.previous_state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].to_list()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                    + sum(
                        ([self.data[tech]] for tech in self.tech_indicator_list),
                        [],
                    )
                )  # for single stock

        # print(f"len of state in initial {len(state)=}")
        return state

    def _update_state(self):
        if self.stock_dimension > 1:

            # for multiple stock
            new_state = (
                [self.state[0]]
                + self.data.close.to_list()
                + list(
                    self.state[
                        (self.stock_dimension + 1) : (self.stock_dimension * 2 + 1)
                    ]
                )
                + sum(
                    (self.data[tech].to_list() for tech in self.tech_indicator_list),
                    [],
                )
            )

        else:
            # for single stock
            new_state = (
                [self.state[0]]
                + [self.data.close]
                + list(
                    self.state[
                        (1 + self.stock_dimension) : (1 + self.stock_dimension * 2)
                    ]
                )
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return new_state

    def _get_date(self):
        if self.stock_dimension > 1:
            return self.data.date.values[0]
        return self.data.date

    def render(self, mode="human", close=False):
        return self.state

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if self.stock_dimension > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        print(f"{date_list=}")
        print(f"{state_list=}")
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if self.stock_dimension > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        return df_actions

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
