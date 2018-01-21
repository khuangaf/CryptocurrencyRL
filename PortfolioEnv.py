import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint
import logging
import os
import tempfile
import time
import gym
import gym.spaces

from config import eps


logger = logging.getLogger(__name__)


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df_list, times, steps=252, scale=True, scale_extra_cols=True, augment=0.00, window_length=50):
        """
        DataSrc.
        df - csv for data frame index of timestamps
             and columns are ['close','high','low','open']
        steps - total steps in episode
        scale - scale the data for each episode
        scale_extra_cols - scale extra columns by global mean and std
        augment - fraction to augment the data by
        """
        # total steps
        self.steps = steps + 1
        self.augment = augment
        self.scale = scale
        # self.scale_extra_cols = scale_extra_cols
        self.window_length = window_length
#         '''for BTC only'''
#         self.asset_names = ['BTC']
#         # get rid of NaN's
#         df = df.copy()
#         df.replace(np.nan, 0, inplace=True)
#         df = df.fillna(method="pad")

#         # dataframe to matrix
#         print (df.columns)
#         self.features = df.columns
#         data = df.as_matrix().reshape(
#             (len(df), 1, len(self.features)))
#         self._data = np.transpose(data, (1, 0, 2))
#         self._times = df.index

#         self.price_columns = ['close', 'high', 'low', 'open']
        '''for 4 coins'''
        self.asset_names = ['BTC', 'LTC', 'ETH', 'XMR']
        
        df_list = df_list.copy()
        self.features = df_list[0].columns
        df_list = [ np.expand_dims(df, 1)for df in df_list]
        # shape = (total timesteps, num_coins, num_features)
        data = np.hstack(df_list)
        
        # df.replace(np.nan, 0, inplace=True)
        # df = df.fillna(method="pad")

        # dataframe to matrix
        
        
        # data = df.as_matrix().reshape(
        #     (len(df), 1, len(self.features)))
        self._data = np.transpose(data, (1, 0, 2))
        self._times = times

        self.price_columns = ['close', 'high', 'low', 'open']    

        self.reset()

    def _step(self):
        # get history matrix from dataframe
        data_window = self.data[:, self.step:self.step +
                                self.window_length].copy()

        # (eq.1) prices
        y1 = data_window[:, -1, 0] / data_window[:, -2, 0]
        y1 = np.concatenate([[1.0], y1])  # add cash price

        # (eq 18) X: prices are divided by close price
        nb_pc = len(self.price_columns)
        if self.scale:
            last_close_price = data_window[:, -1, 0]
            data_window[:, :, :nb_pc] /= last_close_price[:,
                                                          np.newaxis, np.newaxis]

        # if self.scale_extra_cols:
        #     # normalize non price columns
        #     data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
        #     data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
        #     data_window[:, :, nb_pc:] = np.clip(
        #         data_window[:, :, nb_pc:],
        #         self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
        #         self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10
        #     )

        self.step += 1
        history = data_window
        done = bool(self.step >= self.steps)
        return history, y1, done

    def reset(self):
        self.step = 0

        
        
        # the index to start with
        self.idx = np.random.randint(
            low=self.window_length, high=self._data.shape[1] - self.steps)
        
        # get data for this episode
        data = self._data[:, self.idx -
                          self.window_length:self.idx + self.steps + 1].copy()
        self.times = self._times[self.idx -
                                 self.window_length:self.idx + self.steps + 1]

        # augment data to prevent overfitting
        data += np.random.normal(loc=0, scale=self.augment, size=data.shape)

        self.data = data


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=[], steps=128, trading_cost=0.0025, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.reset()

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1] * first element is always 1
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        # print "w1"
        # print w1
        w0 = self.w0
        p0 = self.p0
        
        # w0 prime denotes weight * asset value
        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        
        # transaction
        c1 = self.cost * (
            np.abs(dw1[1:] - w1[1:])).sum()

        p1 = p0 * (1 - c1) * np.dot(y1, w0)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf)

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        # (eq22) immediate reward is log rate of return scaled by episode length
        reward = r1 / self.steps

        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = bool(p1 == 0)

        # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
        }
        # record weights and prices
        for i, name in enumerate(['BTCBTC'] + self.asset_names):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]

        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names))
        self.p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for trading bitcoins.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['notebook', 'ansi']}

    def __init__(self,
                 df_list,
                 times,
                 steps=256,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 augment=0.00,
                 output_mode='EIIE',
                 log_dir=None,
                 scale=True,
                 scale_extra_cols=True,
                 ):
        """
        An environment for trading bitcoins.
        Params:
            df - csv for data frame index of timestamps and columns are ['close','high','low','open']
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(df_list=df_list, times= times, steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_length=window_length)
        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)
        self.log_dir = log_dir

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=nb_assets + 1)

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'atari':
            obs_shape = (
                window_length,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'mlp':
            obs_shape = (nb_assets) * window_length * \
                (len(self.src.features))
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                20 if scale else 1,  # if scale=True observed price changes return could be large fractions
                obs_shape
            ),
            'weights': self.action_space
        })
        # self.observation_space = gym.spaces.Box(
        #         -10,
        #         20 if scale else 1,  # if scale=True observed price changes return could be large fractions
        #         obs_shape
        #     )
        self._reset()

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        # print 'action'
        # print action
        logger.debug('action: %s', action)

        weights = np.clip(action, 0.0, 1.0)
        weights /= weights.sum() + eps
        # print weights
        # Sanity checks
        assert self.action_space.contains(
            weights), 'action should be within %r but is %r' % (self.action_space, weights)
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        history, y1, done1 = self.src._step()
        
        reward, info, done2 = self.sim._step(weights, y1)
        
        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        # info['date'] = pd.to_datetime(self.src.times[self.src.step],unit='s')
        # info['date'] = self.src.times[self.src.step]
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [
                0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()
        # print (history.shape)
        # return history, reward, done1 or done2, info
        return {'history': history, 'weights': weights}, reward, done1 or done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]

    def _render(self, mode='notebook', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot_notebook(close)

    