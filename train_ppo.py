
from Env import *

import tensorflow as tf
import numpy as np
import random
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import ppo2
from policies import CnnPolicy
import gym
import logging
import multiprocessing
import os.path as osp
import tensorflow as tf
import sys
import argparse
from baselines import bench, logger
import _pickle  as pickle
from wrapper.concat_states import ConcatStates
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

def train(df, num_timesteps, seed, policy=CnnPolicy, training=True):
    
    # configure GPU usage
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    
    df = df
    
    # define model parameter
    step_size = 50
    nb_features = 4
    units= 50
    
    
    def make_env():
        env = PortfolioEnv(df=df)
        env = ConcatStates(env)
        env = bench.Monitor(env, logger.get_dir())
        return env
    

    env = DummyVecEnv([make_env])
    set_global_seeds(seed)

    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1000,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1), training=training)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(8e5))
    args = parser.parse_args()
    logger.configure(dir='./logs/ppo_train_unclip')
    df = pd.read_hdf('./data/poloniex_30m.hf',key='train')
    train(df, num_timesteps=int(60000000), seed=args.seed, training=True)
    logger.configure(dir='./logs/ppo_test_unclip')
    df = pd.read_hdf('./data/poloniex_30m.hf',key='test')
    train(df, num_timesteps=48*7, seed=args.seed, training=False)

if __name__ == '__main__':
    main()
