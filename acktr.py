
from TradingEnvironment import *
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
from baselines.ppo2 import ppo2
from policies import CnnPolicy
import gym
import logging
import multiprocessing
import os.path as osp
import tensorflow as tf
import sys
import argparse
from baselines import bench, logger
from acktr_policies import CnnPolicy
from acktr_disc import learn

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

def train(num_timesteps, seed):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
        
    config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    df = pd.read_csv('data/historical_bitcoin.csv').set_index('date')
    def make_env(rank):
        def env_fn():
            env = TradingEnvironment(df=df)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env)
        return env_fn
    def make_env():
        env = TradingEnvironment(df=df)
        env = bench.Monitor(env, logger.get_dir())
        return env
    set_global_seeds(seed)
    # env = SubprocVecEnv([make_env(i) for i in range(3)])
    env = DummyVecEnv([make_env])
    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=3)
    env.close()

def train1(num_timesteps, seed, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    df = pd.read_csv('data/historical_bitcoin.csv').set_index('date')
    

    step_size = 50
    nb_features = 4
    units= 50
    
    
    

    tf.Session(config=config).__enter__()
    def make_env(rank):
        def env_fn():
            env = TradingEnvironment(df=df)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env)
        return env_fn
    def make_env():
        env = TradingEnvironment(df=df)
        env = bench.Monitor(env, logger.get_dir())
        return env
    nenvs = 1
    # env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    # env = TradingEnvironment(df=df)
    env = DummyVecEnv([make_env])
    set_global_seeds(seed)
    policy = CnnPolicy
    # policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    train(num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
