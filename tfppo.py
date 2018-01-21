
from matplotlib import pyplot as plt
import os

# numeric
import numpy as np
from numpy import random
import pandas as pd
import tensorflow as tf

# util
from collections import Counter
import pdb
import glob
import time
import tempfile
import itertools
from tqdm import tqdm_notebook as tqdm
import datetime


# setting tf
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
tf.Session(config=config).__enter__()

# logging
import logging
logger = log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
logging.basicConfig()
logger.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# params
window_length = 50
cash_bias = 0.0
batch_size=128
import datetime
import os
os.sys.path.append(os.path.abspath('.'))
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = './outputs/tensorforce-PPO/tensorforce_PPO_crypto-%s' % ts
# save_path = './outputs/tensorforce-PPO/tensorforce_PPO_crypto_20171105_06-50-31'
save_path

#logging
log_dir = 'logs/tfppo'
try:
    os.makedirs(log_dir)
except OSError:
    pass
print (log_dir)

from tensorforce.contrib.openai_gym import OpenAIGym
class TFOpenAIGymCust(OpenAIGym):
    def __init__(self, gym_id, gym):
        self.gym_id = gym_id
        self.gym = gym
        self.monitor= log_dir
        self.visualize = None
        
from Env import PortfolioEnv
from wrapper import ConcatStates
from wrapper.softmax_actions import SoftmaxActions

df_train = pd.read_hdf('./data/poloniex_30m.hf',key='train')
env = PortfolioEnv(
    df=df_train,
    steps=40, 
    scale=True, 
    trading_cost=0.0025, 
    window_length = window_length,
    output_mode='EIIE',
)
# wrap it in a few wrappers
env = ConcatStates(env)
env = SoftmaxActions(env)
environment = TFOpenAIGymCust('CryptoPortfolioEIIE-v0', env)

env.seed(0)
# sanity check out environment is working
state = env.reset()
state, reward, done, info=env.step(env.action_space.sample())






from tensorforce.agents import PPOAgent
from tensorforce.core.networks import LayeredNetwork, layers, Network, network

from tensorforce.core.networks import Layer, Conv2d, Nonlinearity
class EIIE(Layer):
    """
    EIIE layer
    """

    def __init__(self,
                 size=20,
                 bias=True,
                 activation='relu',
                 l2_regularization=0.0,
                 l1_regularization=0.0,
                 scope='EIIE',
                 summary_labels=()):
        self.size = size
        # Expectation is broadcast back over advantage values so output is of size 1
        self.conv1 = Conv2d(
            size=3,
            bias=bias,
            stride=(1,1),
            window=(1,3),
            padding='VALID',
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        # self.conv1= tf.nn.conv2d()
        self.conv2 = Conv2d(
            size=size,
            bias=bias,
            stride=(1,window_length-2-1),
            window=(1,window_length-2-1),
            padding='VALID',
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        self.conv3 = Conv2d(
            size=1,
            bias=bias,
            stride=(1,1),
            window=(1,1),
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        self.nonlinearity = Nonlinearity(
            name=activation, summary_labels=summary_labels)
        self.nonlinearity2 = Nonlinearity(
            name=activation, summary_labels=summary_labels)
        super(EIIE, self).__init__(
            scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x0, update):
        # where window_size=50, actions=4 (giving the 3), data cols=5
        # x0 = (None,3,50,5)
        # x = (None,3,49,5)
        # x = (None,3,1,1)
        # conv1 => (None,3, 47,3)
        # conv2 => (None,3, 1, 20)
        # concat=> (None,3, 1, 21)
        # conv3 => (None,3, 1, 1)
        # concat=> (None,2, 1, 1)

        w0 = x0[:,:,:1,:1]
        x = x0[:,:,1:,:]
        
        x = self.conv1.apply(x, update=update)
        # x = self.nonlinearity.apply(x=x, update=update)
        
        x = self.conv2.apply(x, update=update)
        # x = self.nonlinearity2.apply(x=x, update=update)
        
        x = tf.concat([x, w0], 3)
        x = self.conv3.apply(x, update=update)
        
        # concat on cash_bias
        cash_bias_int = 0
        # FIXME not sure how to make shape with a flexible size in tensorflow but this works for now
        # cash_bias = tf.ones(shape=(batch_size,1,1,1)) * cash_bias_int
        cash_bias = x[:,:1,:1,:1]*0 
        x = tf.concat([cash_bias, x], 1)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(EIIE, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(EIIE, self).tf_regularization_loss()]

        if self.conv1.regularization_loss() is not None:
            losses.append(self.conv1.regularization_loss())
        if self.conv2.regularization_loss() is not None:
            losses.append(self.conv2.regularization_loss())
        if self.conv1.regularization_loss() is not None:
            losses.append(self.conv3.regularization_loss())

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(EIIE, self).get_variables(
            include_non_trainable=include_non_trainable)

        layer_variables += self.conv1.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.conv2.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.conv3.get_variables(
            include_non_trainable=include_non_trainable)

        layer_variables += self.nonlinearity.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.nonlinearity.get_variables(
            include_non_trainable=include_non_trainable)

        return layer_variables
    
# Add our custom layer
layers['EIIE'] = EIIE

# Network as list of layers
network_spec = [
    dict(type='EIIE', 
         l1_regularization=1e-8,
        l2_regularization=1e-8),
    dict(type='flatten')
]

# also add a custom baseline
from tensorforce.core.baselines import NetworkBaseline
from tensorforce.core.baselines import baselines


class EIIEBaseline(NetworkBaseline):
    """
    CNN baseline (single-state) consisting of convolutional layers followed by dense layers.
    """

    def __init__(self, layers_spec, scope='eiie-baseline', summary_labels=()):
        """
        CNN baseline.
        Args:
            conv_sizes: List of convolutional layer sizes
            dense_sizes: List of dense layer sizes
        """

        super(EIIEBaseline, self).__init__(layers_spec, scope, summary_labels)
        
# Add our custom baseline
baselines['EIIE']=EIIEBaseline

exploration=dict(
    type="epsilon_anneal",
    initial_epsilon=1.0,
    final_epsilon= 0.005,
    timesteps=100000,
    start_timestep=0
)
# from tensorforce.core.explorations.epsilon_anneal import EpsilonAnneal
# explorations_spec=EpsilonAnneal(initial_epsilon=1.0, final_epsilon=0.005, timesteps=1e5, start_timestep=0)

# exploration = tensorforce.core.explorations.EpsilonAnneal(**exploration)
config = dict( 
    type= "ppo_agent",
    batch_size=batch_size,
    
    # Each agent requires the following ``Configuration`` parameters:
    # preprocessing = None,# dict or list containing state preprocessing configuration.
    explorations_spec= exploration, #{'action' + str(n): exploration for n in range(env.action_space.shape[0])}, # dict containing action exploration configuration.
    # reward_preprocessing=None,
    
    # BatchAgent
    keep_last_timestep=True,

    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate = 1e-3, # float of learning rate (alpha). (3e-4 in paper 1e-3 (atari) and 3e-4 in baselines)
    ),
    optimization_steps=4,
    
    # Each model requires the following configuration parameters:
    # https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/model.py#L33
    scope='ppo',
    discount = 0.97, # float of discount factor (gamma).
    saver_spec = dict(
        directory=save_path, 
        steps=100000, 
#         basename=os.path.basename(save_path)
    ),
    
    # DistributionModel
    # distributions=dict(action=dict(type='gaussian', mean=0.25, log_stddev=np.log(5e-2))),
    entropy_regularization=0.01, # 0 and 0.01 in baselines
    
    # PGModel
#     baseline_mode='network', # states or network
    baseline_mode='states',
    baseline=dict(
        type="EIIE",
        layers_spec=network_spec
#         update_batch_size=512,
    ), # string indicating the baseline value function (currently 'linear' or 'mlp').
    baseline_optimizer=dict(type='adam', learning_rate=0.003),
    gae_lambda=0.5,
    # normalize_rewards=False,
    
    # PGLRModel
    likelihood_ratio_clipping=0.2,  # Trust region clipping 0.2 in paper
    
    # Logging
    # log_level = 'info', # string containing log level (e.g. 'info').
    
    # Tensorflow summaries
    # summary_logdir = log_dir, # string directory to write tensorflow summaries. Default None
    # summary_labels=['total-loss'],
    # summary_frequency=10,
    
    # TensorFlow distributed configuration
    # cluster_spec=None,
    # parameter_server=False,
    # task_index=0,
    # device=None,
    # local_model=False,
    # replica_model=False,
)
# config['exloratoins_spec'] = exploration
# I want to use a gaussian dist instead of beta, we will apply post processing to scale everything
actions_spec = environment.actions.copy()
del actions_spec["min_value"]
del actions_spec["max_value"]

# Create an agent
from tensorforce.agents import Agent
agent = Agent.from_spec(
    spec=config,
    kwargs=dict(
        states_spec=environment.states,
        actions_spec=actions_spec,
        network_spec=network_spec
    )
)
agent
report_episodes = 100
from tensorforce.execution import Runner
# from callbacks import *
def episode_finished(r):
    if r.episode % report_episodes == 0:
        steps_per_second = r.timestep / (time.time() - r.start_time)
        logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
            r.agent.episode, r.episode_timestep, steps_per_second
        ))
        logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
        logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
        logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
    return True
runner = Runner(agent=agent, environment=environment)
steps=10e8
env._plot = env._plot2 = env._plot3 = None
episodes = int(steps / 30)
logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
runner.run(
    timesteps=steps,
    episodes=episodes,
    episode_finished=episode_finished
)
logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))