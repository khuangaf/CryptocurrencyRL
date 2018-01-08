from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from TradingEnvironment import *
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape, Input,Concatenate
import tensorflow as tf
import numpy as np
import random
from keras.layers import GRU, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
import keras
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from keras.optimizers import Adam
from rl.random import OrnsteinUhlenbeckProcess


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

df = pd.read_csv('data/historical_bitcoin.csv').set_index('date')
env = TradingEnvironment(df=df)


nb_actions = env.action_space.shape[0]
observation_space= env.observation_space.shape

# define model parameter
step_size = 50
nb_features = 4
units= 50

# actor
# actor = Sequential()
# actor.add(GRU(units=units, input_shape=observation_space[1:],return_sequences=False))
# actor.add(Activation('relu'))
# actor.add(Dropout(0.2))
# actor.add(Dense(nb_actions, activation=K.softmax))
# # model.add(Activation('softmax'))
# print(actor.summary())
print env.observation_space.shape


#actor
actor = Sequential()
actor.add(Flatten(input_shape= (1,)+env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('softmax'))
print(actor.summary())

# critic
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)

    

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=1)


# build agent
ddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                   gamma=.99, target_model_update=1e-3)

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that you can the built-in Keras callbacks!
weights_filename = 'weights/dqn_weights.h5f'
checkpoint_weights_filename = 'weights/dqn_weights_{step}.h5f'
log_filename = 'logs/dqn_log.json'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
ddpg.compile(Adam(lr=.001), metrics=['mae'])
ddpg.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, verbose=1)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=False)