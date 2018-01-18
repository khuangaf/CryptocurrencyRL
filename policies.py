import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, conv1d, sample, softmax_sample
from baselines.common.distributions import make_pdtype

# class LnLstmPolicy(object):
#     def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
#         nenv = nbatch // nsteps
#         nh, nw, nc = ob_space.shape
#         ob_shape = (nbatch, nh, nw, nc)
#         nact = ac_space.n
#         X = tf.placeholder(tf.uint8, ob_shape) #obs
#         M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
#         S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
#         with tf.variable_scope("model", reuse=reuse):
#             h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
#             h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
#             h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
#             h3 = conv_to_fc(h3)
#             h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
#             xs = batch_to_seq(h4, nenv, nsteps)
#             ms = batch_to_seq(M, nenv, nsteps)
#             h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
#             h5 = seq_to_batch(h5)
#             pi = fc(h5, 'pi', nact, act=lambda x:x)
#             vf = fc(h5, 'v', 1, act=lambda x:x)

#         self.pdtype = make_pdtype(ac_space)
#         self.pd = self.pdtype.pdfromflat(pi)

#         v0 = vf[:, 0]
#         a0 = self.pd.sample()
#         neglogp0 = self.pd.neglogp(a0)
#         self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

#         def step(ob, state, mask):
#             return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

#         def value(ob, state, mask):
#             return sess.run(v0, {X:ob, S:state, M:mask})

#         self.X = X
#         self.M = M
#         self.S = S
#         self.pi = pi
#         self.vf = vf
#         self.step = step
#         self.value = value

# class LstmPolicy(object):

#     def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
#         nenv = nbatch // nsteps

#         nh, nw, nc = ob_space.shape
#         ob_shape = (nbatch, nh, nw, nc)
#         # print (ac_space.shape)
#         # nact = ac_space.n
#         nact = ac_space.shape[0]
#         X = tf.placeholder(tf.float32, ob_shape) #obs
#         M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
#         S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
#         with tf.variable_scope("model", reuse=reuse):
#             h = conv1d(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
#             h2 = conv1d(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
#             h3 = conv1d(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
#             h3 = conv_to_fc(h3)
#             h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
#             xs = batch_to_seq(h4, nenv, nsteps)
#             ms = batch_to_seq(M, nenv, nsteps)
#             h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
#             h5 = seq_to_batch(h5)
#             pi = fc(h5, 'pi', nact, act=lambda x:x)
#             vf = fc(h5, 'v', 1, act=lambda x:x)

#         self.pdtype = make_pdtype(ac_space)
#         self.pd = self.pdtype.pdfromflat(pi)

#         v0 = vf[:, 0]
#         a0 = self.pd.sample()
#         neglogp0 = self.pd.neglogp(a0)
#         self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

#         def step(ob, state, mask):
#             return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

#         def value(ob, state, mask):
#             return sess.run(v0, {X:ob, S:state, M:mask})

#         self.X = X
#         self.M = M
#         self.S = S
#         self.pi = pi
#         self.vf = vf
#         self.step = step
#         self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        print (ob_shape)
        # nact = ac_space.n
        # print (ob_shape)
        nact = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=16, rf=3, stride=4, init_scale=np.sqrt(2))
            max1=tf.nn.max_pool(h, ksize=(1,2,2,1), padding='VALID', strides= (1,1,1,1))
            
            # h2 = conv(h, 'c2', nf=8, rf=16, stride=2, init_scale=np.sqrt(2))
            # h3 = conv(h2, 'c3', nf=4, rf=8, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(max1)
            h4 = fc(h3, 'fc1', nh=64, init_scale=np.sqrt(2))
            p = fc(h4, 'pi', 5, act=lambda x:x, init_scale=0.01)
            pi = tf.nn.softmax(p)
            vf = fc(h4, 'v', 1, act=lambda x:x)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)
        
        # a0 = self.pd.sample()
        a0 = pi
        # print (softmax_sample(pi))
        # print ("a0")
        # print (self.pd)
        # print (a0)
        # print (pi)
        # print (h)
        # print ("nact")
        # print (nact)
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            # print ("a"+str(a))
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value