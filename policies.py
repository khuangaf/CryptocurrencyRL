import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, conv1d, sample, softmax_sample
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        print (ob_space)
        print (ac_space)
        
        window_length = ob_space.shape[1] -1

        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs

        with tf.variable_scope("model", reuse=reuse):
            w0 = tf.slice(X, [0,0,0,0],[-1,-1,1,1])
            x = tf.slice(X, [0,0,1,0],[-1,-1,-1,-1])
            x = conv(tf.cast(x, tf.float32),'c1', fh=1,fw=3,nf=3, stride=1, init_scale=np.sqrt(2))
            print (x)
            x = conv(x, 'c2', fh=1, fw=window_length -2, nf=20, stride= window_length -2, init_scale=np.sqrt(2))
            print (x)
            x = tf.concat([x, w0], 3)
            print (x)
            x = conv(x, 'c3', fh=1, fw=1, nf=1, stride= 1, init_scale=np.sqrt(2))
            print (x)
            cash_bias = tf.ones([x.shape[0],1,1,1], tf.float32)
            c = tf.concat([cash_bias, x], 1)
            
            v = conv_to_fc(x)
            vf = fc(v, 'v',1)[:,0]
            print (v)
            f = tf.contrib.layers.flatten(c)
            print (f)
            pi = tf.nn.softmax(f)
            print (pi)
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.truncated_normal_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        a0 = tf.nn.softmax(a0)
        # print (a0)
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