import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, conv1d, sample, softmax_sample
from baselines.common.distributions import make_pdtype

def softmax(w, t=1.0):
    """softmax that avoid inf/ nan."""
    eps = 1e-7
    log_eps = np.log(eps)
    w = tf.clip_by_value(w, -log_eps, log_eps)  # avoid inf/nan
    e = tf.exp(w / t)
    dist = e / tf.reduce_sum(e)
    return dist


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=True): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]

        
        window_length = ob_space.shape[1] -1

        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        
        
        with tf.variable_scope("model", reuse=reuse) as scope:
            w0 = tf.slice(X, [0,0,0,0],[-1,-1,1,1])
            x = tf.slice(X, [0,0,1,0],[-1,-1,-1,-1])
            
            # reuse when testing
            try:
                x = conv(tf.cast(x, tf.float32),'c1', fh=1,fw=3,nf=3, stride=1, init_scale=np.sqrt(2))
            except:
                scope.reuse_variables()
                x = conv(tf.cast(x, tf.float32),'c1', fh=1,fw=3,nf=3, stride=1, init_scale=np.sqrt(2))

            x = conv(x, 'c2', fh=1, fw=window_length -2, nf=20, stride= window_length -2, init_scale=np.sqrt(2))

            x = tf.concat([x, w0], 3)

            x = conv(x, 'c3', fh=1, fw=1, nf=1, stride= 1, init_scale=np.sqrt(2))

            cash_bias = tf.ones([x.shape[0],1,1,1], tf.float32)
            c = tf.concat([cash_bias, x], 1)
            
            v = conv_to_fc(x)
            vf = fc(v, 'v',1)[:,0]
       
            f = tf.contrib.layers.flatten(c)
       
            pi = tf.nn.softmax(f)
       
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.truncated_normal_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        a0 = tf.nn.softmax(a0) 
        
        
        # a0 = tf.clip_by_value(a0, -eps, eps, 'clip')
        
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