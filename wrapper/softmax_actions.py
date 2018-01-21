import gym.wrappers
import numpy as np
eps = 1e-7
def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

class SoftmaxActions(gym.Wrapper):
    """
    Environment wrapper to softmax actions.
    Usage:
        env = gym.make('Pong-v0')
        env = SoftmaxActions(env)
    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md
    """

    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))

        action = softmax(action, t=1)

        return self.env.step(action)