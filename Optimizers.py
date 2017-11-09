import numpy as np

class IOptimizer(object):

    def Optimize(self, theta, dtheta, cache):
        raise NotImplementedError('Calling an abstract method')

class AdamOptimizer(IOptimizer):

    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def Optimize(self, theta, dtheta, cache):
        m = cache.get('m', np.zeros_like(theta))
        v = cache.get('v', np.zeros_like(theta))
        t = cache.get('t', 0)
        t += 1
        m = self.beta1 * m + (1 - self.beta1) * dtheta
        v = self.beta2 * v + (1 - self.beta2) * dtheta**2
        corrected_m = m/(1 - self.beta1**t)
        corrected_v = v/(1 - self.beta2**t)
        cache['t'] = t
        cache['m'] = m
        cache['v'] = v
        return theta + (-self.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)), cache

