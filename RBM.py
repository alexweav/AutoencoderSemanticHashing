import numpy as np

class RBM(object):

    def __init__(self, visible_size, hidden_size):
        self.weights = np.random.randn(visible_size, hidden_size) / np.sqrt(visible_size)
        self.bias_visible = np.zeros(visible_size)
        self.bias_hidden = np.zeros(hidden_size)

    def PropHidden(self, inputs):
        return self.Sigmoid(np.dot(inputs, self.weights) + self.bias_hidden)

    def PropVisible(self, inputs):
        return self.Sigmoid(np.dot(inputs, self.weights.T) + self.bias_visible)

    def Sample(self, probabilities):
        return self.BinaryStochastic(probabilities)

    def Sigmoid(self, inputs):
        return 1. / (1 + np.exp(-inputs))

    def BinaryStochastic(self, x):
        pad = np.random.uniform(size=x.shape)
        x[pad <= x] = 1.
        x[pad > x] = 0.
        return x

    def Cycle(self, initial_state):
        hidden = self.Sample(self.PropHidden(initial_state))
        return self.Sample(self.PropVisible(hidden))

    def Train(self, initial_state):
        pass

