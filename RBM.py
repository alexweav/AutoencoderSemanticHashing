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
        return self.Sample(self.CycleContinuous(initial_state))

    def CycleContinuous(self, initial_state):
        hidden = self.Sample(self.PropHidden(initial_state))
        return self.PropVisible(hidden)

    def Train(self, initial_state, learning_rate):
        hidden = self.Sample(self.PropHidden(initial_state))
        recon = self.Sample(self.PropVisible(hidden))
        hidden_recon = self.Sample(self.PropHidden(recon))
        self.weights = self.weights + learning_rate*(np.dot(initial_state.T, hidden) - np.dot(recon.T, hidden_recon))
        self.bias_visible = self.bias_visible + np.sum(learning_rate*(initial_state - recon), axis=0)
        self.bias_hidden = self.bias_hidden + np.sum(learning_rate*(hidden - hidden_recon), axis=0)

    def Weights(self):
        return self.weights

    def VisibleBias(self):
        return self.bias_visible

    def HiddenBias(self):
        return self.bias_hidden

