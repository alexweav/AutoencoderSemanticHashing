import numpy as np

class RBM(object):

    def __init__(self, visible_size, hidden_size):
        self.weights = np.random.randn(visible_size, hidden_size) / np.sqrt(visible_size)
        self.bias_visible = np.zeros(in_size)
        self.bias_hidden = np.zeros(hidden_size)

    def PropHidden(self, inputs):
        pass

    def Sigmoid(self, inputs):
        return 1. / (1 + np.exp(-inputs))

