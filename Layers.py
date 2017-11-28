import numpy as np

class Layer(object):

    def Forward(self, inputs):
        raise NotImplementedError('Calling an abstract method')

    def Backward(self, prev_inputs, dout):
        raise NotImplementedError('Calling an abstract method')

    def Param(self):
        raise NotImplementedError('Calling an abstract method')

    def Optimize(self, optimizer, optimizer_cache, d_param):
        raise NotImplementedError('Calling an abstract method')

class MatMul(Layer):
    
    def __init__(self, rows, cols):
        self.weights = np.random.randn(rows, cols) / np.sqrt(rows)

    def __init__(self, weights):
        self.weights = weights

    def Forward(self, inputs):
        return inputs.dot(self.weights)

    def Backward(self, prev_inputs, dout):
        self.d_weights = np.dot(prev_inputs.T, dout)
        return np.dot(dout, self.weights.T), self.d_weights

    def Param(self):
        return self.weights

    def Optimize(self, optimizer, optimizer_cache, d_param):
        new_weights, cache = optimizer.Optimize(self.weights, d_param, optimizer_cache)
        self.weights = new_weights
        return cache

class Bias(Layer):

    def __init__(self, shape):
        self.bias = np.zeros(shape)

    def Forward(self, inputs):
        return inputs + self.bias

    def Backward(self, prev_inputs, dout):
        self.d_bias = np.sum(dout, axis=0)
        return dout, self.d_bias

    def Param(self):
        return self.bias

    def Optimize(self, optimizer, optimizer_cache, d_param):
        new_bias, cache = optimizer.Optimize(self.bias, d_param, optimizer_cache)
        self.bias = new_bias
        return cache

class ReLU(Layer):
    
    def Forward(self, inputs):
        return inputs * (inputs > 0)

    def Backward(self, prev_inputs, dout):
        return dout * (prev_inputs >= 0), None

    def Param(self):
        return None

    def Optimize(self, optimizer, optimizer_cache, d_param):
        return None

class Sigmoid(Layer):
    
    def Forward(self, inputs):
        return 1. / (1 + np.exp(-inputs))

    def Backward(self, prev_inputs, dout):
        return (self.Forward(prev_inputs) * (1. - self.Forward(prev_inputs))) * dout, None

    def Param(self):
        return None

    def Optimize(self, optimizer, optimizer_cache, d_param):
        return None

class BinaryStochastic(Layer):
    
    def Forward(self, inputs):
        pad = np.random.uniform(size=inputs.shape)
        inputs[pad <= inputs] = 1
        inputs[pad > inputs] = 0
        return inputs

    def Backward(self, prev_inputs, dout):
        return dout, None

    def Param(self):
        return None

    def Optimize(self, optimizer, optimizer_cache, d_param):
        return None

