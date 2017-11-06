import numpy as np

class Layer(object):

    def Forward(self, inputs):
        raise NotImplementedError('Calling an abstract method')

    def Backward(self, prev_inputs, dout):
        raise NotImplementedError('Calling an abstract method')

class MatMul(Layer):
    
    def __init__(self, rows, cols):
        self.weights = np.random.randn(rows, cols)/np.sqrt(rows)

    def Forward(self, inputs):
        return inputs.dot(self.weights)

    def Backward(self, prev_inputs, dout):
        self.d_weights = np.dot(prev_inputs.T, dout)
        return np.dot(dout, self.weights.T), self.d_weights

class Bias(Layer):

    def __init__(self, shape):
        self.bias = np.zeros(shape)

    def Forward(self, inputs):
        return inputs + self.bias

    def Backward(self, prev_inputs, dout):
        self.d_bias = np.sum(dout, axis=0)
        return dout, self.d_bias

class ReLU(Layer):
    
    def Forward(self, inputs):
        inputs[inputs < 0] = 0
        return inputs

    def Backward(self, prev_inputs, dout):
        dout[prev_inputs < 0] = 0.0
        return dout

class Sigmoid(Layer):
    
    def Forward(self, inputs):
        return 1. / (1 + np.exp(-inputs))

    def Backward(self, prev_inputs, dout):
        return (self.sigmoid(prev_inputs) * (1. - self.sigmoid(prev_inputs))) * dout

class BinaryStochastic(Layer):
    
    def Forward(self, inputs):
        pad = np.random.uniform(size=inputs.shape)
        inputs[pad <= inputs] = 1
        inputs[pad > inputs] = 0
        return inputs

    def Backward(self, prev_inputs, dout):
        return dout

