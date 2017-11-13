from Layers import *

class Autoencoder(object):

    def __init__(self, architecture):
        self.encoder = []
        self.decoder = []
        previous_layer_size = architecture[0]
        for layer_size in architecture[1:-1]:
            self.encoder.append(MatMul(previous_layer_size, layer_size))
            self.encoder.append(Bias(layer_size))
            self.encoder.append(ReLU())
            previous_layer_size = layer_size
        #encoding layer
        self.encoder.append(MatMul(previous_layer_size, architecture[-1]))
        self.encoder.append(Bias(architecture[-1]))
        self.encoder.append(Sigmoid())
        self.encoder.append(BinaryStochastic())
        
        previous_layer_size = architecture[-1]
        for layer_size in reversed(architecture[1:-1]):
            self.decoder.append(MatMul(previous_layer_size, layer_size))
            self.decoder.append(Bias(layer_size))
            self.decoder.append(ReLU())
            previous_layer_size = layer_size
        self.decoder.append(MatMul(architecture[1], architecture[0]))
        self.decoder.append(Bias(architecture[0]))

    def Encoder(self):
        return self.encoder

    def Decoder(self):
        return self.decoder

    def MSELoss(self, image, reconstruction):
        loss = 0.5*np.sum((reconstruction-image)**2, axis=1)
        d_loss = reconstruction-image
        return loss, d_loss

    def EvaluateEncoder(self, data, inputs={}):
        for layer in self.encoder:
            inputs[layer] = data
            data = layer.Forward(data)
        return data, inputs


    def EvaluateDecoder(self, data, inputs={}):
        for layer in self.decoder:
            inputs[layer] = data
            data = layer.Forward(data)
        return data, inputs

    def EvaluateFull(self, data, inputs={}):
        code, inputs = self.EvaluateEncoder(data, inputs)
        return self.EvaluateDecoder(code, inputs)

    def Backprop(self, prev_inputs, d_loss):
        grads = {}
        deriv = d_loss
        for layer in reversed(self.decoder):
            deriv, d_param = layer.Backward(prev_inputs[layer], deriv)
            grads[layer] = (deriv, d_param)
        for layer in reversed(self.encoder):
            deriv, d_param = layer.Backward(prev_inputs[layer], deriv)
            grads[layer] = (deriv, d_param)
        return grads

    def Optimize(self, grads):
        pass

