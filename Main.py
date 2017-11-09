from Layers import *
from DataLoaders import *
from Autoencoder import *
from Optimizers import *

def main():
    data = LoadMNIST()

    test = np.random.randn(2, 2)
    test_original = test
    autoencoder = Autoencoder([2, 3, 4, 5])

    inputs = {}
    
    data, inputs = autoencoder.EvaluateEncoder(test, inputs)
    print(data)
    data, inputs = autoencoder.EvaluateDecoder(data, inputs)
    print(data)
    data, inputs = autoencoder.EvaluateFull(data)
    print(data)
    loss, deriv = autoencoder.MSELoss(test_original, data)
    grads = autoencoder.Backprop(inputs, deriv)
    optim = AdamOptimizer(1, 0.99, 0.99)
    cache = {}
    theta, cache = optim.Optimize(autoencoder.Decoder()[-1].Param(), grads[autoencoder.Decoder()[-1]][1], cache)
    print('final bias')
    print(autoencoder.Decoder()[-1].Param())
    print(theta)



main()
