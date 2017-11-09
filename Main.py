from Layers import *
from DataLoaders import *
from Autoencoder import *
from Optimizers import *

def main():
    data = LoadMNIST()

    test = np.random.randn(1, 5)
    test_original = test
    autoencoder = Autoencoder([5, 4, 3])

    inputs = {}
    
    print('original')
    print(test)
    data, inputs = autoencoder.EvaluateEncoder(test, inputs)
    print('code')
    print(data)
    data, inputs = autoencoder.EvaluateDecoder(data, inputs)
    print('reconstruction')
    print(data)
    data, inputs = autoencoder.EvaluateFull(data)
    print('full evaluation reconstruction')
    print(data)
    loss, deriv = autoencoder.MSELoss(test_original, data)
    print('loss', loss[0])
    grads = autoencoder.Backprop(inputs, deriv)
    optim = AdamOptimizer(1, 0.99, 0.99)
    cache = {}
    theta, cache = optim.Optimize(autoencoder.Encoder()[0].Param(), grads[autoencoder.Encoder()[0]][1], cache)
    print('old first weights')
    print(autoencoder.Encoder()[0].Param())
    print('new first weights')
    print(theta)



main()
