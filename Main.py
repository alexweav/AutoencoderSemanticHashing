from Layers import *
from DataLoaders import *
from Autoencoder import *

def main():
    data = LoadMNIST()

    test = np.random.randn(2, 2)
    test_original = test
    autoencoder = Autoencoder([2, 3, 4, 5])

    inputs = {}
    
    """
    print(test)
    encoder = autoencoder.Encoder()
    for layer in encoder:
        inputs[layer] = test
        test = layer.Forward(test)
        #print(test)

    decoder = autoencoder.Decoder()
    for layer in decoder:
        inputs[layer] = test
        test = layer.Forward(test)
        #print(test)

    loss, deriv = autoencoder.MSELoss(test_original, test)
    print("Loss: ", loss)
    print(deriv)

    for layer in reversed(decoder):
        deriv, d_param = layer.Backward(inputs[layer], deriv)
        print("input", inputs[layer].shape)
        print("d_input", deriv.shape)

    for layer in reversed(encoder):
        deriv, d_param = layer.Backward(inputs[layer], deriv)
        print(deriv)
    """
    data, inputs = autoencoder.EvaluateEncoder(test, inputs)
    print(data)
    data, inputs = autoencoder.EvaluateDecoder(data, inputs)
    print(data)
    data, inputs = autoencoder.EvaluateFull(data)
    print(data)



main()
