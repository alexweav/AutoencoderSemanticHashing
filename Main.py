from Layers import *
from DataLoaders import *
from Autoencoder import *
from Optimizers import *
import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab



def plot_dual(img1, img2):
    img1 = img1.reshape(28, 28)
    img2 = img2.reshape(28, 28)
    plot_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(plot_image, cmap='Greys')
    pylab.show()


def main():
    data = LoadMNIST()
    img = data['train_X'][0:1]
    autoencoder = Autoencoder([784, 500, 32])
    reconstruction, inputs = autoencoder.EvaluateFull(img)
    plot_dual(img, reconstruction)
    loss, deriv = autoencoder.MSELoss(img, reconstruction)
    print(loss)
    grads = autoencoder.Backprop(inputs, deriv)
    optim = AdamOptimizer(1, 0.99, 0.99)
    cache = autoencoder.Optimize(grads, optim)
    reconstruction, inputs = autoencoder.EvaluateFull(img)
    plot_dual(img, reconstruction)




    """
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
    """
    """
    cache = {}
    theta, cache = optim.Optimize(autoencoder.Encoder()[0].Param(), grads[autoencoder.Encoder()[0]][1], cache)
    print('old first weights')
    print(autoencoder.Encoder()[0].Param())
    print('new first weights')
    print(theta)
    """
    """
    print(autoencoder.Encoder()[0].Param())
    autoencoder.Optimize(grads, optim)
    print(autoencoder.Encoder()[0].Param())
    """
    



main()
