from Layers import *
from DataLoaders import *
from Autoencoder import *
from Optimizers import *
from RBM import *
from RBMStack import *

import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

num_steps = 2000
learning_rate = 5e-4
batch_size = 128
show_every = 1000
num_evals = 100

print_every = 100

def plot_dual(img1, img2):
    img1 = img1.reshape(28, 28)
    img2 = img2.reshape(28, 28)
    plot_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(plot_image, cmap='Greys')
    pylab.show()

def plot_points(points):
    plt.plot(points)
    plt.ylabel('loss')
    plt.xlabel('step')
    pylab.show()

def select_batch(data, count):
    indices = np.random.choice(data.shape[0], count, replace=False)
    return data[indices], indices

def main():
    data = LoadMNIST()
    adam_cache = {}
    autoencoder = Autoencoder([784, 512, 32])
    optim = AdamOptimizer(learning_rate, 0.95, 0.95)
    losses = []

    stack = RBMStack([784, 512, 256])
    img, _ = SelectBatch(data['train_X'], 1)
    #rbm = RBM(784, 64)
    rbm = stack.Stack()[0]

    plot_dual(img, rbm.Cycle(img))

    for step in range(num_steps):
        batch, _ = SelectBatch(data['train_X'], 128)
        rbm.Train(batch, 0.01)
        if step % print_every == 0:
            print('Step', step, 'completed.')
    old_rbm = stack.Stack()[0]
    rbm = stack.Stack()[1]
    for step in range(num_steps):
        batch, _ = SelectBatch(data['train_X'], 128)
        rbm.Train(old_rbm.Sample(old_rbm.PropHidden(batch)), 0.01)
        if step % print_every == 0:
            print('Step', step, 'completed.')

    autoencoder = RBMAutoencoder(stack)

    for step in range(num_evals):
        img, _ = SelectBatch(data['train_X'], 1)
        recon, _ = autoencoder.EvaluateFull(img)
        plot_dual(img, recon)

    """
    recon, _ = autoencoder.EvaluateFull(img)
    plot_dual(img, recon)
    for step in range(num_steps):
        batch, indices = select_batch(data['train_X'], batch_size)
        reconstruction, inputs = autoencoder.EvaluateFull(batch)
        loss, deriv = autoencoder.MSELoss(batch, reconstruction)
        print('Step', step, 'Loss: ', np.sum(loss))
        losses.append(np.sum(loss))
        grads = autoencoder.Backprop(inputs, deriv)
        adam_cache = autoencoder.Optimize(grads, optim, adam_cache)
        if step % show_every == 0:
            img = data['train_X'][0:1]
            recon, _ = autoencoder.EvaluateFull(img)
            plot_dual(img, recon)
            plot_points(losses)
    plot_points(losses)
    for step in range(num_evals):
        batch, index = select_batch(data['test_X'], 1)
        recon, _ = autoencoder.EvaluateFull(batch)
        plot_dual(batch, recon)
    """







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
