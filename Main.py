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

num_steps = 8000
learning_rate = 12e-5
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
    autoencoder = Autoencoder([784, 512, 64])
    optim = AdamOptimizer(learning_rate, 0.95, 0.95)
    losses = []

    stack = RBMStack([784, 512, 64])
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
    losses = []
    for step in range(num_steps):
        batch, _ = SelectBatch(data['train_X'], 128)
        reconstruction, inputs = autoencoder.EvaluateFull(batch)
        loss, deriv = autoencoder.MSELoss(batch, reconstruction)
        losses.append(np.sum(loss))
        if step % print_every == 0:
            print('Step', step, 'completed. Avg loss: ', np.mean(np.array(losses)))
            losses = []
        grads = autoencoder.Backprop(inputs, deriv)
        adam_cache = autoencoder.Optimize(grads, optim, adam_cache)

    """
    for step in range(num_evals):
        img, _ = SelectBatch(data['train_X'], 1)
        recon, _ = autoencoder.EvaluateFull(img)
        #recon = stack.CycleContinuous(img, stack.Depth())
        #recon = rbm.CycleContinuous(img)
        plot_dual(img, recon)
    """

    fullimage = []
    for col_step in range(10):
        column = []
        for row_step in range(10):
            img, _ = SelectBatch(data['train_X'], 1)
            recon, _ = autoencoder.EvaluateFull(img)
            img1 = img.reshape(28, 28)
            img2 = recon.reshape(28, 28)
            #disp = np.concatenate((img1, img2), axis=1)
            disp = np.hstack([img1, img2])
            column.append(disp)
        fullimage.append(np.vstack(column))
    plt.imshow(np.hstack(fullimage), cmap='Greys')
    pylab.show()





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

main()
