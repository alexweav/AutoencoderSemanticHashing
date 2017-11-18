from DataLoaders import *
import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import numpy as np

num_steps = 3000
num_evals = 100
eval_avgs = 30
learning_rate = 0.01

def main():
    data = LoadMNIST()

    weights = np.random.randn(784, 256)/np.sqrt(784)
    bias_visible = np.zeros(784)
    bias_hidden = np.zeros(256)

    #img, _ = select_batch(data['train_X'], 1)
    for step in range(num_steps):
        img, _ = select_batch(data['train_X'], 128)
        hidden = hidden_step(img, weights, bias_hidden)
        visible = visible_step(hidden, weights, bias_visible)
        new_weights = weights + learning_rate*(np.dot(img.T, hidden_step(img, weights, bias_hidden)) - np.dot(visible.T, hidden_step(visible, weights, bias_hidden)))
        new_bias_hidden = bias_hidden + np.sum(learning_rate*(hidden_step(img, weights, bias_hidden) - hidden_step(visible, weights, bias_hidden)), axis=0)
        new_bias_visible = bias_visible + np.sum(learning_rate*(img - visible), axis=0)
        weights = new_weights
        bias_hidden = new_bias_hidden
        bias_visible = new_bias_visible
        print(step)
    for step in range(num_evals):
        test_img, _ = select_batch(data['train_X'], 1)
        samples = []
        for avgstep in range(eval_avgs):
            hidden = hidden_step(test_img, weights, bias_hidden)
            print(hidden.shape)
            print(weights.shape)
            print(bias_hidden.shape)
            visible = visible_step(hidden, weights, bias_visible)
            samples.append(visible)
        samples = np.array(samples)
        print(samples.shape)
        recon = np.mean(samples, axis=0)
        plot_dual(test_img[0:1], recon)

def select_batch(data, count):
    indices = np.random.choice(data.shape[0], count, replace=False)
    return data[indices], indices

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def binary_stochastic(x):
    pad = np.random.uniform(size=x.shape)
    x[pad <= x] = 1.
    x[pad > x] = 0.
    return x

def hidden_step(x, weights, bias_hidden):
    return binary_stochastic(sigmoid(np.dot(x, weights) + bias_hidden))

def visible_step(hidden, weights, bias_visible):
    return binary_stochastic(sigmoid(np.dot(hidden, weights.T) + bias_visible))

def plot_dual(img1, img2):
    img1 = img1.reshape(28, 28)
    img2 = img2.reshape(28, 28)
    plot_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(plot_image, cmap='Greys')
    pylab.show()

main()

