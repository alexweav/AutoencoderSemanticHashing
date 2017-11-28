from DataLoaders import *
import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import numpy as np

num_steps = 2500
num_evals = 100
eval_avgs = 30
learning_rate = 0.01

def main():
    data = LoadMNIST()

    weights1 = np.random.randn(784, 512)/np.sqrt(784)
    weights2 = np.random.randn(512, 256)/np.sqrt(512)
    bias_visible = np.zeros(784)
    bias_center1 = np.zeros(512)
    bias_center2 = np.zeros(512)
    bias_hidden = np.zeros(256)

    #img, _ = select_batch(data['train_X'], 1)
    for step in range(num_steps):
        img, _ = select_batch(data['train_X'], 128)
        hidden = hidden_step(img, weights1, bias_center1)
        visible = visible_step(hidden, weights1, bias_visible)
        hidden_second = hidden_step(visible, weights1, bias_center1)
        new_weights = weights1 + learning_rate*(np.dot(img.T, hidden) - np.dot(visible.T, hidden_second))
        new_bias_hidden = bias_center1 + np.sum(learning_rate*(hidden - hidden_second), axis=0)
        new_bias_visible = bias_visible + np.sum(learning_rate*(img - visible), axis=0)
        weights1 = new_weights
        bias_center1 = new_bias_hidden
        bias_visible = new_bias_visible
        print(step)
    for step in range(num_steps):
        img, _ = select_batch(data['train_X'], 128)
        center = hidden_step(img, weights1, bias_center1)
        hidden = hidden_step(center, weights2, bias_hidden)
        center_back = visible_step(hidden, weights2, bias_center2)
        hidden_second = hidden_step(center_back, weights2, bias_hidden)
        weights2 = weights2 + learning_rate*(np.dot(center.T, hidden) - np.dot(center_back.T, hidden_second))
        bias_hidden = bias_hidden + np.sum(learning_rate*(hidden - hidden_second), axis=0)
        bias_center2 = bias_center2 + np.sum(learning_rate*(center - center_back), axis=0)

        print(step)
    for step in range(num_evals):
        test_img, _ = select_batch(data['train_X'], 1)
        samples = []
        for avgstep in range(eval_avgs):
            center = hidden_step(test_img, weights1, bias_center1)
            hidden = hidden_step(center, weights2, bias_hidden)
            center_back = visible_step(hidden, weights2, bias_center2)
            visible = visible_step(center_back, weights1, bias_visible)
            samples.append(visible)
        samples = np.array(samples)
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

