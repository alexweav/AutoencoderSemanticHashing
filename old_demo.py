import numpy as np
import _pickle as cPickle
import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

in_dim = 28*28
target_dim = 32
hidden_size = 500

learning_rate = 7e-5
beta1 = 0.99
beta2 = 0.99
corruption_std = 3e-2

num_steps = 2000
num_evals = 100
batch_size = 100
show_every = 500

def main():
    data = load_mnist()
    model = init_model()
    adam_cache = {}

    for step in range(num_steps):
        batch, indices = select_batch(data['train_X'], batch_size)
        corrupted_batch = add_gaussian_noise(batch, corruption_std)
        code, h1_act = encoder_forward_pass(corrupted_batch, model)
        reconstruction, h2_act = reconstruction_forward_pass(code, model)
        loss, d_loss = mse_loss(batch, reconstruction)
        print("Step ", step, " total loss ", np.sum(loss))
        grads = backprop(model, d_loss, reconstruction, h2_act, code, h1_act, corrupted_batch)
        model, adam_cache = update(model, grads, adam_cache)
        if step % show_every == 0:
            disc = discretize_activation(code)
            print(disc[0:1])
            plot_dual(corrupted_batch[0:1], reconstruction[0:1])

    for step in range(num_evals):
        batch, index = select_batch(data['test_X'], 1)
        code, _ = encoder_forward_pass(batch, model)
        reconstruction, _ = reconstruction_forward_pass(code, model)
        print(code)
        disc = discretize_activation(code)
        print(disc)
        code = hash_code(disc)
        print(code)
        plot_dual(batch, reconstruction)

    
"""
Loads the MNIST set into a dictionary
"""
def load_mnist():
    data = {}
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()
    data['train_X'], data['train_y'] = train_set
    data['valid_X'], data['valid_y'] = valid_set
    data['test_X'], data['test_y'] = test_set
    return data

"""
Plots a 28x28 image stored in a numpy array
Blocks until the plot is closed
"""
def plot(img):
    img = img.reshape(28, 28)
    imgplot = plt.imshow(img, cmap='Greys')
    pylab.show()

"""
Plots two 28x28 images, side by side, each stored in a numpy array
Blocks until the plot is closed
"""
def plot_dual(img1, img2):
    img1 = img1.reshape(28, 28)
    img2 = img2.reshape(28, 28)
    plot_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(plot_image, cmap='Greys')
    pylab.show()

"""
Produces a mirrored-architecture fully connected autoencoder
One hidden layer in each of the encoder and decoder halves
"""
def init_model():
    model = {}
    model['W1'] = np.random.randn(in_dim, hidden_size)/np.sqrt(in_dim) 
    model['b1'] = np.zeros(hidden_size)
    model['W2'] = np.random.randn(hidden_size, target_dim)/np.sqrt(hidden_size)
    model['b2'] = np.zeros(target_dim)
    model['W3'] = np.random.randn(target_dim, hidden_size)/np.sqrt(target_dim)
    model['b3'] = np.zeros(hidden_size)
    model['W4'] = np.random.randn(hidden_size, in_dim)/np.sqrt(hidden_size)
    model['b4'] = np.zeros(in_dim)
    return model

"""
Selects a random batch of "count" samples from the given dataset
Sampled from an even distribution without replacement
"""
def select_batch(data, count):
    indices = np.random.choice(data.shape[0], count, replace=False)
    return data[indices], indices

"""
Given a signal, returns the same signal corrupted with gaussian noise
std is the standard deviation of the noise
"""
def add_gaussian_noise(data, std):
    noise = np.random.normal(0, std, data.shape)
    return data + noise

"""
Maps a batch of datapoints to their low dimensional representations through the autoencoder
"""
def encoder_forward_pass(data, model):
    h1_scores = np.dot(data, model['W1']) + model['b1']
    h1_activation = relu(h1_scores)
    encoder_scores = np.dot(h1_activation, model['W2']) + model['b2']
    encoder_activation = relu(encoder_scores)
    return encoder_activation, h1_activation

"""
Maps a batch of low dimensional representations to their reconstructed datapoints
"""
def reconstruction_forward_pass(data, model):
    h2_scores = np.dot(data, model['W3']) + model['b3']
    h2_activation = relu(h2_scores)
    reconstructor_scores = np.dot(h2_activation, model['W4']) + model['b4']
    return reconstructor_scores, h2_activation

"""
A full forward pass through the autoencoder:
Maps a batch of data to their low dimensional representations,
then attempts to reconstruct the data from the representations
"""
def full_forward_pass(data, model):
    encoder_act, h1_act = encoder_forward_pass(data, model)
    reconstruction, h2_act = reconstruction_forward_pass(encoder_act, model)
    return reconstruction, h2_act, encoder_act, h1_act

"""
Rectified linear activation
"""
def relu(x):
    x[x<0] = 0
    return x

"""
Mean squared error loss for a batch
Returns the loss values for every item in the batch
and the derivative of the loss function for every datapoint in the reconstruction
"""
def mse_loss(image, reconstruction):
    loss = 0.5*np.sum((reconstruction-image)**2, axis=1)
    d_loss = reconstruction-image
    return loss, d_loss

"""
Standard backpropagation of gradients
Returns a dictionary with the same keys/array sizes as the model
containing derivatives of loss with respect to corresponding model parameters
"""
def backprop(model, d_loss, reconstruction, h2_act, code_act, h1_act, image):
    d_params = {}
    N, D = reconstruction.shape
    d_params['b4'] = np.sum(d_loss, axis=0)
    d_params['W4'] = np.dot(h2_act.T, d_loss)
    d_h2_act = np.dot(d_loss, model['W4'].T)
    d_h2_act[h2_act <= 0] = 0.0
    d_params['b3'] = np.sum(d_h2_act, axis=0)
    d_params['W3'] = np.dot(code_act.T, d_h2_act)
    d_code_act = np.dot(d_h2_act, model['W3'].T)
    d_code_act[code_act <= 0] = 0.0
    d_params['b2'] = np.sum(d_code_act, axis=0)
    d_params['W2'] = np.dot(h1_act.T, d_code_act)
    d_h1_act = np.dot(d_code_act, model['W2'].T)
    d_h1_act[h1_act <= 0] = 0.0
    d_params['b1'] = np.sum(d_h1_act, axis=0)
    d_params['W1'] = np.dot(image.T, d_h1_act)
    return d_params

"""
Performs a step of stochastic gradient descent given a model and its loss derivatives
"""
def update(model, model_derivatives, cache):
    for key, value in model.items():
        subcache = cache.get(key, {})
        model[key], subcache = adam(model[key], model_derivatives[key], subcache)
        cache[key] = subcache
    return model, cache

"""
Single step adaptive moment optimization of a single matrix,
along with its first order derivatives
theta is the matrix to be optimized
"""
def adam(theta, dtheta, cache):
    m = cache.get('m', np.zeros_like(theta))
    v = cache.get('v', np.zeros_like(theta))
    t = cache.get('t', 0)
    t += 1
    m = beta1 * m + (1 - beta1) * dtheta
    v = beta2 * v + (1 - beta2) * dtheta**2
    corrected_m = m/(1 - beta1**t)
    corrected_v = v/(1 - beta2**t)
    cache['t'] = t
    cache['m'] = m
    cache['v'] = v
    return theta + (-learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)), cache
    
"""
Discretizes a layer's activation into a signal representing whether or not a unit fired
"""
def discretize_activation(activation):
    activation[activation <= 0] = 0
    activation[activation > 0] = 1
    return activation

"""
Obtains an autoencoded hash code for a sample,
given the encoder's activation pattern for that sample
"""
def hash_code(activation):
    code = 0
    discrete_activations = discretize_activation(activation)
    for signal in discrete_activations[0]:
        if signal == 1.0:
            code += 1
        code *= 2
    return code
    

    
main()
