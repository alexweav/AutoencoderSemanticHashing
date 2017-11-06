import numpy as np

def numerical_gradient(f, x, accuracy=1e-5):
    grads = np.zeros(x.shape)
    iterator = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not iterator.finished:
        index = iterator.multi_index
        original = x[index]
        x[index] += accuracy
        upper_bound = f(x)
        x[index] = original
        x[index] -= accuracy
        lower_bound = f(x)
        x[index] = original
        grads[index] = (upper_bound - lower_bound) / (2*accuracy)
        iterator.iternext()
    return grads

def numerical_gradient_layer(f, x, dout, accuracy=1e-5):
    grads = np.zeros(x.shape)
    iterator = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not iterator.finished:
        index = iterator.multi_index
        original = x[index]
        x[index] += accuracy
        upper_bound = f(x).copy()
        x[index] = original
        x[index] -= accuracy
        lower_bound = f(x).copy()
        x[index] = original
        grads[index] = np.sum((upper_bound - lower_bound) * dout) / (2*accuracy)
        iterator.iternext()
    return grads

def relative_error(result, expected):
    return np.abs(result - expected) / np.maximum(np.abs(result) + np.abs(expected), 1e-8)
