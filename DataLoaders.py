import _pickle as cPickle
import gzip
import numpy as np

def LoadMNIST():
    data = {}
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()
    data['train_X'], data['train_y'] = train_set
    data['valid_X'], data['valid_y'] = valid_set
    data['test_X'], data['test_y'] = test_set
    return data

def SelectBatch(data, count):
    indices = np.random.choice(data.shape[0], count, replace=False)
    return data[indices], indices



