from RBM import *

class RBMStack(object):

    def __init__(self, architecture):
        self.stack = []
        previous_layer_size = architecture[0]
        if len(architecture) > 1:
            for layer_size in architecture[1:]:
                self.stack.append(RBM(previous_layer_size, layer_size))
                previous_layer_size = layer_size

    def Stack(self):
        return self.stack
