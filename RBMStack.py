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

    def Depth(self):
        return len(self.stack)

    def Cycle(self, initial_state):
        return self.CycleDepth(initial_state, self.Depth())

    def CycleDepth(self, initial_state, depth):
        return self.stack[0].Sample(self.CycleContinuous(initial_state, depth))

    def CycleContinuous(self, initial_state, depth):
        data = initial_state
        for layer_index in range(depth):
            rbm = self.stack[layer_index]
            data = rbm.Sample(rbm.PropHidden(data))
        for layer_index in reversed(range(1, depth)):
            rbm = self.stack[layer_index]
            data = rbm.Sample(rbm.PropVisible(data))
        rbm = self.stack[0]
        return rbm.PropVisible(data)

