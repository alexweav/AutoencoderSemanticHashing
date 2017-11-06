from NumericalGradient import *
from Layers import *

def main():
    CheckAllLayers()

def CheckAllLayers():
    CheckBias()
    CheckMatMul()
    CheckReLU()
    CheckSigmoid()

def CheckBias():
    bias = Bias(10)
    test_input = np.random.randn(50, 10)
    dout = np.random.randn(50, 10)
    dx_num = numerical_gradient_layer(lambda x : bias.Forward(x), test_input, dout)
    dx, db = bias.Backward(test_input, dout)
    print('Bias dx error:', np.max(relative_error(dx, dx_num)))

def CheckMatMul():
    mm = MatMul(30, 10)
    test_input = np.random.randn(50, 30)
    dout = np.random.randn(50, 10)
    dx_num = numerical_gradient_layer(lambda x : mm.Forward(x), test_input, dout)
    dx, dW = mm.Backward(test_input, dout)
    print('MatMul dx error:', np.max(relative_error(dx, dx_num)))

def CheckReLU():
    relu = ReLU()
    test_input = np.random.randn(50, 30)
    dout = np.random.randn(50, 30)
    dx_num = numerical_gradient_layer(lambda x : relu.Forward(x), test_input, dout)
    dx = relu.Backward(test_input, dout)
    print('ReLU dx error:', np.max(relative_error(dx, dx_num)))

def CheckSigmoid():
    sig = Sigmoid()
    test_input = np.random.randn(50, 30)
    dout = np.random.randn(50, 30)
    dx_num = numerical_gradient_layer(lambda x : sig.Forward(x), test_input, dout)
    dx = sig.Backward(test_input, dout)
    print('Sigmoid dx error:', np.max(relative_error(dx, dx_num)))


main()
