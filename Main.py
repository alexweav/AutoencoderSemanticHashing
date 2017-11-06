from Layers import *

def main():
    bias = Bias(4)
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(bias.Forward(a))
    dout = np.ones((2, 4))
    print(bias.Backward(a, dout))
    mm = MatMul(4, 5)
    print(mm.weights)
    print(mm.Forward(a))


main()
