import random
import matplotlib.pyplot as plt

from Layer import *
from LayerClasses.Layer_Configuration import *

from LayerClasses.LearningStrategy import *
from LayerClasses.Initialisation import *
from LayerClasses.Activation import *

from Network import *
from NetworkClasses.Network_Configuration import *
from NetworkClasses.Cost import *

class ShallowNetwork1:
    def test(self):
        np.random.seed(1)
        configs = [
            LayerConfiguration("Hidden 1", 6, (4000,), Relu(), Random()),
            LayerConfiguration("Hidden 2", 12, (6,), LeakyRelu(0.1), Random(), Adaptive(0.5)),
            LayerConfiguration("Neurons 3", 16, (12,), TanH(), Random()),
            LayerConfiguration("Neurons 4", 3, (16,), Sigmoid(), Random(10.0), Adaptive(0.2))
        ]
        layers = [Layer(config) for config in configs]

        for layer in layers:
            print(layer)
            print()

        print("\n" + "*"*80 + "\n")

        Z = np.array([[1,-2,3,-4],
                      [-10,20,30,-40]])
        for layer in layers:
            print(layer.activation.calculate(Z))

        print("\n" + "*"*80 + "\n")

        np.random.seed(2)
        m = 3
        X = np.random.randn(4000,m)
        Al = X
        for i, layer in enumerate(layers):
            Al = layer.feedforward(Al)
            print('layer', i," A", str(Al.shape), ":\n", Al)

        print("\n" + "*"*80 + "\n")

        Al = X
        for i, layer in enumerate(layers):
            Al = layer.feedforward(Al)
            dZ = layer.activation.gradient(layer._Z) * Al
            print('layer',i," dZ", str(dZ.shape), ":\n", dZ)

        print("\n" + "*"*80 + "\n")

        Al = X
        for layer in layers:
            Al = layer.feedforward(Al)

        np.random.seed(3)
        fig, axes = plt.subplots(1, 4, figsize=(12, 16))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        dAl = np.random.randn(Al.shape[0], m) * np.random.randint(-100, 100, Al.shape)
        print(dAl)
        for i, layer in enumerate(layers[::-1]):
            i = 4-i
            axes[i - 1].hist(dAl.reshape(-1), align='left')
            axes[i - 1].set_title('dAl[' + str(i) + ']')
            dAl = layer.backward_propagation(dAl)
        plt.show()

        print("\n" + "*"*80 + "\n")

        np.random.seed(4)
        random.seed(4)

        configs = [
            LayerConfiguration("Hidden1", 3, (4,), TrimSigmoid(), Zero(), Adaptive(0.2)),
            LayerConfiguration("Hidden2", 2, (3,), Relu(), Random(), Standard(1.5))
        ]
        l1, l2 = (Layer(config) for config in configs)

        print("before update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
        print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

        l1.dW = np.random.randn(3,4) * random.randrange(-100,100)
        l1.db = np.random.randn(3,1) * random.randrange(-100,100)
        l2.dW = np.random.randn(2,3) * random.randrange(-100,100)
        l2.db = np.random.randn(2,1) * random.randrange(-100,100)
        l1.update_parameters()
        l2.update_parameters()
        print("after update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
        print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

class ShallowNetwork2:
    def test(self):
        np.random.seed(1)
        m1 = Network(
            NetworkConfiguration("Model 1", CrossEntropy())
        )
        AL = np.random.rand(4, 3)
        Y = np.random.rand(4, 3) > 0.7
        errors = m1.cost.calculate(AL, Y)
        dAL = m1.cost.gradient(AL, Y)
        print("cross entropy error:\n", errors)
        print("cross entropy dAL:\n", dAL)
        m2 = Network(
            NetworkConfiguration("Model 2", SquaredMean())
        )
        errors = m2.cost.calculate(AL, Y)
        dAL = m2.cost.gradient(AL, Y)
        print("squared means error:\n", errors)
        print("squared means dAL:\n", dAL)

        print("\n" + "*"*80 + "\n")

        print("cost m1:", m1.cost.compute_cost(AL, Y))
        print("cost m2:", m2.cost.compute_cost(AL, Y))

        np.random.seed(1)
        model = Network( NetworkConfiguration("Model", CrossEntropy(), 0.7) )
        configurations = [
            LayerConfiguration("Perceptron 1", 10, (12288,), Relu(), Random()),
            LayerConfiguration("Perceptron 1", 1, (10,), TrimSigmoid(), Random())
        ]
        model.add(*(Layer(config) for config in configurations))

        X = np.random.randn(12288, 10) * 256
        print("predict:", model.predict(X))

        print(model)


if __name__ == "__main__":
    ShallowNetwork1().test()