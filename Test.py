import random
import matplotlib.pyplot as plt
from PIL import Image

from LayerClasses.Layer_Configuration import *

from LayerClasses.LearningStrategy import *
from LayerClasses.Initialisation import *
from LayerClasses.Activation import *
from Network import *
from NetworkClasses.Network_Configuration import *
from NetworkClasses.Cost import *

import torch

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
            LayerConfiguration("Perceptron 2", 1, (10,), TrimSigmoid(), Random())
        ]
        model.add(*(Layer(config) for config in configurations))

        X = np.random.randn(12288, 10) * 256
        print("predict:", model.predict(X))

        print(model)

class ShallowNetwork3:
    def test(self):
        from unit10 import c1w3_utils as u10

        X, Y = u10.load_planar_dataset()
        # plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)

        shape_X = X.shape
        shape_Y = Y.shape
        m = shape_X[1]

        # Train the logistic regression classifier
        # clf = sklearn.linear_model.LogisticRegressionCV()
        # clf.fit(X.T, Y[0, :])
        # Plot the decision boundary for logistic regression
        # u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
        # plt.title("Logistic Regression")
        # plt.show()
        # Print accuracy
        # LR_predictions = clf.predict(X.T)
        # print('Accuracy of logistic regression: %d ' % float(
        #     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
        #       '% ' + "(percentage of correctly labelled datapoints)")

        model = Network(
            NetworkConfiguration("Model", CrossEntropy(), 0.5)
        )
        configs = [
            LayerConfiguration("Layer 1", 4, (2,), TanH(), Random(0.01), Standard(0.1)),
            LayerConfiguration("Output Layer", 1, (4,), Sigmoid(), Random(0.01), Standard(0.1))
        ]
        model.add(*(Layer(config) for config in configs))

        np.random.seed(1)
        print(model)

        costs = model.train(X, Y, 10000)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.show()

        u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))
        plt.show()
        predictions = model.predict(X)
        print('Accuracy: %d' % float(
            (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

class DeepNetwork1:
    def test(self):
        from unit10 import c2w1_init_utils as u10

        plt.rcParams['figure.figsize'] = (7.0, 4.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        # load image dataset: blue/red dots in circles
        train_X, train_Y, test_X, test_Y = u10.load_dataset()
        plt.close()

        np.random.seed(1)
        # configs = [
        #     LayerConfiguration("Perceptron 1", 30, (12288,), Relu(), Xaviar(), Standard(0.0075)),
        #     LayerConfiguration("Perceptron 2", 15, (30,), TrimSigmoid(), He(), Standard(0.1))
        # ]
        # hidden1, hidden2 = (Layer(config) for config in configs)
        # print(hidden1)
        # print(hidden2)
        #
        #
        # hidden1 = Layer(LayerConfiguration("Perceptron 1", 10, (10,), Relu(), Xaviar(), Standard(0.0075)))
        # hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
        # hidden1.save_weights("SaveDir", "Hidden1")
        # hidden2 = Layer(LayerConfiguration("Perceptron 2", 10, (10,), TrimSigmoid(), File("SaveDir/Hidden1.h5"), Standard(0.1)))
        # print(hidden1)
        # print(hidden2)
        # m1 = Network(NetworkConfiguration("Model", SquaredMean(), 0.5))
        # m1.add(hidden1, hidden2)
        # dir = "m1"
        # m1.save_weights(dir)
        # print(os.listdir(dir))

        np.random.seed(2)
        configs = [
            LayerConfiguration("Layer 1", 10, (2,), Relu(), He(), Standard(0.01)),
            LayerConfiguration("Layer 2", 5, (10,), Relu(), He(), Standard(0.01)),
            LayerConfiguration("Output", 1, (5,), TrimSigmoid(), He(), Standard(0.1))
        ]
        model = Network(NetworkConfiguration("Model", CrossEntropy(), 0.5))
        model.add(*(Layer(config) for config in configs))

        init = "large random"
        costs = model.train(train_X, train_Y, 15000)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per 150s)')
        plt.title(init + " initialization")
        plt.show()

        plt.title("Model with " + init + " initialization")
        axes = plt.gca()
        axes.set_xlim([-1.5, 1.5])
        axes.set_ylim([-1.5, 1.5])
        u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)

        predictions = model.predict(train_X)
        print('Train accuracy: %d' % float(
            (np.dot(train_Y, predictions.T) + np.dot(1 - train_Y, 1 - predictions.T)) / float(
                train_Y.size) * 100) + '%')
        predictions = model.predict(test_X)
        print('Test accuracy: %d' % float(
            (np.dot(test_Y, predictions.T) + np.dot(1 - test_Y, 1 - predictions.T)) / float(test_Y.size) * 100) + '%')

class DeepNetwork2:
    def test(self):
        from unit10 import c1w4_utils as u10

        train_x_orig, train_y, test_x_orig, test_y, classes = u10.load_datasetC1W4()
        # Example of a picture
        index = 87
        # plt.imshow(train_x_orig[index])
        # plt.show()
        print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

        m_train = train_x_orig.shape[0]
        num_px = train_x_orig[0].shape[0]
        m_test = test_x_orig.shape[0]

        print("Number of training examples: " + str(m_train))
        print("Number of testing examples: " + str(m_test))
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("train_x_orig shape: " + str(train_x_orig.shape))
        print("train_y shape: " + str(train_y.shape))
        print("test_x_orig shape: " + str(test_x_orig.shape))
        print("test_y shape: " + str(test_y.shape))

        # Reshape the training and test examples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        # Standardize data to have feature values between -0.5 and 0.5.
        train_x = train_x_flatten / 255 - 0.5
        test_x = test_x_flatten / 255 - 0.5

        print("train_x's shape: " + str(train_x.shape))
        print("test_x's shape: " + str(test_x.shape))
        print("normelized train color: ", str(train_x[10][10]))
        print("normelized test color: ", str(test_x[10][10]))


        model = Network(NetworkConfiguration("Model", CrossEntropy(), 0.5))
        configs = [
            LayerConfiguration("Layer 1", 7, (12288,), Relu(), Xaviar(), Standard(0.007)),
            LayerConfiguration("Layer 2", 1, (7,), Sigmoid(), Xaviar(), Standard(0.007))
        ]
        model.add(*(Layer(config) for config in configs))

        # costs = model.train(train_x, train_y, 2500)
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per 25s)')
        # plt.title("Learning rate =" + str(0.007))
        # plt.show()
        # print("train accuracy:", np.mean(model.predict(train_x) == train_y))
        # print("test accuracy:", np.mean(model.predict(test_x) == test_y))

        model = Network(NetworkConfiguration("Model", CrossEntropy(), 0.5))
        configs = [
            LayerConfiguration("Layer 1", 30, (train_x.shape[0],), Relu(), Xaviar(), Standard(0.0075)),
            LayerConfiguration("Layer 2", 15, (30,), Relu(), Xaviar(), Standard(0.0075)),
            LayerConfiguration("Layer 3", 10, (15,), Relu(), Xaviar(), Standard(0.0075)),
            LayerConfiguration("Layer 4", 10, (10,), Relu(), Xaviar(), Standard(0.0075)),
            LayerConfiguration("Layer 5", 5, (10,), Relu(), Xaviar(), Standard(0.0075)),
            LayerConfiguration("Layer 6", 1, (5,), TrimSigmoid(), Xaviar(), Standard(0.0075))
        ]
        model.add(*(Layer(config) for config in configs))

        costs = model.train(train_x, train_y, 2500)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 25s)')
        plt.title("Learning rate =" + str(0.007))
        plt.show()
        print("train accuracy:", np.mean(model.predict(train_x) == train_y))
        print("test accuracy:", np.mean(model.predict(test_x) == test_y))

        # Test your image
        img_path = r'C:\Users\smash\Documents\PycharmProjects\Machine Learning Base\Util\cat.jpeg'  # full path of the image
        my_label_y = [0]  # the true class of your image (1 -> cat, 0 -> non-cat)
        img = Image.open(img_path)
        image64 = img.resize((num_px, num_px), Image.Resampling.LANCZOS)
        plt.imshow(img)
        plt.show()
        plt.imshow(image64)
        plt.show()
        my_image = np.reshape(image64, (num_px * num_px * 3, 1))
        my_image = my_image / 255. - 0.5
        p = model.predict(my_image)
        print("L-layer model predicts a \"" + classes[int(p),].decode("utf-8") + "\" picture.")

class Softmax1:
    def test(self):
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt

        # set default size of plots
        plt.rcParams['figure.figsize'] = (5.0, 4.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        # set seed
        np.random.seed(1)

        np.random.seed(1)
        softmax_layer = Layer(
            LayerConfiguration("Softmax 1", 3, (4,), Softmax(), Random())
        )
        A_prev = np.random.randn(4, 5)
        A = softmax_layer.feedforward(A_prev)
        dA = A
        dA_prev = softmax_layer.backward_propagation(dA)
        print("A:\n", A)
        print("dA_prev:\n", dA_prev)

        print("\n" + "*"*80 + "\n")

        np.random.seed(2)
        softmax_layer = Layer(LayerConfiguration("Layer", 3, (4,), Softmax(), Random(), Standard(0.1)))
        print("W before:\n", softmax_layer.W)
        print("b before:\n", softmax_layer.b)
        model = Network(NetworkConfiguration("Model", CategoricalCrossEntropy()))
        model.add(softmax_layer)
        X = np.random.randn(4, 5)
        Y = np.random.rand(3, 5)
        Y = np.where(Y == Y.max(axis=0), 1, 0)
        cost = model.train(X, Y, 1)
        print("cost:", cost[0])
        print("W after:\n", softmax_layer.W)
        print("b after:\n", softmax_layer.b)

        print("\n" + "*"*80 + "\n")

        np.random.seed(3)
        softmax_layer = Layer(LayerConfiguration("Layer1", 3, (4,), Softmax(), Random()))
        model = Network(NetworkConfiguration("Model", CategoricalCrossEntropy()))
        model.add(softmax_layer)
        X = np.random.randn(4, 50000) * 10
        Y = np.zeros((3, 50000))
        sumX = np.sum(X, axis=0)
        for i in range(len(Y[0])):
            if sumX[i] > 5:
                Y[0][i] = 1
            elif sumX[i] < -5:
                Y[2][i] = 1
            else:
                Y[1][i] = 1
        costs = model.train(X, Y, 2)
        plt.plot(costs)
        plt.show()
        predictions = model.predict(X)
        print("right", np.sum(Y.argmax(axis=0) == predictions.argmax(axis=0)))
        print("wrong", np.sum(Y.argmax(axis=0) != predictions.argmax(axis=0)))

class PyTorch:
    def test(self):
        torch.



if __name__ == "__main__":
    Softmax1().test()