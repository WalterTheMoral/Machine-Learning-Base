from Layer import *
from Layer_Configuration import *

from LearningStrategy import *
from Initialisation import *
from Activation import *

np.random.seed(1)
configs = [
    LayerConfiguration("Hidden 1", 6, (4000,), Relu(), Zero()),
    LayerConfiguration("Hidden 2", 12, (6,), LeakyRelu(), Random(), Adaptive(), 0.5),
    LayerConfiguration("Neurons 3", 16, (12,), TanH(), Zero()),
    LayerConfiguration("Neurons 4", 3, (16,), Sigmoid(), Random(), Adaptive(), 10.0)
]
l = [Layer(config) for config in configs]

for layer in l:
    print(layer)
    print()
