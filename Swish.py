'''
output_sigmoid = swish.sigmoid(x)
output_softmax = swish.softmax(x)
'''
import numpy as np

class Swish:
    @staticmethod
    def activation_sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_softmax(x):
        """Softmax activation function."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    @staticmethod
    def sigmoid(x):
        """Swish activation function using sigmoid."""
        return x * Swish.activation_sigmoid(x)

    @staticmethod
    def softmax(x):
        """Swish activation function using softmax."""
        return x * Swish.activation_softmax(x)
        
'''
### !!!!!!!!!!!!
### Обратите внимание, что в функции swish мы используем сигмоидную функцию активации, но Swish может работать с другими функциями активации, такими как ReLU. Вы можете использовать другие функции активации, если необходимо. Кроме того, для повышения производительности такие функции могут быть реализованы с использованием библиотек для машинного обучения, таких как TensorFlow или PyTorch.
### !!!!!!!!!!!!

import numpy as np

def swish(x):
    """Функция активации Swish."""
    return x * sigmoid(x)

def sigmoid(x):
    """Функция активации Sigmoid."""
    return 1 / (1 + np.exp(-x))

# Пример использования функции активации Swish
x = np.array([-2, -1, 0, 1, 2])
output = swish(x)
print(output)


В этом примере мы определяем функцию swish, которая применяет функцию активации Swish к входному массиву x. Эта функция умножает x на значениe сигмоидной функции sigmoid. Затем, мы определяем функцию sigmoid, которая реализует сигмоидную функцию активации. Затем мы передаем входной массив -2, -1, 0, 1, 2 и печатаем результат.

'''
