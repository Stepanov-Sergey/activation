'''
ReLU (Rectified Linear Unit) функция активации

Описание функции:

Функция принимает входное значение x и возвращает максимум между 0 и x.
Если x положительное число, то ReLU возвращает x, в противном случае возвращает 0.
Преимущества функции ReLU включают в себя простоту вычисления и отсутствие проблемы затухающего градиента,
которая может возникнуть при использовании других функций активации, таких как сигмоида или тангенс.
Функция ReLU широко применяется в глубоком обучении, поскольку способствует разрежению активаций,
что помогает обеспечить более эффективное вычисление и сокращение времени обучения модели.
'''
import numpy as np

def relu(x):
    return np.maximum(0, x)