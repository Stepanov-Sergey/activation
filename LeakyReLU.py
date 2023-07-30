'''Leaky ReLU функция активации:'''
import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)