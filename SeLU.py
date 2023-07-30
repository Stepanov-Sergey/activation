'''
import numpy as np

def selu(x, alpha=1.67326, scale=1.0507):
    """Функция активации SELU."""
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Пример использования функции активации SELU
x = np.array([-2, -1, 0, 1, 2])
output = selu(x)
print(output)

'''
def selu(x, alpha=1.67326, scale=1.0507):
    """Функция активации SELU."""
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))