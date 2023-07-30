'''
how to use.
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
    def activation_tanh(x):
        """Tanh activation function."""
        return np.tanh(x)
    def activation_relu(x):
        """ReLU activation function."""
        return np.maximum(x, 0)


    @staticmethod
    def sigmoid(x):
        """Swish activation function using sigmoid."""
        return x * Swish.activation_sigmoid(x)
    @staticmethod
    def softmax(x):
        """Swish activation function using softmax."""
        return x * Swish.activation_softmax(x)
    @staticmethod
    def tanh(x):
        """Swish activation function using tanh."""
        return x * Swish.activation_tanh(x)
    @staticmethod
    def relu(x):
        """Swish activation function using relu."""
        return x * Swish.activation_relu(x)

''''
TO DO:
- Softsquare: `return np.power(np.clip(x, 0, 1), 2)`
- Bent Identity 2: `return (np.sqrt(x * x + 1) - 1) / 2 + x`
- HardTanh2: `return np.clip(x, -2, 2)`
- LogLogLog: `return np.log(np.log(np.log(x + 1) + 1) + 1)`
- Sigmoid-1: `return x * (1 / (1 + np.exp(-x)))`
- Inverse Square Root Cubic Unit (ISRCU): `return np.where(x >= 0, x, x / np.cbrt(1 + alpha * np.power(x, 3)))` (where `alpha` is a hyperparameter)
- SoftCube: `return np.power(np.clip(x, -1, 1), 3)`
- Gaussian 2: `return np.exp(-np.power(x, 2) / 2)`
- Softplus 2: `return np.log(1 + np.exp(x))`
- Sinusoid 2: `return np.sin(2*x)`
- Gaussian Error Linear Unit 2 (GELU2): `return x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) / 2`
- HardTanh: `return np.clip(x, -1, 1)`
- LogSoftmax: `return np.log(softmax(x))`
- SoftExponential2: `return np.where(x < 0, -np.log(1 - alpha * (x + alpha)) / alpha, np.exp(alpha * x) - 1) / alpha`
- Inverse Square Root Exponential Linear Unit (ISRELU): `return np.where(x >= 0, x, x / np.sqrt(1 + alpha * np.power(x, 2)))` (where `alpha` is a hyperparameter)
- Softmin: `return np.exp(-x) / np.sum(np.exp(-x), axis=0)`
- HardSigmoid: `return np.clip((x + 1) / 2, 0, 1)`
- LogSigmoid: `return np.log(1 / (1 + np.exp(-x)))`
- Cube Root: `return np.power(x, 1/3)`
- Inverse Tangent: `return np.arctan(1 / x)`
- Sinh: `return np.sinh(x)`
- ReLU (Rectified Linear Unit): return x * max(0, x)
- Leaky ReLU: return x * max(0.01 * x, x)
- Tanh (Hyperbolic tangent):return x * np.tanh(x)
- ELU (Exponential Linear Unit):return x if x >= 0 else np.exp(x) - 1
- Parametric ReLU (PReLU): return x * max(alpha * x, x) #(where `alpha` is a learnable parameter)
- Exponential Linear Unit (ELU): return x if x >= 0 else alpha * (np.exp(x) - 1)
- Softplus: return x * np.log(1 + np.exp(x))
- Gaussian Error Linear Unit (GELU):return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
- Swish-1:return x * sigmoid(beta * x)
- Mish: return x * np.tanh(np.log(1 + np.exp(x)))
- Softsign: `return x / (1 + np.abs(x))`
- ArcTan: `return np.arctan(x)`
- Bent Identity: `return (np.sqrt(x*x + 1) - 1) / 2 + x`- Cube: `return np.power(x, 3)`
- Sinusoid: `return np.sin(x)`
- Inverse Square Root Unit (ISRU): `return x / np.sqrt(1 + alpha * np.power(x, 2))` (where `alpha` is a hyperparameter)
- SoftExponential: `return np.where(x < 0, -np.log(1 - alpha * (x + alpha)) / alpha, x)`
- SineReLU: `return np.where(x > 0, np.sin(x), alpha * np.exp(x) - alpha)`
- Gaussian: `return np.exp(-np.power(x, 2))`
- ISRLU (Inverse Square Root Linear Unit): `return np.where(x > 0, x, x / np.sqrt(1 + alpha * np.power(x, 2)))` (where `alpha` is a hyperparameter)
- Bent Identity: `return (np.sqrt(x * x + 1) - 1) / 2 + x`
- LogLog: `return np.log(np.log(x + 1) + 1)`
- SoftClip: `return np.clip(x, -alpha, alpha)`- HardSwish: `return x * np.clip(x + 3, 0, 6) / 6`
- Symmetric Sigmoid: `return 2 / (1 + np.exp(-x)) - 1`
- SoftExponential: `return np.where(x < 0, -np.log(1 - alpha * (x + alpha)) / alpha, x)`
- Gaussian Error Linear Unit (GELU): `return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))`
