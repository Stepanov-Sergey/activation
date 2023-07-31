'''
Класс созвращает:
- статус проверки типа данных для вызываемой функции, 
- название функции
- результат вычисления
Применение:
Functions.НазваниеФункции(Данные)
'''
import numpy as np
'''
декоратор получает название функции с результатом ее вычисления

'''
def get_function_name(func):
    def wrapper(*args, **kwargs):
        return func.__name__, *func(*args, **kwargs)
    return wrapper

class Functions:
    @staticmethod
    @get_function_name
    def SwishTanh(x):
        if isinstance(x, (int, float)):
            return x * np.tanh(x), 200
        else:
            return 0, 404
    @staticmethod
    @get_function_name
    def SwishLogSigmoid(x):
        if isinstance(x, np.float64):
            return x * np.log(1 / (1 + np.exp(-x))), 200
        elif isinstance(x, np.ndarray):
            return x * np.log(1 / (1 + np.exp(-x))), 200
        else:
            return 0, 404
    @staticmethod
    @get_function_name
    def LogSigmoid(x):
        if isinstance(x, np.float64):
            return np.log(1 / (1 + np.exp(-x))), 200
        elif isinstance(x, np.ndarray):
            return np.log(1 / (1 + np.exp(-x))), 200
        else:
            return 0, 404

name5, result5, code5 = Functions.SwishTanh(0.5)
print(code5, name5, result5)  # Выводит "Tanh"

name6, result6,code6 = Functions.SwishLogSigmoid(np.array([np.float64(1), np.float64(2), np.float64(3)]))
print(code6, name6, result6)  # Выводит "LogSigmoid"

name7, result7,code7 = Functions.LogSigmoid(np.array([np.float64(1), np.float64(2), np.float64(3)]))
print(code7, name7, result7)  # Выводит "LogSigmoid"
