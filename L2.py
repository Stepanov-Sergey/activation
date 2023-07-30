'''
Функция L2 (Евклидово расстояние):
point1 = [1, 2, 3]
point2 = [4, 5, 6]
l2_dist = l2_distance(point1, point2)
printL2 distance:", l2_dist```

Обратите внимание, что эти функции предполагают, что входные данные `x` и `y` являются списками одинаковой длины, представляющими координаты точек в n-мерном пространстве.
'''

python
import math

def l2_distance(x, y):
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i]) 2
    distance = math.sqrt(distance)
    return distance